"""Reusable original-frame lumen segmentation and diameter video renderer."""
import csv
import hashlib
import html
import json
import math
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from _runtime import configure
from calibration import detect_tick_calibration
from diameter_quality import SPATIAL_SUPPORT_GATE_VERSION, apply_spatial_support_gate
from measurement import measure_mask

ROOT = Path(__file__).resolve().parents[1]


def fit_line(value, font, max_width):
    if font.getlength(value) <= max_width:
        return value
    while value and font.getlength(value + "…") > max_width:
        value = value[:-1]
    return value + "…" if value else ""


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_video(path, expected, fps, dimensions):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open encoded video: {path}")
    actual_fps, count = cap.get(cv2.CAP_PROP_FPS), 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if (frame.shape[1], frame.shape[0]) != dimensions:
                raise RuntimeError("Encoded frame dimensions differ")
            count += 1
    finally:
        cap.release()
    if count != expected or abs(actual_fps - fps) > .001:
        raise RuntimeError(f"Encoded video verification failed: {count}/{expected}, fps={actual_fps}")
    return {"decoded_frames": count, "fps": actual_fps, "width": dimensions[0],
            "height": dimensions[1], "duration_seconds": count / actual_fps, "verified": True}


def _source_info(source):
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"Cannot decode source video: {source}")
    info = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)), "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
    cap.release()
    if min(info["width"], info["height"], info["frames"]) <= 0 or not math.isfinite(info["fps"]) or info["fps"] <= 0:
        raise ValueError("Source must have valid dimensions, frame rate and frame count")
    checks = {}
    manifest_path = ROOT / "data/manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        match = next((v for v in manifest["videos"] if Path(v["source_path"]).resolve() == source), None)
        if match:
            info["frames"] = match["decoded_frame_count"]
            sample_indices = {0, info["frames"] // 2, info["frames"] - 1}
            checks = {f["frame_index"]: ROOT / f["image_path"] for f in manifest["frames"]
                      if f["video_id"] == match["video_id"] and f["frame_index"] in sample_indices}
    return info, checks


def _draw_overlay(original, mask, measurement):
    overlay = original.copy()
    overlay[mask] = (.91 * overlay[mask] + .09 * np.array([66, 174, 255])).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (66, 174, 255), 1, cv2.LINE_AA)
    center = measurement.get("centerline", []) if measurement.get("status") == "ok" else []
    if len(center) > 1:
        cv2.polylines(overlay, [np.rint(center).astype(np.int32)], False, (190, 216, 54), 1, cv2.LINE_AA)
    lines = measurement.get("scanlines", []) if measurement.get("status") == "ok" else []
    for line in lines[::max(1, len(lines) // 9)]:
        cv2.line(overlay, (round(line["x1"]), round(line["y1"])),
                 (round(line["x2"]), round(line["y2"])), (255, 210, 65), 1, cv2.LINE_AA)
    for line in measurement.get("rejected_scanlines", [])[::3]:
        if "x" in line and "y" in line:
            cv2.circle(overlay, (round(line["x"]), round(line["y"])), 2, (120, 120, 120), -1)
    return overlay


def run_video(source, weights, output, tick_mm=5.0, mm_per_pixel=None, roi=None,
              conf=.25, device="0", max_frames=None, progress_path=None, min_axis_coverage=.35):
    """Preserve source frames; display independent valid diameters without FMD.

    Physical scale is supplied explicitly or verified from visible ticks. Missing
    boundaries and unavailable calibration are never replaced with invented values.
    The output directory must be new. Inputs and live annotations are read only.
    """
    source, weights, output = Path(source).resolve(), Path(weights).resolve(), Path(output).resolve()
    if not source.is_file() or not weights.is_file():
        raise FileNotFoundError("Source video or model checkpoint is missing")
    if not math.isfinite(float(tick_mm)) or float(tick_mm) <= 0:
        raise ValueError("tick_mm must be finite and positive")
    if mm_per_pixel is not None and (not math.isfinite(float(mm_per_pixel)) or float(mm_per_pixel) <= 0):
        raise ValueError("mm_per_pixel must be finite and positive")
    if not 0 < float(conf) <= 1:
        raise ValueError("Confidence must be greater than zero and at most one")
    if roi is not None and (len(roi) != 4 or any(not math.isfinite(float(v)) for v in roi)
                            or roi[2] <= roi[0] or roi[3] <= roi[1]):
        raise ValueError("ROI must be x1,y1,x2,y2 with positive area")
    if max_frames is not None and int(max_frames) <= 0:
        raise ValueError("max_frames must be positive")
    if not math.isfinite(float(min_axis_coverage)) or not 0 <= min_axis_coverage <= 1:
        raise ValueError("min_axis_coverage must be between zero and one")
    info, source_checks = _source_info(source)
    width, height, fps, total = info["width"], info["height"], info["fps"], info["frames"]
    # Known acquisitions have a reviewed ultrasound viewport. A new source must
    # not inherit those screen coordinates merely because its dimensions match.
    effective_roi = roi if roi is not None else ([94, 40, 545, 382] if source_checks and (width, height) == (640, 448)
                                                else [0, 0, width, height])
    target = min(total, int(max_frames)) if max_frames is not None else total
    partial = target < total
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / ".diameter_pipeline_output.json", {"purpose": "diameter_video_output", "source": str(source)})
    progress_path = Path(progress_path) if progress_path else output / "progress.json"
    comparison = output / f"{source.stem}_D_비교.mp4"
    standalone = output / f"{source.stem}_D_표시.mp4"
    csv_path = output / f"{source.stem}_D_값.csv"
    pane_width, header_height = width + width % 2, 64
    screen_height, screen_width = height + header_height + height % 2, pane_width * 2
    rate = Fraction(str(fps)).limit_denominator(100000)
    options = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an"]
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
               "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{screen_width}x{screen_height}",
               "-framerate", f"{rate.numerator}/{rate.denominator}", "-i", "pipe:0",
               "-filter_complex", f"[0:v]split=2[comparison][right];[right]crop={pane_width}:{screen_height}:{pane_width}:0[diameter]",
               "-map", "[comparison]", *options, str(comparison), "-map", "[diameter]", *options, str(standalone)]
    configure()
    from ultralytics import YOLO
    model = YOLO(str(weights), task="segment")
    if model.names != {0: "lumen"}:
        raise ValueError("Use a trained one-class lumen segmentation checkpoint")
    fonts = {size: ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", size) for size in (15, 19, 28)}
    base = Image.new("RGB", (screen_width, header_height), (15, 23, 32))
    draw = ImageDraw.Draw(base)
    short_name = source.name if len(source.name) < 40 else source.name[:37] + "…"
    draw.text((12, 3), fit_line("원본 · " + short_name, fonts[19], pane_width - 24), font=fonts[19], fill="white")
    draw.text((pane_width + 12, 3), fit_line("내강 경계 · 직각 측정", fonts[19], pane_width - 232), font=fonts[19], fill=(255, 174, 66))
    started = time.monotonic()
    processed = measured = masks_empty = total_lines = total_excluded = support_rejected = 0
    records, snapshots = [], []
    sample_indices = {0, min(116, target - 1), target // 4, target // 2, 3 * target // 4, target - 1}
    scale_counts, scale_values, methods = Counter(), [], Counter()
    error_log = (output / "encoder.log").open("wb")
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=error_log)
    predictions = None
    try:
        predictions = model.predict(source=str(source), stream=True, device=device, imgsz=640, conf=conf,
                                    retina_masks=True, vid_stride=1, verbose=False)
        with (output / "diameter_geometry.jsonl").open("w", encoding="utf-8") as detail:
            for prediction in predictions:
                original = prediction.orig_img
                if original.shape[:2] != (height, width):
                    raise RuntimeError("Source dimensions changed during decoding")
                if processed in source_checks:
                    saved = cv2.imdecode(np.fromfile(source_checks[processed], dtype=np.uint8), cv2.IMREAD_COLOR)
                    if not np.array_equal(saved, original):
                        raise RuntimeError(f"Frame order or original pixels differ at frame {processed}")
                if mm_per_pixel is None:
                    calibration = detect_tick_calibration(original, tick_mm=tick_mm)
                else:
                    calibration = {"available": True, "method": "user_mm_per_pixel_override",
                                   "mm_per_pixel": float(mm_per_pixel), "pitch_px": None,
                                   "tick_mm": float(tick_mm), "reason": "Explicit user override"}
                scale = calibration["mm_per_pixel"] if calibration["available"] else None
                scale_counts["available" if scale is not None else "unavailable"] += 1
                if scale is not None:
                    scale_values.append(scale)
                mask = np.zeros((height, width), dtype=bool)
                if prediction.masks is not None:
                    for item in prediction.masks.data.detach().cpu().numpy():
                        if item.shape != mask.shape:
                            raise RuntimeError("Model mask is not at original resolution")
                        mask |= item > .5
                masks_empty += int(not mask.any())
                measurement = measure_mask(mask, mm_per_pixel=scale, roi=effective_roi)
                measurement = apply_spatial_support_gate(measurement, effective_roi, min_axis_coverage=min_axis_coverage)
                support_rejected += int(measurement.get("quality_gate_rejected_scanline_count", 0) > 0)
                d, d_px = measurement.get("mean_diameter_mm"), measurement.get("mean_diameter_px")
                if d is not None and (not math.isfinite(d) or d <= 0):
                    raise RuntimeError("Invalid diameter")
                measured += int(d is not None)
                methods[measurement.get("method", "unknown")] += 1
                lines = measurement.get("scanlines", [])
                total_lines += len(lines)
                total_excluded += measurement.get("excluded_scanline_count", 0)
                overlay = _draw_overlay(original, mask, measurement)
                header = base.copy()
                pen = ImageDraw.Draw(header)
                number = f"D {d:.2f} mm" if d is not None else (f"D {d_px:.1f} px" if d_px is not None else "측정 불가")
                pen.text((screen_width - 12, 0), number, font=fonts[28], anchor="ra",
                         fill=(65, 210, 255) if d is not None else (186, 192, 201))
                pen.text((12, 37), f"{processed / fps:.3f}초 · 프레임 {processed + 1:,} / {total:,}",
                         font=fonts[15], fill=(190, 203, 213))
                angle = measurement.get("angle_degrees")
                description = f"혈관축 {angle:.1f}° · 측정선 {len(lines)}개" if angle is not None else "유효한 내강 경계를 찾지 못함"
                if measurement.get("status") != "ok":
                    description = measurement.get("message") or "측정 구간이 불충분함"
                if scale is None:
                    description = "배율 확인 필요 · " + description
                pen.text((pane_width + 12, 37), fit_line(description, fonts[15], pane_width - 24), font=fonts[15], fill=(190, 203, 213))
                canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
                canvas[:header_height] = np.asarray(header)[:, :, ::-1]
                canvas[header_height:header_height + height, :width] = original
                canvas[header_height:header_height + height, pane_width:pane_width + width] = overlay
                encoder.stdin.write(canvas.tobytes())
                if processed in sample_indices:
                    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
                    image.save(output / f"preview_{processed:06d}.jpg", quality=93)
                    snapshots.append(image)
                record = {"frame_index": processed, "timestamp_seconds": processed / fps,
                          "status": measurement.get("status"), "diameter_mm": d, "diameter_px": d_px,
                          "angle_degrees": angle, "accepted_scanlines": len(lines),
                          "excluded_scanlines": measurement.get("excluded_scanline_count", 0),
                          "outlier_scanlines": measurement.get("outlier_scanline_count", 0),
                          "coverage": measurement.get("coverage"), "message": measurement.get("message", ""),
                          "calibration_available": calibration["available"], "mm_per_pixel": scale,
                          "tick_pitch_px": calibration.get("pitch_px"), "calibration_reason": calibration.get("reason")}
                record["axis_coverage"] = measurement.get("spatial_support_gate", {}).get("axis_coverage")
                records.append(record)
                detail.write(json.dumps({"frame_index": processed, "timestamp_seconds": processed / fps,
                                         "calibration": calibration, "measurement": measurement}, ensure_ascii=False, allow_nan=False) + "\n")
                processed += 1
                if processed == 1 or processed % 100 == 0:
                    progress = {"state": "running", "source": str(source), "processed": processed,
                                "total": target, "measured": measured, "elapsed_seconds": round(time.monotonic() - started, 1),
                                "output": str(output)}
                    write_json(progress_path, progress)
                    print(json.dumps(progress, ensure_ascii=False), flush=True)
                if processed >= target:
                    break
        if predictions is not None:
            predictions.close()
        encoder.stdin.close()
        if encoder.wait(timeout=120):
            raise RuntimeError("Video encoder failed; see encoder.log")
        if processed != target:
            raise RuntimeError(f"Incomplete source decoding: {processed} / {target}")
    except BaseException as error:
        if predictions is not None:
            predictions.close()
        if encoder.poll() is None:
            if encoder.stdin and not encoder.stdin.closed:
                encoder.stdin.close()
            encoder.terminate()
            encoder.wait(timeout=15)
        write_json(progress_path, {"state": "failed", "source": str(source), "processed": processed,
                                   "total": target, "output": str(output), "error": str(error)})
        raise
    finally:
        error_log.close()
    verified = {"comparison": verify_video(comparison, target, fps, (screen_width, screen_height)),
                "diameter": verify_video(standalone, target, fps, (pane_width, screen_height))}
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["프레임(0부터)", "시간(초)", "측정 상태", "D(mm)", "D(px)", "혈관각도(도)",
                         "유효 측정선", "제외 측정선", "이상 폭 제외", "유효 비율", "배율(mm/px)", "눈금간격(px)", "배율 확인", "배율 설명", "혈관 길이 비율", "측정 설명"])
        for r in records:
            state = "측정" if r["diameter_mm"] is not None else ("배율 확인 필요" if r["diameter_px"] is not None else "측정 불가")
            writer.writerow([r["frame_index"], f"{r['timestamp_seconds']:.6f}", state,
                             "" if r["diameter_mm"] is None else f"{r['diameter_mm']:.6f}",
                             "" if r["diameter_px"] is None else f"{r['diameter_px']:.6f}", r["angle_degrees"],
                             r["accepted_scanlines"], r["excluded_scanlines"], r["outlier_scanlines"], r["coverage"],
                             r["mm_per_pixel"], r["tick_pitch_px"], r["calibration_available"], r["calibration_reason"],
                             r["axis_coverage"], r["message"]])
    summary = {"state": "partial" if partial else "complete", "created_at": datetime.now(timezone.utc).isoformat(),
               "calculation": "diameter_only", "source": str(source), "source_sha256": sha256(source),
               "weights": str(weights), "weights_sha256": sha256(weights),
               "algorithm_sha256": sha256(ROOT / "scripts/measurement.py"),
               "quality_algorithm_sha256": sha256(ROOT / "scripts/diameter_quality.py"),
               "pipeline_sha256": sha256(Path(__file__)),
               "settings": {"tick_mm": float(tick_mm), "mm_per_pixel_override": mm_per_pixel,
                            "roi": roi, "effective_roi": measurement.get("roi"), "confidence": conf,
                            "device": device, "imgsz": 640, "retina_masks": True,
                            "min_axis_coverage": min_axis_coverage},
               "spatial_support_rejected_frames": support_rejected,
               "spatial_support_gate_version": SPATIAL_SUPPORT_GATE_VERSION,
               "calibration_assumption": "Display pixels are isotropic; supplied tick_mm is the distance between adjacent ticks.",
               "source_reference_frame_checks": sorted(index for index in source_checks if index < processed),
               "calibration": {"status_counts": dict(scale_counts),
                               "mm_per_pixel_min": min(scale_values) if scale_values else None,
                               "mm_per_pixel_max": max(scale_values) if scale_values else None},
               "processed": processed, "total": total, "measured": measured, "unmeasurable": processed - measured,
               "frames_without_model_masks": masks_empty, "accepted_scanlines_total": total_lines,
               "excluded_scanlines_total": total_excluded, "status_counts": dict(Counter(r["status"] for r in records)),
               "temporal_smoothing": False, "frames_skipped": total - processed,
               "live_annotations_modified": False, "fmd_calculated": False, "medical_validation": False,
               "comparison_video": str(comparison), "diameter_video": str(standalone), "csv": str(csv_path),
               "poster": str(output / "poster.jpg"), "elapsed_seconds": round(time.monotonic() - started, 1),
               "verification": verified, "method_counts": dict(methods),
               "display_note": "Main lumen; middle 70%; local centerline-normal wall distances; spatial outlier rejection. No interpolation."}
    if snapshots:
        (snapshots[1] if len(snapshots) > 1 else snapshots[0]).save(output / "poster.jpg", quality=94)
        gallery = Image.new("RGB", (screen_width, screen_height * len(snapshots)), (15, 23, 32))
        for index, image in enumerate(snapshots):
            gallery.paste(image, (0, index * screen_height))
        gallery.save(output / "timeline.jpg", quality=92)
    write_json(output / "report.json", summary)
    write_json(progress_path, summary)
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>{html.escape(source.stem)} · D 측정</title>
<style>body{{background:#0f1720;color:#eaf1f5;font:17px 'Malgun Gothic',sans-serif;max-width:1100px;margin:30px auto;padding:0 20px;line-height:1.7}}video{{width:100%;background:#000;border-radius:12px}}a{{color:#36d8be}}</style>
<h1>{html.escape(source.name)} · 내강 지름 D</h1>
<p>왼쪽 원본 / 오른쪽 내강 경계와 D. 주황은 예측 경계, 파랑은 혈관 방향에 직각인 측정선입니다.</p>
<video controls preload="metadata" poster="poster.jpg" src="{html.escape(comparison.name)}"></video>
<p>양 끝 15%씩을 제외한 가운데 구간에서 폭을 평균합니다. 분지·경계 교차·급격히 튀는 폭을 제외했습니다. 검출한 혈관 길이가 혈관 방향으로 본 촬영 영역 길이의 {min_axis_coverage:.0%}보다 짧으면 측정하지 않습니다. 전체 {processed:,}프레임 중 {measured:,}프레임에서 D(mm)를 얻었습니다.</p>
<p>인접 눈금 한 칸은 사용자가 지정한 {tick_mm:g} mm로 설정했습니다. 배율을 확인하지 못한 프레임은 px로 표시하며, 경계가 불충분하면 ‘측정 불가’로 표시합니다. 각 프레임을 독립적으로 계산했습니다.</p>
<p>화면 안의 기울기를 보정한 추정치입니다. 촬영 단면이 혈관 중심을 벗어나는 문제까지 보정하지는 못합니다. FMD는 계산하지 않았습니다.</p>
<p><a href="{html.escape(standalone.name)}">D 표시 영상</a> · <a href="{html.escape(csv_path.name)}">프레임별 D 값</a> · <a href="report.json">처리 기록</a></p></html>'''
    (output / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps({"state": summary["state"], "source": str(source), "processed": processed,
                      "measured": measured, "output": str(output)}, ensure_ascii=False), flush=True)
    return summary
