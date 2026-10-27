#!/usr/bin/env python3
"""Extract every decodable ultrasound frame and prepare provisional review picks.

No clinical labels are generated. Frames remain lossless, uncropped, full-size PNG.
Quality measurements use a visually inspected ROI, excluding UI/ECG. Timestamps
are derived from zero-based source frame index / container FPS, not ECG phase.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(r"C:\HJK\existing data-20261008T123158Z-1-001\existing data")
ROI = [100, 125, 540, 365]
SCAN_ROI = [94, 40, 545, 382]
# Source-specific review ranges refined after inspecting representative frames
# and all first-pass candidate sheets. They only guide candidate selection;
# every source frame remains accessible for labeling. These are not FMD phases.
VISUALLY_REVIEWED_RANGES = {
    "EBL_20260904_CAROTID_0002.AVI": [(0, 27.8), (31.2, 43.8), (48, 55.8), (60, 75.3)],
    "EBL_20260904_CAROTID_0004.AVI": [(4.5, 31), (51, 55), (62, 65.5), (68, 72.7)],
    "EBL_20260904_CAROTID_0005.AVI": [(9.5, 75)],
    "EBL_20260904_CAROTID_0006.AVI": [(0, 11), (32, 37), (39, 41.5), (65, 71.8)],
    "EBL_20260904_CAROTID_0007.AVI": [(0, 60)],
    "EBL_20260904_CAROTID_0009.AVI": [(0, 18.7), (21.5, 34), (38.5, 75.3)],
}
VIDEO_WEIGHT_MULTIPLIERS = {
    "EBL_20260904_CAROTID_0004.AVI": .5,
    "EBL_20260904_CAROTID_0006.AVI": .3,
    "EBL_20260904_CAROTID_0007.AVI": .8,
}


def quality_metrics(gray):
    """Image-quality proxies; the dark-band proxy is not an anatomical mask."""
    x0, y0, x1, y1 = ROI
    r = gray[y0:y1, x0:x1]
    p5, p50, p95 = np.percentile(r, [5, 50, 95])
    smoothed = cv2.GaussianBlur(r, (7, 7), 1.5)
    threshold = max(12.0, min(30.0, float(np.percentile(smoothed, 22))))
    centers, widths, valid = [], [], []
    for x in range(20, r.shape[1] - 20, 10):
        column = smoothed[:, x] < threshold
        starts = np.flatnonzero(np.diff(np.r_[False, column, False].astype(np.int8)) == 1)
        ends = np.flatnonzero(np.diff(np.r_[False, column, False].astype(np.int8)) == -1)
        lengths = ends - starts
        eligible = np.flatnonzero((lengths >= 15) & (starts > 4) & (ends < r.shape[0] - 4))
        if len(eligible):
            k = eligible[np.argmax(lengths[eligible])]
            centers.append(float((starts[k] + ends[k]) / 2))
            widths.append(float(lengths[k]))
            valid.append(x)
    coverage = len(valid) / len(range(20, r.shape[1] - 20, 10))
    width_cv, residual, slope = 1.0, 50.0, 1.0
    if len(valid) >= 8:
        coeff = np.polyfit(valid, centers, 1)
        residual = float(np.sqrt(np.mean((np.polyval(coeff, valid) - centers) ** 2)))
        width_cv = float(np.std(widths) / max(1.0, np.mean(widths)))
        slope = float(abs(coeff[0]))
    metrics = {
        "laplacian_variance": round(float(cv2.Laplacian(r, cv2.CV_64F).var()), 6),
        "contrast_p95_p5": round(float(p95 - p5), 6),
        "median_intensity": round(float(p50), 6),
        "dark_fraction": round(float(np.mean(r < 5)), 6),
        "bright_fraction": round(float(np.mean(r > 245)), 6),
        "dark_band_coverage": round(float(coverage), 6),
        "dark_band_width_cv": round(float(width_cv), 6),
        "dark_band_center_residual_px": round(float(residual), 6),
        "dark_band_abs_slope": round(float(slope), 6),
    }
    descriptor = cv2.resize(smoothed, (32, 16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    return metrics, descriptor.flatten()


def save_images(frame, image_path, thumbnail_path):
    if not image_path.exists():
        ok, encoded = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not ok:
            raise RuntimeError(f"PNG encoding failed: {image_path}")
        image_path.write_bytes(encoded.tobytes())
    if not thumbnail_path.exists():
        preview = cv2.resize(frame, (320, 224), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 83])
        if not ok:
            raise RuntimeError(f"JPEG encoding failed: {thumbnail_path}")
        thumbnail_path.write_bytes(encoded.tobytes())


def extract(source, root):
    paths = sorted([p for p in source.iterdir() if p.suffix.lower() == ".avi"])
    if not paths:
        raise SystemExit(f"No AVI files found: {source}")
    frames, videos, descriptors, pending = [], [], [], []
    (root / "data/thumbnails").mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=6) as pool:
        for vi, path in enumerate(paths, 1):
            video_id = f"v{vi:02d}"
            target = root / "data/all_frames" / video_id
            target.mkdir(parents=True, exist_ok=True)
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open source: {path}")
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0:
                raise RuntimeError(f"Invalid source FPS: {path}")
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            video = {
                "video_id": video_id, "source_video": path.name, "source_path": str(path),
                "patient_id": "unassigned", "anatomy": "unconfirmed",
                "fps": fps, "expected_frame_count": expected,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "codec": "".join(chr((fourcc >> (8 * j)) & 255) for j in range(4)),
                "timestamp_method": "zero_based_frame_index_divided_by_container_fps",
                "decoded_frame_count": 0, "decode_failures": [],
            }
            index = 0
            while index < expected or expected == 0:
                ok, frame = cap.read()
                if not ok:
                    if expected == 0:
                        break
                    video["decode_failures"].append(index)
                    index += 1
                    if index < expected:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    continue
                frame_id = f"{video_id}_f{index:06d}"
                image_rel = f"data/all_frames/{video_id}/{frame_id}.png"
                thumb_rel = f"data/thumbnails/{frame_id}.jpg"
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                metrics, descriptor = quality_metrics(gray)
                digest = hashlib.sha256(frame.tobytes()).hexdigest()
                frames.append({
                    "id": frame_id, "source_video": path.name, "video_id": video_id,
                    "patient_id": "unassigned", "frame_index": index,
                    "timestamp_seconds": round(index / fps, 9),
                    "width": int(frame.shape[1]), "height": int(frame.shape[0]),
                    "image_path": image_rel, "thumbnail_path": thumb_rel,
                    "selected": False, "selection_rank": None, "quality_score": 0.0,
                    "quality_metrics": metrics, "roi": ROI,
                    "pixel_sha256": digest,
                })
                descriptors.append(descriptor)
                pending.append(pool.submit(save_images, frame.copy(), root / image_rel, root / thumb_rel))
                if len(pending) >= 48:
                    pending.pop(0).result()
                video["decoded_frame_count"] += 1
                index += 1
                if index % 500 == 0:
                    print(f"{path.name}: {index}/{expected} decoded; elapsed {time.monotonic()-started:.0f}s", flush=True)
            cap.release()
            video["duration_seconds"] = round(expected / fps, 9)
            videos.append(video)
            print(f"Completed {path.name}: {video['decoded_frame_count']} frames", flush=True)
        for result in pending:
            result.result()
    np.savez_compressed(root / "work/selection_descriptors.npz", descriptors=np.asarray(descriptors, dtype=np.float32))
    return {"version": 1, "frames": frames, "videos": videos}, np.asarray(descriptors, dtype=np.float32)


def percentile_scale(values):
    lo, hi = np.percentile(values, [5, 95])
    return np.clip((values - lo) / max(hi - lo, 1e-6), 0, 1)


def select(manifest, descriptors, count, minimum_gap_seconds):
    frames = manifest["frames"]
    q = [f["quality_metrics"] for f in frames]
    focus = percentile_scale(np.log1p([m["laplacian_variance"] for m in q]))
    contrast = percentile_scale(np.asarray([m["contrast_p95_p5"] for m in q]))
    coverage = np.asarray([m["dark_band_coverage"] for m in q])
    regularity = np.exp(-np.asarray([m["dark_band_width_cv"] for m in q]) * 2.2)
    straightness = np.exp(-np.asarray([m["dark_band_center_residual_px"] for m in q]) / 18)
    visibility = np.clip(1 - np.asarray([m["dark_fraction"] for m in q]) / .85, .15, 1)
    scores = .28 * focus + .25 * contrast + .15 * coverage + .14 * regularity + .12 * straightness + .06 * visibility
    for i, f in enumerate(frames):
        f["quality_score"] = round(float(scores[i]), 6)
        f["selected"] = False
        f["selection_rank"] = None
        f.pop("selected_image_path", None)
        f["quality_metrics"].pop("nearest_prior_selected_roi_rms", None)
    def reviewed_eligible(i):
        f = frames[i]
        intervals = VISUALLY_REVIEWED_RANGES.get(f["source_video"])
        return scores[i] >= .50 and (intervals is None or any(a <= f["timestamp_seconds"] < b for a, b in intervals))

    eligible = {i for i in range(len(frames)) if reviewed_eligible(i)}
    for i, f in enumerate(frames):
        f["candidate_eligible"] = i in eligible
    # Quality-weighted quotas preserve coverage while reducing branch-heavy clips.
    groups = [[i for i, f in enumerate(frames) if f["video_id"] == v["video_id"] and i in eligible] for v in manifest["videos"]]
    medians = np.asarray([np.median(scores[g]) for g in groups])
    weights = np.clip((medians - .2), .05, None) ** 1.5
    weights *= np.asarray([VIDEO_WEIGHT_MULTIPLIERS.get(v["source_video"], 1) for v in manifest["videos"]])
    floor = min(20, count // len(groups))
    remaining = count - floor * len(groups)
    raw = remaining * weights / weights.sum()
    quotas = np.floor(raw).astype(int) + floor
    for k in np.argsort(-(raw - np.floor(raw)))[:count - int(quotas.sum())]:
        quotas[k] += 1
    selected, by_video, digests = [], {}, set()
    selection_distances = {}
    gaps_relaxed = False

    def feasible(i, gap):
        f = frames[i]
        return i in eligible and f["pixel_sha256"] not in digests and all(
            abs(f["timestamp_seconds"] - frames[j]["timestamp_seconds"]) >= gap
            for j in by_video.get(f["video_id"], [])
        )

    def choose(candidates, gap, top=16):
        feasible_candidates = [i for i in candidates if feasible(i, gap)]
        if not feasible_candidates:
            return None
        ordered = sorted(feasible_candidates, key=lambda i: scores[i], reverse=True)[:top]
        best, best_objective, best_distance = None, -math.inf, 1.0
        for i in ordered:
            if selected:
                distance = float(np.sqrt(np.mean((descriptors[selected] - descriptors[i]) ** 2, axis=1)).min())
            else:
                distance = 1.0
            # A small novelty bonus favors different appearances without allowing
            # poor image quality to dominate. Exact-pixel duplicates are forbidden.
            objective = float(scores[i]) + .18 * min(distance / .10, 1)
            if objective > best_objective:
                best, best_objective, best_distance = i, objective, distance
        selection_distances[best] = best_distance
        return best

    def add(i):
        selected.append(i)
        f = frames[i]
        by_video.setdefault(f["video_id"], []).append(i)
        digests.add(f["pixel_sha256"])

    # Bin-centered picks give temporal spread and avoid drawing 400 neighboring
    # frames from a short high-quality interval.
    for group, quota, video in zip(groups, quotas, manifest["videos"]):
        for chunk in np.array_split(group, int(quota)):
            candidates = chunk.tolist()
            chosen = choose(candidates, minimum_gap_seconds)
            if chosen is not None:
                add(chosen)
        while len(by_video.get(video["video_id"], [])) < int(quota):
            chosen = choose(group, minimum_gap_seconds, top=48)
            if chosen is None:
                break
            add(chosen)
    # Normally unnecessary, but make the count explicit if a video's bins fail.
    while len(selected) < count:
        chosen = choose(range(len(frames)), minimum_gap_seconds, top=80)
        if chosen is None:
            gaps_relaxed = True
            chosen = choose(range(len(frames)), .50, top=80)
        if chosen is None:
            raise RuntimeError(f"Only {len(selected)} unique, temporally spaced picks available; requested {count}")
        add(chosen)
    for rank, i in enumerate(sorted(selected, key=lambda i: scores[i], reverse=True), 1):
        frames[i]["selected"] = True
        frames[i]["selection_rank"] = rank
        frames[i]["quality_metrics"]["nearest_prior_selected_roi_rms"] = round(selection_distances[i], 6)
    manifest["selection"] = {
        "status": "provisional_candidates_requiring_human_review",
        "target_count": count, "actual_count": len(selected),
        "minimum_temporal_gap_seconds": minimum_gap_seconds,
        "minimum_gap_relaxed": gaps_relaxed, "quality_roi": ROI, "scan_roi": SCAN_ROI,
        "pixel_duplicate_policy": "exclude_exact_decoded_pixel_duplicates_across_all_videos",
        "quality_score_is_clinical_assessment": False,
        "quality_score_floor": .50,
        "visually_reviewed_selection_ranges_seconds": VISUALLY_REVIEWED_RANGES,
        "video_weight_multipliers": VIDEO_WEIGHT_MULTIPLIERS,
        "method": "Visually inspected ROI and first-pass candidate contact sheets; focus, contrast, dark-band visibility and regularity proxies; quality floor; visually reviewed source time ranges; quality-weighted video quotas; temporal bins; small ROI appearance-diversity bonus.",
        "per_video": {v["video_id"]: len(by_video.get(v["video_id"], [])) for v in manifest["videos"]},
    }
    return sorted(selected, key=lambda i: (frames[i]["video_id"], frames[i]["frame_index"]))


def contact_sheet(frames, root, outpath, crop=False, columns=8):
    tile_w, tile_h = (240, 178) if not crop else (240, 165)
    rows = math.ceil(len(frames) / columns)
    sheet = Image.new("RGB", (columns * tile_w, rows * tile_h), "#101723")
    draw = ImageDraw.Draw(sheet)
    for j, frame in enumerate(frames):
        im = Image.open(root / frame["image_path"]).convert("RGB")
        if crop:
            im = im.crop(tuple(SCAN_ROI))
        im.thumbnail((tile_w - 8, tile_h - 28), Image.Resampling.LANCZOS)
        x, y = (j % columns) * tile_w, (j // columns) * tile_h
        sheet.paste(im, (x + (tile_w - im.width) // 2, y + 22))
        draw.text((x + 5, y + 4), f"{frame['id']}  {frame['timestamp_seconds']:.2f}s  q={frame['quality_score']:.2f}", fill="#e1e7ef")
    sheet.save(outpath, quality=94)


def write_reports(manifest, selected, root):
    frames = manifest["frames"]
    images = root / "data/selected400/images"
    images.mkdir(parents=True, exist_ok=True)
    selected_ids = {frames[i]["id"] for i in selected}
    # Remove only obsolete generated pick files in this exact managed directory.
    for old in images.glob("v??_f??????.png"):
        if old.stem not in selected_ids:
            old.unlink()
    for i in selected:
        f = frames[i]
        target = images / (f["id"] + ".png")
        if not target.exists():
            shutil.copy2(root / f["image_path"], target)
        f["selected_image_path"] = target.relative_to(root).as_posix()
    manifest_temp = root / "data/manifest.json.tmp"
    manifest_temp.write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    manifest_temp.replace(root / "data/manifest.json")
    fields = ["id", "source_video", "video_id", "patient_id", "frame_index", "timestamp_seconds", "width", "height", "image_path", "thumbnail_path", "selected", "selection_rank", "quality_score", "pixel_sha256"]
    for path, rows in [(root / "data/manifest.csv", frames), (root / "reports/selection_400.csv", [frames[i] for i in selected])]:
        with path.open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    picks = [frames[i] for i in selected]
    for page_i in range(0, len(picks), 80):
        contact_sheet(picks[page_i:page_i+80], root, root / f"reports/selection_contact_{page_i//80+1:02d}.jpg", columns=8)
        contact_sheet(picks[page_i:page_i+80], root, root / f"reports/selection_crop_{page_i//80+1:02d}.jpg", crop=True, columns=8)
    # Compact overview, supplemented by detailed 80-frame pages.
    overview = picks[::max(1, len(picks)//40)][:40]
    contact_sheet(overview, root, root / "reports/selection_overview.jpg", columns=5)
    hashes = [f["pixel_sha256"] for f in frames]
    gaps = {}
    for v in manifest["videos"]:
        ts = sorted(f["timestamp_seconds"] for f in picks if f["video_id"] == v["video_id"])
        gaps[v["video_id"]] = round(min(np.diff(ts)), 6) if len(ts) > 1 else None
    report = {
        "decoded_frames": len(frames), "selected_frames": len(picks),
        "unique_decoded_pixel_frames": len(set(hashes)),
        "exact_duplicate_decoded_frames": len(hashes)-len(set(hashes)),
        "minimum_selected_temporal_gap_seconds_per_video": gaps,
        "videos": manifest["videos"], "selection": manifest["selection"],
        "limitations": [
            "File names say CAROTID; anatomical identity, cuff release, baseline/post-release phase, calibration, and patient identities are unconfirmed.",
            "Some videos contain branching/oblique views; selection metrics cannot certify correct FMD acquisition or reliable diameter measurement.",
            "These are provisional review candidates, not clinical labels or independently verified optimal training images.",
            "All seven videos may belong to one person; no independent patient count or valid train/test split is established.",
            "Timestamps use source frame index / container FPS; no cardiac phase or cuff event is inferred.",
        ],
    }
    (root / "reports/selection_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 프레임 추출 및 400장 후보 선별", "", f"전체 {len(frames):,}프레임을 원본 크기 PNG로 추출했고, 검토할 후보 {len(picks)}장을 별도 폴더에 복사했습니다.", "", "후보는 영상의 선명도·대비, 혈관처럼 보이는 어두운 띠의 가시성·규칙성, 시간 간격과 모습의 다양성을 기준으로 잠정 선별했습니다. ROI는 화질 점수를 계산할 때만 사용했고, 저장된 원본 프레임을 자르거나 보정하지 않았습니다.", "", "원본 샘플과 후보 미리보기를 직접 확인한 뒤, 경계가 가려지거나 분지가 크게 나타나는 구간을 줄였습니다. 화질 점수 0.50 이상으로 제한했고, 같은 영상에서 선택한 두 프레임 사이에는 최소 18프레임(약 0.626초) 간격이 있습니다. 이 시간 구간은 후보 선별용이며 기준/커프 해제 후 구간을 뜻하지 않습니다.", "", "| 영상 | 전체 프레임 | 후보 |", "|---|---:|---:|"]
    for v in manifest["videos"]:
        lines.append(f"| {v['source_video']} | {v['decoded_frame_count']:,} | {manifest['selection']['per_video'][v['video_id']]} |")
    lines += ["", "모든 영상: 640×448, 28.775 fps, MP42. 프레임 번호는 0부터 시작합니다. 시간은 프레임 번호 ÷ FPS입니다.", "", "주의: 파일명은 CAROTID이며, 일부 장면에는 분지·비스듬한 혈관이 보입니다. FMD에 필요한 실제 측정 혈관, 커프 해제 시각, 기준/반응 구간, mm/픽셀 환산, 환자 구분은 사용자가 확인해야 합니다. 이 선별은 임상적 적합성 판단이나 정답 레이블을 뜻하지 않습니다. 연속 프레임을 무작위로 나누어 성능을 평가하지 마세요.", "", "- 전체: `data/all_frames/`", "- 후보 400장: `data/selected400/images/`", "- 목록/시간/점수: `data/manifest.json`, `data/manifest.csv`, `reports/selection_400.csv`", "- 전체 화면 미리보기: `reports/selection_contact_01.jpg` ~ `05.jpg`", "- 초음파 영역 확대 미리보기: `reports/selection_crop_01.jpg` ~ `05.jpg`", "- 간략 보기: `reports/selection_overview.jpg`", "", "재실행: 프로젝트 환경에서 `python scripts/extract_frames.py`를 실행합니다. 재선별만 하려면 `--select-only`를 덧붙입니다. 기존 원본 PNG와 축소 이미지는 재사용합니다."]
    (root / "reports/selection_readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--min-gap", type=float, default=.60)
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()
    for folder in ["data", "reports", "work"]:
        (args.root / folder).mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    if args.select_only:
        manifest = json.loads((args.root / "data/manifest.json").read_text(encoding="utf-8"))
        descriptors = np.load(args.root / "work/selection_descriptors.npz")["descriptors"]
    else:
        manifest, descriptors = extract(args.source_dir, args.root)
    selected = select(manifest, descriptors, args.count, args.min_gap)
    write_reports(manifest, selected, args.root)
    print(json.dumps({"frames": len(manifest["frames"]), "selection": manifest["selection"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
