"""Attempt SAM proposals on every source frame, preserving human annotations."""
import argparse
import json
import os
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

try:
    from ._runtime import ROOT, resolve
    from .assist import propose
    from .auto_prompt import make_prompt
except ImportError:
    from _runtime import ROOT, resolve
    from assist import propose
    from auto_prompt import make_prompt


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.1)


def save_image(path, image, parameters=None):
    ok, encoded = cv2.imencode(path.suffix, image, parameters or [])
    if not ok:
        raise RuntimeError(f"Image encoding failed: {path.name}")
    encoded.tofile(path)


def source_image(frame):
    return resolve(frame["image_path"])


def relative(path):
    return path.relative_to(ROOT).as_posix()


def save_assets(frame, result, output, image):
    frame_id = frame["id"]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for obj in result["objects"]:
        cv2.fillPoly(mask, [np.rint(obj["points"]).astype(np.int32)], 1)
    mask_path = output / "masks" / f"{frame_id}.png"
    overlay_path = output / "overlays" / f"{frame_id}.jpg"
    thumbnail_path = output / "preview_thumbnails" / f"{frame_id}.jpg"
    original_path = output / "images" / f"{frame_id}.png"
    save_image(mask_path, mask)
    overlay = image.copy()
    pixels = mask > 0
    overlay[pixels] = (overlay[pixels] * 0.75 + np.array([190, 216, 54]) * 0.25).astype(np.uint8)
    for obj in result["objects"]:
        cv2.polylines(overlay, [np.rint(obj["points"]).astype(np.int32)], True, (190, 216, 54), 2)
    save_image(overlay_path, overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])
    thumb = cv2.resize(overlay, (320, round(320 * image.shape[0] / image.shape[1])), interpolation=cv2.INTER_AREA)
    save_image(thumbnail_path, thumb, [cv2.IMWRITE_JPEG_QUALITY, 83])
    if not original_path.exists():
        try:
            os.link(source_image(frame), original_path)
        except OSError:
            shutil.copy2(source_image(frame), original_path)
    ys, xs = np.where(pixels)
    flags = []
    scan = frame.get("roi", [0, 0, image.shape[1], image.shape[0]])
    scan_area = max(1, (scan[2] - scan[0]) * (scan[3] - scan[1]))
    area_ratio = float(mask.sum() / scan_area)
    if area_ratio < 0.025 or area_ratio > 0.7:
        flags.append("내강 후보 면적이 자동 확인 범위를 벗어납니다.")
    if len(xs) and (xs.max() - xs.min() + 1) / max(1, ys.max() - ys.min() + 1) < 1.8:
        flags.append("세로 방향으로 넓은 후보입니다. 주 혈관 내강인지 확인하세요.")
    if result.get("diagnostics", {}).get("components", 0) > 1:
        flags.append("분리된 영역 중 가장 큰 영역만 사용되었습니다.")
    if frame.get("quality_score", 1) < 0.5:
        flags.append("영상 품질 후보 점수가 낮습니다.")
    result.setdefault("diagnostics", {}).update({"polygon_mask_pixels": int(mask.sum()), "roi_area_ratio": area_ratio, "geometry_flags": flags})
    result["questionable"] = bool(flags)
    result["warnings"] = list(result.get("warnings", [])) + flags
    return {"frame_id": frame_id, "video_id": frame["video_id"], "frame_index": frame["frame_index"],
            "timestamp_seconds": frame.get("timestamp_seconds"), "status": "draft", "approved": False,
            "proposal_path": relative(output / "proposals" / f"{frame_id}.json"),
            "image_path": relative(original_path), "mask_path": relative(mask_path),
            "overlay_path": relative(overlay_path), "thumbnail_path": relative(thumbnail_path),
            "questionable": bool(flags), "warnings": flags, "selected400": bool(frame.get("selected"))}


def main():
    parser = argparse.ArgumentParser(description="전체 프레임 SAM 경계 초안 생성 (자동 승인 없음)")
    parser.add_argument("--output", default="data/auto_generated")
    parser.add_argument("--limit", type=int, help="벤치마크용 일부 프레임 수; 기본은 전체")
    parser.add_argument("--retry-failures", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    frames = manifest["frames"]
    if args.limit:
        frames = frames[:args.limit]
    output = resolve(args.output)
    if not output.is_relative_to(ROOT):
        parser.exit(2, "출력 폴더는 작업대 프로젝트 안에 있어야 합니다.\n")
    for name in ("proposals", "masks", "images", "overlays", "preview_thumbnails"):
        (output / name).mkdir(parents=True, exist_ok=True)
    generated, failures = [], []
    processed = resumed = 0
    per_video = Counter()
    start = time.perf_counter()
    benchmark_times = []
    initial_time = now()

    def checkpoint(state="running", write_index=False):
        elapsed = time.perf_counter() - start
        newly_processed = processed - resumed
        fps = newly_processed / elapsed if elapsed else 0
        status = {"version": 1, "state": state, "running": state == "running", "total": len(frames),
                  "processed": processed, "attempted": processed, "generated": len(generated), "failed": len(failures),
                  "resumed": resumed, "remaining": len(frames) - processed, "elapsed_seconds": round(elapsed, 2),
                  "fps": round(fps, 3), "eta_seconds": round((len(frames) - processed) / fps, 1) if fps else None,
                  "last_frame_id": frames[processed - 1]["id"] if processed else None,
                  "started_at": initial_time, "updated_at": now(), "review_required": True,
                  "approved": False, "generated_means": "SAM returned a retained polygon; anatomical accuracy is unverified",
                  "per_video_generated": dict(per_video)}
        atomic_json(output / "status.json", status)
        if write_index:
            atomic_json(output / "index.json", {"version": 1, "complete": state == "completed", "total": len(frames),
                        "generated_count": len(generated), "failed_count": len(failures), "review_required": True,
                        "approved": False, "frames": generated, "updated_at": now()})
            atomic_json(ROOT / "reports" / "batch_progress.json", status)
        return status

    checkpoint(write_index=True)
    try:
        for frame in frames:
            tick = time.perf_counter()
            path = output / "proposals" / (frame["id"] + ".json")
            previous = None
            if path.is_file():
                try:
                    previous = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    pass
            if previous and previous.get("frame_id") == frame["id"] and (previous.get("objects") or not args.retry_failures):
                if previous.get("objects"):
                    entry = previous.get("gallery_entry")
                    if entry and all(resolve(entry[k]).is_file() for k in ("image_path", "mask_path", "overlay_path", "thumbnail_path")):
                        generated.append(entry)
                        per_video[frame["video_id"]] += 1
                    else:
                        image = cv2.imdecode(np.fromfile(str(source_image(frame)), dtype=np.uint8), cv2.IMREAD_COLOR)
                        entry = save_assets(frame, previous, output, image)
                        previous["gallery_entry"] = entry
                        atomic_json(path, previous)
                        generated.append(entry)
                        per_video[frame["video_id"]] += 1
                else:
                    failures.append({"frame_id": frame["id"], "error": previous.get("error", "No polygon generated")})
                resumed += 1
            else:
                result = {"frame_id": frame["id"], "status": "failed", "objects": [], "approved": False, "review_required": True, "origin": "auto"}
                try:
                    image = cv2.imdecode(np.fromfile(str(source_image(frame)), dtype=np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError("원본 프레임을 읽지 못했습니다.")
                    try:
                        box, prompt_diagnostics = make_prompt(frame, image)
                    except Exception as prompt_error:
                        # Still attempt real SAM inference if a heuristic fails.
                        box = frame.get("roi", [0, 0, image.shape[1] - 1, image.shape[0] - 1])
                        prompt_diagnostics = {"fallback": True, "error": str(prompt_error)}
                    result = propose(str(source_image(frame)), box=box)
                    result.update(frame_id=frame["id"], status="draft", approved=False, review_required=True, origin="auto",
                                  video_id=frame["video_id"], frame_index=frame["frame_index"], timestamp_seconds=frame.get("timestamp_seconds"),
                                  prompt_diagnostics=prompt_diagnostics, created_at=now())
                    if not result.get("objects"):
                        raise ValueError("SAM 예측에서 유효한 polygon이 생성되지 않았습니다.")
                    entry = save_assets(frame, result, output, image)
                    result["gallery_entry"] = entry
                    atomic_json(path, result)
                    generated.append(entry)
                    per_video[frame["video_id"]] += 1
                except Exception as exc:
                    result.update(status="failed", error=f"{type(exc).__name__}: {exc}", objects=[], created_at=now())
                    atomic_json(path, result)
                    failure = {"frame_id": frame["id"], "video_id": frame["video_id"], "error": result["error"], "time": now()}
                    failures.append(failure)
                    with (output / "failures.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(failure, ensure_ascii=False) + "\n")
                benchmark_times.append(time.perf_counter() - tick)
            processed += 1
            if processed == 20 and benchmark_times:
                warm = benchmark_times[5:] or benchmark_times
                report = {"frames": len(benchmark_times), "mean_seconds_including_startup": float(np.mean(benchmark_times)),
                          "warm_mean_seconds": float(np.mean(warm)), "estimated_total_seconds": float(np.mean(warm) * len(frames)),
                          "precision": "float32", "model": "sam2.1_t.pt", "imgsz": 1024, "real_sam_inference_every_frame": True}
                atomic_json(ROOT / "reports" / "batch_benchmark.json", report)
                print("BENCHMARK " + json.dumps(report), flush=True)
            if processed % 25 == 0:
                state = checkpoint(write_index=processed % 100 == 0)
                if processed % 100 == 0:
                    print("PROGRESS " + json.dumps({k: state[k] for k in ("processed", "total", "generated", "failed", "fps", "eta_seconds")}), flush=True)
        status = checkpoint("completed", True)
        (output / "failures.jsonl").touch(exist_ok=True)
        atomic_json(ROOT / "reports" / "batch_final.json", status)
        print("COMPLETE " + json.dumps(status), flush=True)
    except BaseException:
        checkpoint("interrupted", True)
        raise


if __name__ == "__main__":
    main()
