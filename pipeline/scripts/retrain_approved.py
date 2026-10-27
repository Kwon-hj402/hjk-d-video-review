"""Freeze current approved labels, then train afresh from official COCO weights."""
import argparse
import collections
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

try:
    from ._runtime import ROOT, configure
    from .export_dataset import effective_annotation, export, validate_objects
    from .train import UNKNOWN, validate_dataset
except ImportError:
    from _runtime import ROOT, configure
    from export_dataset import effective_annotation, export, validate_objects
    from train import UNKNOWN, validate_dataset

OFFICIAL_SHA256 = "1caa81c0195412efa411b632bcfb8c184939dddb6ae41f6a80c41b211ff257c3"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def capture_approved_snapshot(project_root=ROOT):
    """Read live labels once; write only a new snapshot and its bookkeeping."""
    root = Path(project_root).resolve()
    captured_at = datetime.now(timezone.utc).isoformat()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    snapshot = root / "data/training_snapshots" / stamp
    run_name = "fresh_lumen_" + stamp
    manifest = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
    metadata_path = root / "data/video_metadata.json"
    video_meta = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    video_meta = video_meta.get("videos", video_meta)
    captured = []
    for frame in manifest["frames"]:
        annotation_path = root / "data/annotations" / (frame["id"] + ".json")
        if not annotation_path.exists():
            continue
        raw = annotation_path.read_bytes()
        annotation = json.loads(raw)
        if annotation.get("status") != "approved":
            continue
        annotation = effective_annotation(frame, annotation, video_meta)
        if annotation.get("negative") and not annotation.get("objects"):
            annotation["objects"] = []
        else:
            validate_objects(annotation.get("objects", []), frame["width"], frame["height"])
            if annotation.get("negative"):
                raise ValueError(f"내강 없음 레이블에 polygon이 있습니다: {frame['id']}")
        source = (root / frame["image_path"]).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise ValueError(f"원본 프레임 경로가 유효하지 않습니다: {frame['id']}")
        captured.append((frame, annotation, hashlib.sha256(raw).hexdigest(), source))
    if not captured or not any(annotation["objects"] for _, annotation, _, _ in captured):
        raise ValueError("승인한 내강 양성 레이블이 필요합니다. 자동 초안과 미승인 레이블은 사용하지 않습니다.")
    known_patients = all(str(a.get("patient_id", "unassigned")).strip().lower() not in UNKNOWN for _, a, _, _ in captured)
    mode = "patient" if known_patients else "video-exploratory"
    group_key = "patient_id" if known_patients else "video_id"
    counts = collections.Counter(str(a[group_key]) if known_patients else f[group_key] for f, a, _, _ in captured)
    if len(counts) < 3:
        raise ValueError("학습·검증·시험 분할에는 서로 다른 환자 또는 원본 영상 그룹이 최소 3개 필요합니다.")
    seed = 42
    selection = "Patient groups split with fixed seed 42 before training."
    if not known_patients:
        ranked = sorted(counts, key=lambda key: (-counts[key], key))
        val_group = ranked[1]
        test_group = min(ranked[2:], key=lambda key: (counts[key], key))
        ntest, nval = max(1, round(len(counts) * .1)), max(1, round(len(counts) * .1))
        for candidate in range(10000):
            groups = sorted(counts)
            random.Random(candidate).shuffle(groups)
            if groups[0] == test_group and groups[ntest] == val_group and ranked[0] not in groups[:ntest+nval]:
                seed = candidate
                break
        else:
            raise ValueError("문서화한 영상 그룹 분할을 구성하지 못했습니다.")
        selection = "Largest group retained for training; second-largest included in validation; smallest remaining included in test. Other groups assigned by the documented seed; chosen before training."
    if snapshot.exists() or (root / "models/runs" / run_name).exists():
        raise ValueError("동일한 학습 이름이 이미 존재합니다. 새 실행이 필요합니다.")
    annotation_dir = snapshot / "data/annotations"
    annotation_dir.mkdir(parents=True)
    for frame, annotation, _, source in captured:
        atomic_json(annotation_dir / (frame["id"] + ".json"), annotation)
        target = snapshot / frame["image_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.hardlink_to(source)
        except OSError:
            shutil.copy2(source, target)
    manifest["frames"] = [frame for frame, _, _, _ in captured]
    atomic_json(snapshot / "data/manifest.json", manifest)
    exported = export(root=snapshot, mode=mode, seed=seed)
    dataset = Path(exported["versioned_output"]) / "dataset.yaml"
    _, split, frame_counts = validate_dataset(dataset, allow_exploratory=not known_patients)
    if {f["id"] for f, _, _, _ in captured} != {f["frame_id"] for f in split["frames"]}:
        raise ValueError("승인 snapshot과 내보낸 프레임이 다릅니다.")
    positives = sum(bool(annotation["objects"]) for _, annotation, _, _ in captured)
    report = {
        "captured_at": captured_at, "snapshot": str(snapshot), "dataset": str(dataset),
        "approved_frames": len(captured), "positive_frames": positives, "negative_frames": len(captured) - positives,
        "frame_counts": frame_counts, "source_video_groups": split["source_video_groups"],
        "split_policy": split["split_policy"], "seed": seed, "split_selection": selection,
        "drafts_included": False, "unapproved_auto_proposals_included": False, "live_annotations_modified": False,
        "medical_validation": False, "warnings": exported["warnings"],
        "frames": [{"frame_id": f["id"], "status": a["status"], "negative": bool(a.get("negative")),
                    "source_annotation_sha256": sha} for f, a, sha, _ in captured],
    }
    report_path = root / "reports" / ("approved_training_snapshot_" + stamp + ".json")
    atomic_json(report_path, report)
    captured_info = {key: report[key] for key in ("dataset", "snapshot", "approved_frames", "positive_frames", "negative_frames", "frame_counts", "split_policy")}
    captured_info.update(report=str(report_path), run_name=run_name, fresh_from_original_pretrained=True, resume=False)
    atomic_json(root / "work/latest_approved_snapshot.json", captured_info)
    return captured_info


def training_command(captured, project_root=ROOT, epochs=100, batch=4, device="0"):
    root = Path(project_root).resolve()
    command = [sys.executable, "-u", str(root / "scripts/train.py"), "--data", captured["dataset"],
               "--model", str(root / "models/yolo11s-seg.pt"), "--epochs", str(epochs),
               "--batch", str(batch), "--device", str(device), "--name", captured["run_name"]]
    if captured["split_policy"] == "video_group_exploratory":
        command.append("--allow-exploratory")
    return command


def verify_and_publish(captured, project_root=ROOT, model_loader=None):
    root = Path(project_root).resolve()
    initial = root / "models/yolo11s-seg.pt"
    run = root / "models/runs" / captured["run_name"]
    arguments = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
    if Path(arguments["model"]).resolve() != initial or arguments.get("resume") is not False or arguments.get("pretrained") is not True or digest(initial) != OFFICIAL_SHA256:
        raise ValueError("원본 사전학습 모델에서 시작한 새 학습임을 확인하지 못했습니다.")
    metadata_path = run / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("purpose") != "user_approved_lumen_training" or Path(metadata["initial_model"]).resolve() != initial:
        raise ValueError("학습 메타데이터가 원본 새 학습과 일치하지 않습니다.")
    best = run / "weights/best.pt"
    if not best.is_file():
        raise ValueError("성공한 best.pt가 없습니다. 기존 추론 모델은 유지됩니다.")
    if model_loader is None:
        configure()
        from ultralytics import YOLO
        model_loader = YOLO
    model = model_loader(str(best), task="segment")
    if model.task != "segment" or model.names != {0: "lumen"}:
        raise ValueError("완료 모델은 내강 lumen 1개 클래스 segmentation이어야 합니다.")
    dataset = Path(captured["dataset"])
    _, split, counts = validate_dataset(dataset, allow_exploratory=captured["split_policy"] == "video_group_exploratory")
    sample = next((row for row in split["frames"] if row["split"] == "val" and not row.get("negative")), None)
    if sample is None:
        sample = next((row for row in split["frames"] if not row.get("negative")), None)
    if sample is None:
        raise ValueError("완료 모델의 추론 확인에 사용할 승인 양성 이미지가 없습니다.")
    sample_image = dataset.parent / "images" / sample["split"] / (sample["frame_id"] + ".png")
    predictions = model.predict(source=str(sample_image), device=str(arguments["device"]), imgsz=640, conf=.25, retina_masks=True, save=False, verbose=False)
    if len(predictions) != 1:
        raise ValueError("완료 모델의 검증 추론에 실패했습니다.")
    from PIL import Image
    with Image.open(sample_image) as image:
        expected_shape = (image.height, image.width)
    if tuple(predictions[0].orig_shape) != expected_shape:
        raise ValueError("완료 모델의 예측 좌표가 원본 이미지 크기와 다릅니다.")
    if predictions[0].masks is not None and tuple(predictions[0].masks.data.shape[-2:]) != expected_shape:
        raise ValueError("완료 모델의 마스크가 원본 해상도와 다릅니다.")
    with (run / "results.csv").open(encoding="utf-8", newline="") as stream:
        completed_epochs = len(list(csv.DictReader(stream)))
    if completed_epochs < 1:
        raise ValueError("완료한 학습 epoch가 없습니다.")
    proof = {"verified_at": datetime.now(timezone.utc).isoformat(), "training_exit_code": 0,
             "fresh_from_original_pretrained": True, "resume": False, "initial_model": str(initial),
             "initial_model_sha256": OFFICIAL_SHA256, "approved_only": True,
             "approved_frames": captured["approved_frames"], "positive_frames": captured["positive_frames"], "negative_frames": captured["negative_frames"],
             "negative_labels_accepted_and_verified": True, "frame_counts": counts, "split_policy": captured["split_policy"],
             "epochs_completed": completed_epochs, "requested_epochs": int(arguments["epochs"]),
             "early_stopped": completed_epochs < int(arguments["epochs"]), "best_weights": str(best),
             "best_weights_sha256": digest(best), "snapshot_report": captured["report"],
             "checkpoint_inference_verified": True, "annotations_modified_by_training": False, "medical_validation": False}
    metadata.update(proof)
    atomic_json(metadata_path, metadata)
    atomic_json(run / "completion_verification.json", proof)
    registry = {"updated_at": proof["verified_at"], "run_name": captured["run_name"], "weights": str(best),
                "trained_weights": str(best), "best_path": str(best), "snapshot": captured["snapshot"],
                "snapshot_report": captured["report"], "dataset": captured["dataset"], "base_weights": str(initial),
                "base_weights_sha256": OFFICIAL_SHA256, "fresh_from_original_pretrained": True, "resume": False,
                "approved_frames": captured["approved_frames"], "classes": {0: "lumen"},
                "imgsz": 640, "conf": .25, "retina_masks": True, "review_required": True, "medical_validation": False}
    # Last mutation: a failed training/checkpoint never replaces the active model.
    atomic_json(root / "config/inference_latest.json", registry)
    return proof


def run_fresh_training(epochs=100, batch=4, device="0", source=None, output=None, project_root=ROOT,
                       trainer_runner=None, model_loader=None, video_runner=None, tick_mm=5.0, mm_per_pixel=None):
    root = Path(project_root).resolve()
    if not isinstance(epochs, int) or epochs < 1 or not isinstance(batch, int) or batch < 1:
        raise ValueError("epochs와 batch는 1 이상의 정수여야 합니다.")
    if str(device) != "cpu" and not all(part.isdigit() for part in str(device).split(",")):
        raise ValueError("device는 cpu 또는 GPU 번호(예: 0)여야 합니다.")
    scales = [tick_mm] + ([mm_per_pixel] if mm_per_pixel is not None else [])
    if any(not math.isfinite(float(value)) or float(value) <= 0 for value in scales):
        raise ValueError("tick-mm와 mm-per-pixel은 유한한 양수여야 합니다.")
    training_config = yaml.safe_load((root / "config/train.yaml").read_text(encoding="utf-8"))
    if training_config.get("resume") not in (None, False):
        raise ValueError("config/train.yaml의 resume이 켜져 있습니다. 새 학습에서는 resume:false여야 합니다.")
    if training_config.get("pretrained") is False:
        raise ValueError("새 학습은 공식 사전학습 원본 가중치를 사용하므로 pretrained:false는 허용하지 않습니다.")
    if output is not None and not source:
        raise ValueError("--output을 지정하려면 분석할 --source도 필요합니다.")
    if source:
        try:
            from .process_videos import discover_sources
        except ImportError:
            from process_videos import discover_sources
        discover_sources(source, Path(output).expanduser().resolve() if output else None)
        if output and Path(output).expanduser().exists():
            raise ValueError("결과 폴더가 이미 존재합니다. 새 --output 폴더를 지정하세요.")
    initial = root / "models/yolo11s-seg.pt"
    if not initial.is_file() or digest(initial) != OFFICIAL_SHA256:
        raise ValueError("공식 원본 models/yolo11s-seg.pt 가중치가 없거나 SHA256이 다릅니다.")
    captured = capture_approved_snapshot(root)
    print(f"승인 {captured['approved_frames']}장 (양성 {captured['positive_frames']}/음성 {captured['negative_frames']}) · {captured['frame_counts']}", flush=True)
    print("항상 공식 원본 사전학습 모델에서 새로 시작합니다. 이전 학습 결과와 resume을 사용하지 않습니다.", flush=True)
    command = training_command(captured, root, epochs, batch, device)
    runner = trainer_runner or subprocess.run
    completed = runner(command, cwd=str(root), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"학습 종료 코드 {completed.returncode}. 기존 추론 모델은 유지됩니다.")
    proof = verify_and_publish(captured, root, model_loader)
    report = {"training": proof, "batch": None}
    print(f"새 모델 확인 및 최신 모델 등록 완료: {proof['best_weights']}", flush=True)
    if source:
        if video_runner is None:
            try:
                from .process_videos import run_batch
            except ImportError:
                from process_videos import run_batch
            video_runner = run_batch
        report["batch"] = video_runner(source_args=source, output=output, weights=proof["best_weights"], device=str(device),
                                       tick_mm=float(tick_mm), mm_per_pixel=mm_per_pixel, project_root=root)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="현재 승인 레이블로 원본 YOLO11s에서 새 학습. 이전 학습을 이어서 학습하지 않습니다.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--source", nargs="+", help="학습 후 자동 분석할 폴더 또는 영상 파일(여러 개 가능)")
    parser.add_argument("--output", help="학습 후 영상 분석을 저장할 새 결과 폴더")
    parser.add_argument("--tick-mm", type=float, default=5.0, help="학습 후 영상 분석에 사용할 인접 깊이 눈금 간격(mm), 기본 5")
    parser.add_argument("--mm-per-pixel", type=float, help="학습 후 영상 분석의 명시적 배율. 지정하면 자동 눈금 감지를 사용하지 않음")
    arguments = parser.parse_args(argv)
    try:
        report = run_fresh_training(arguments.epochs, arguments.batch, arguments.device, arguments.source, arguments.output,
                                    tick_mm=arguments.tick_mm, mm_per_pixel=arguments.mm_per_pixel)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"새 학습을 완료하지 못했습니다: {error}", file=sys.stderr)
        return 2
    return 1 if report["batch"] and report["batch"]["failed_video_count"] else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
