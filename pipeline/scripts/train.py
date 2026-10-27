"""Train approved lumen annotations after validating a documented group split."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from ._runtime import ROOT, configure, resolve
except ImportError:
    from _runtime import ROOT, configure, resolve

UNKNOWN = {"", "unassigned", "unknown", "unconfirmed", "none", "null", "미확인", "미지정"}
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def validate_dataset(data_path, allow_exploratory=False):
    import yaml
    if not data_path.is_file():
        raise ValueError("학습 데이터가 없습니다. 먼저 레이블을 승인하고 데이터 내보내기를 실행하세요.")
    data = yaml.safe_load(data_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or "download" in data:
        raise ValueError("외부 다운로드가 포함되지 않은 로컬 데이터 YAML이 필요합니다.")
    names = data.get("names")
    if names not in (["lumen"], {0: "lumen"}, {"0": "lumen"}):
        raise ValueError("현재 모델의 학습 클래스는 lumen(혈관 내강) 하나여야 합니다.")
    report_path = data_path.parent / "split_report.json"
    if not report_path.is_file():
        raise ValueError("환자/영상 단위 분할 기록(split_report.json)이 없습니다.")
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    if report.get("approved_only") is not True:
        raise ValueError("사용자가 승인한 레이블만 포함된 데이터셋이 필요합니다.")
    policy = report.get("split_policy")
    exploratory = policy == "video_group_exploratory"
    if policy not in {"patient_group", "video_group_exploratory"}:
        raise ValueError("프레임 무작위 분할은 허용하지 않습니다. 환자 단위로 분할하세요.")
    if exploratory and not allow_exploratory:
        raise ValueError("영상 단위 탐색 학습입니다. 한 환자의 여러 영상이 섞일 수 있습니다. 확인 후 --allow-exploratory를 명시하세요.")
    group_key = "source_video_groups" if exploratory else "patient_groups"
    groups = report.get(group_key)
    if not isinstance(groups, dict):
        raise ValueError(f"분할 그룹 정보({group_key})가 없습니다.")
    seen = set()
    for split in ("train", "val", "test"):
        members = groups.get(split, [])
        if not isinstance(members, list):
            raise ValueError(f"{split} 그룹 목록 형식이 잘못되었습니다.")
        member_set = {str(x).strip() for x in members}
        if any(x.lower() in UNKNOWN for x in member_set):
            raise ValueError(f"{split}에 미지정 환자/영상이 있습니다.")
        if seen.intersection(member_set):
            raise ValueError("학습·검증·테스트에 같은 환자/영상이 중복되어 있습니다.")
        seen.update(member_set)
        if split in ("train", "val") and not member_set:
            raise ValueError(f"{split}에 독립된 환자/영상 그룹이 필요합니다.")
    source_groups = report.get("source_video_groups")
    if not isinstance(source_groups, dict):
        raise ValueError("원본 영상별 분할 정보가 없습니다.")
    seen_videos = set()
    for split in ("train", "val", "test"):
        current = {str(x).strip() for x in source_groups.get(split, [])}
        if seen_videos.intersection(current):
            raise ValueError("같은 원본 영상이 서로 다른 분할에 포함되어 있습니다.")
        if any(x.lower() in UNKNOWN for x in current):
            raise ValueError("원본 영상 ID가 미지정입니다.")
        seen_videos.update(current)
    frame_rows = report.get("frames")
    if not isinstance(frame_rows, list) or not frame_rows:
        raise ValueError("승인된 각 프레임의 분할 기록이 필요합니다.")
    frames = {}
    video_patients = {}
    for row in frame_rows:
        frame_id = row.get("frame_id")
        split = row.get("split")
        if not frame_id or frame_id in frames or split not in ("train", "val", "test"):
            raise ValueError("프레임 ID가 중복되거나 분할 값이 잘못되었습니다.")
        if row.get("status", "approved") != "approved":
            raise ValueError("미승인 프레임이 분할 기록에 있습니다.")
        video = str(row.get("video_id", "unassigned")).strip()
        patient = str(row.get("patient_id", "unassigned")).strip()
        if video not in source_groups.get(split, []):
            raise ValueError("프레임의 원본 영상과 그룹 분할 정보가 다릅니다.")
        if not exploratory:
            if patient.lower() in UNKNOWN or patient not in groups.get(split, []):
                raise ValueError("프레임의 대상자 정보와 환자 분할 정보가 다릅니다.")
            video_patients.setdefault(video, set()).add(patient)
            if len(video_patients[video]) > 1:
                raise ValueError("한 원본 영상에 여러 대상자 ID가 지정되어 있습니다.")
        frames[frame_id] = row
    dataset_root = Path(data.get("path", data_path.parent))
    if not dataset_root.is_absolute():
        dataset_root = (data_path.parent / dataset_root).resolve()
    counts = {}
    positive_counts = {}
    checked_frames = set()
    for split in ("train", "val", "test"):
        location = data.get(split)
        if split == "test" and not location:
            continue
        if not isinstance(location, str):
            raise ValueError(f"{split} 이미지 폴더 경로가 필요합니다.")
        images_dir = Path(location)
        if not images_dir.is_absolute():
            images_dir = dataset_root / images_dir
        if not images_dir.is_dir():
            raise ValueError(f"{split} 이미지 폴더가 없습니다: {images_dir}")
        images = sorted(x for x in images_dir.rglob("*") if x.suffix.lower() in EXTENSIONS)
        if split in ("train", "val") and not images:
            raise ValueError(f"{split}에 학습 가능한 이미지가 없습니다.")
        positive_counts[split] = 0
        for image in images:
            row = frames.get(image.stem)
            if row is None or row["split"] != split or image.stem in checked_frames:
                raise ValueError(f"이미지와 승인 프레임 분할 기록이 다릅니다: {image.name}")
            checked_frames.add(image.stem)
            parts = list(image.parts)
            if "images" not in parts:
                raise ValueError("YOLO 데이터는 images/<split> 및 labels/<split> 폴더 구조가 필요합니다.")
            parts[parts.index("images")] = "labels"
            label = Path(*parts).with_suffix(".txt")
            if not label.is_file():
                raise ValueError(f"레이블이 없습니다: {label.name}")
            rows = [x for x in label.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
            if not rows:
                if row.get("negative") is not True:
                    raise ValueError(f"내강 없음으로 승인되지 않은 빈 레이블입니다: {label.name}")
                continue
            if row.get("negative") is True:
                raise ValueError(f"내강 없음 프레임에 양성 polygon이 있습니다: {label.name}")
            positive_counts[split] += 1
            for line in rows:
                values = line.split()
                if len(values) < 7 or len(values) % 2 != 1 or values[0] != "0":
                    raise ValueError(f"내강 polygon 레이블 형식이 잘못되었습니다: {label.name}")
                coordinates = [float(x) for x in values[1:]]
                if not all(0 <= x <= 1 for x in coordinates):
                    raise ValueError(f"정규화 좌표가 0~1 범위를 벗어납니다: {label.name}")
        counts[split] = len(images)
        expected = report.get("frame_counts", {}).get(split)
        if expected is None or expected != len(images):
            raise ValueError(f"{split} 이미지 수와 분할 기록이 다릅니다. 데이터를 다시 내보내세요.")
        if split in ("train", "val") and positive_counts[split] == 0:
            raise ValueError(f"{split}에는 내강 양성 레이블이 최소 한 장 필요합니다.")
    if checked_frames != set(frames):
        raise ValueError("분할 기록에만 존재하는 프레임이 있습니다. 데이터를 다시 내보내세요.")
    return data, report, counts


def main():
    parser = argparse.ArgumentParser(description="승인 레이블로 혈관 내강 segmentation 학습")
    parser.add_argument("--data", default="data/export/current/dataset.yaml")
    parser.add_argument("--config", default="config/train.yaml")
    parser.add_argument("--model", help="기본 사전학습 가중치 또는 선택한 segmentation 모델 YAML")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--device")
    parser.add_argument("--name", default="fmd_lumen")
    parser.add_argument("--allow-exploratory", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    data_path = resolve(args.data)
    try:
        _, report, counts = validate_dataset(data_path, args.allow_exploratory)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"학습 준비 확인 실패: {exc}\n")
    print(f"승인 레이블과 그룹 분할 확인 완료: {counts}, {report['split_policy']}")
    if args.check_only:
        return
    import yaml
    configure()
    import torch
    from ultralytics import YOLO
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8-sig"))
    model_path = resolve(args.model or config.pop("model"))
    config.pop("model", None)
    if not model_path.is_file():
        parser.exit(2, f"모델 파일이 없습니다: {model_path}\n")
    for key in ("epochs", "batch", "imgsz", "device"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if str(config.get("device")) != "cpu" and not torch.cuda.is_available():
        parser.exit(2, "GPU를 사용할 수 없습니다. 환경 확인 또는 --device cpu를 지정하세요.\n")
    config.update(data=str(data_path), project=str(ROOT / "models" / "runs"), name=args.name)
    model = YOLO(str(model_path), task="segment")
    model.train(**config)
    run_dir = Path(model.trainer.save_dir)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "user_approved_lumen_training", "medical_validation": False,
        "data_yaml": str(data_path), "split_policy": report["split_policy"],
        "exploratory": report["split_policy"] == "video_group_exploratory", "frame_counts": counts,
        "dataset_yaml_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "split_report_sha256": hashlib.sha256((data_path.parent / "split_report.json").read_bytes()).hexdigest(),
        "initial_model": str(model_path), "config": config,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (run_dir / "training_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"학습 결과: {run_dir}")
    print("연구용 학습 결과입니다. FMD 직경/변화율에 대한 독립 검증은 별도로 수행하세요.")


if __name__ == "__main__":
    main()
