"""Export full-resolution lumen masks as unreviewed model predictions."""
import argparse
import json
from pathlib import Path

try:
    from ._runtime import ROOT, configure, resolve
except ImportError:
    from _runtime import ROOT, configure, resolve


def main():
    parser = argparse.ArgumentParser(description="학습한 내강 모델의 미검토 예측 마스크 저장")
    parser.add_argument("--weights", required=True, help="models/runs/<학습이름>/weights/best.pt")
    parser.add_argument("--source", required=True, help="이미지 또는 이미지 폴더")
    parser.add_argument("--output", default="data/predictions")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--allow-unverified", action="store_true", help="외부 내강 가중치의 학습 기록이 없음을 확인")
    parser.add_argument("--allow-synthetic-smoke", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    weights = resolve(args.weights)
    source = resolve(args.source)
    if not weights.is_file() or not source.exists():
        parser.exit(2, "가중치 또는 입력 경로가 없습니다.\n")
    metadata_path = weights.parent.parent / "training_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        if metadata.get("purpose") == "synthetic_pipeline_smoke" and not args.allow_synthetic_smoke:
            parser.exit(2, "합성 데이터 환경 확인 모델로 환자 영상을 예측할 수 없습니다.\n")
    elif not args.allow_unverified:
        parser.exit(2, "학습 기록(training_metadata.json)이 없습니다. 외부 내강 모델이면 --allow-unverified를 명시하세요.\n")
    configure()
    import cv2
    import numpy as np
    from ultralytics import YOLO
    model = YOLO(str(weights), task="segment")
    if len(model.names) != 1 or list(model.names.values())[0] != "lumen":
        parser.exit(2, "혈관 내강(lumen)으로 학습한 1개 클래스 모델이 필요합니다. 제공된 COCO 모델은 학습 시작용입니다.\n")
    output = resolve(args.output)
    for sub in ("masks", "overlays", "polygons"):
        (output / sub).mkdir(parents=True, exist_ok=True)
    records = []
    for number, result in enumerate(model.predict(source=str(source), stream=True, device=args.device, imgsz=args.imgsz, conf=args.conf, retina_masks=True, verbose=False)):
        h, w = result.orig_img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        objects = []
        if result.masks is not None:
            for idx, item in enumerate(result.masks.data.detach().cpu().numpy()):
                item = cv2.resize(item, (w, h), interpolation=cv2.INTER_NEAREST) if item.shape != (h, w) else item
                mask[item > 0.5] = 255
                objects.append({"class_id": 0, "label": "lumen", "points": result.masks.xy[idx].tolist(), "confidence": float(result.boxes.conf[idx])})
        stem = f"{number:06d}_{Path(result.path).stem}"
        overlay = result.orig_img.copy()
        pixels = mask > 0
        overlay[pixels] = (0.55 * overlay[pixels] + 0.45 * np.array([190, 216, 54])).astype(np.uint8)
        for sub, im in (("masks", mask), ("overlays", overlay)):
            ok, encoded = cv2.imencode(".png", im)
            if not ok:
                raise RuntimeError("예측 이미지 저장 실패")
            encoded.tofile(output / sub / f"{stem}.png")
        record = {"source": str(result.path), "width": w, "height": h, "status": "prediction_unreviewed", "approved": False, "objects": objects, "mask": f"masks/{stem}.png", "overlay": f"overlays/{stem}.png"}
        (output / "polygons" / f"{stem}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        records.append(record)
    (output / "prediction_report.json").write_text(json.dumps({"weights": str(weights), "approved": False, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"미검토 예측 {len(records)}개 저장: {output}")


if __name__ == "__main__":
    main()
