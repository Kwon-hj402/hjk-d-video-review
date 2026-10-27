"""Local SAM 2.1 prompted proposals. This module never saves/approves labels."""
import argparse
import json
import threading

try:
    from ._runtime import ROOT, configure, resolve
except ImportError:
    # The GUI may load this file under an importlib alias rather than as a
    # scripts package. Resolve the trusted sibling runtime in that case too.
    import sys
    from pathlib import Path
    sibling_directory = str(Path(__file__).resolve().parent)
    if sibling_directory not in sys.path:
        sys.path.insert(0, sibling_directory)
    from _runtime import ROOT, configure, resolve

_LOCK = threading.Lock()
_MODEL = None


def propose(image_path, box=None, points=None, labels=None):
    """Return a draft polygon in ORIGINAL image pixels, using a bbox and/or points.

    box: [x1,y1,x2,y2]. points: [[x,y],...]. labels: 1 foreground / 0 background.
    Requires a human prompt. Review boundary accuracy before approving in the UI.
    The server should pass a manifest-resolved local frame path.
    """
    global _MODEL
    import cv2
    import numpy as np
    image_path = resolve(image_path)
    if not image_path.is_file():
        raise ValueError("프레임 이미지가 없습니다.")
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("프레임 이미지를 읽을 수 없습니다.")
    h, w = image.shape[:2]
    if box is None and not points:
        raise ValueError("내강을 둘러싼 사각형 또는 내강 내부 점을 먼저 지정하세요.")
    if box is not None:
        if len(box) != 4 or not all(np.isfinite(float(x)) for x in box):
            raise ValueError("사각형은 원본 좌표 [x1,y1,x2,y2]여야 합니다.")
        box = [float(x) for x in box]
        if not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
            raise ValueError("사각형이 이미지 범위를 벗어나거나 크기가 0입니다.")
    if points is not None:
        points = [[float(p[0]), float(p[1])] for p in points if len(p) == 2]
        if not points or not all(np.isfinite(x) and 0 <= x < w and np.isfinite(y) and 0 <= y < h for x, y in points):
            raise ValueError("프롬프트 점은 이미지 안의 원본 좌표여야 합니다.")
        labels = labels if labels is not None else [1] * len(points)
        if len(labels) != len(points) or any(x not in (0, 1) for x in labels):
            raise ValueError("점마다 전경 1 또는 배경 0 레이블이 필요합니다.")
    elif labels is not None:
        raise ValueError("점 없이 점 레이블만 지정할 수 없습니다.")
    weights = ROOT / "models" / "sam2.1_t.pt"
    if not weights.is_file():
        raise ValueError("SAM 2.1 가중치 파일이 없습니다. 환경 설정을 실행하세요.")
    import torch
    from ultralytics import SAM
    warnings = ["SAM은 혈관/FMD 전용으로 학습되지 않았습니다. 경계를 확인하고 수정한 뒤 직접 승인하세요."]
    # Both singleton construction and inference must be locked: Ultralytics
    # predictors retain image embeddings and prompts between invocations.
    with _LOCK:
        if _MODEL is None:
            configure()
            _MODEL = SAM(str(weights))
        results = _MODEL.predict(
            source=image, bboxes=[box] if box is not None else None,
            points=[points] if points is not None else None,
            labels=[labels] if labels is not None else None,
            device="0" if torch.cuda.is_available() else "cpu",
            imgsz=1024, verbose=False, save=False, retina_masks=True,
        )
        candidates = []
        component_count = 0
        hole_count = 0
        raw_area = 0
        for result in results:
            if result.masks is None:
                continue
            for tensor_mask in result.masks.data:
                m = tensor_mask.detach().cpu().numpy()
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                binary = (m > 0.5).astype(np.uint8)
                raw_area += int(binary.sum())
                contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                if hierarchy is None:
                    continue
                for idx, contour in enumerate(contours):
                    if hierarchy[0][idx][3] >= 0:
                        hole_count += 1
                        continue
                    if cv2.contourArea(contour) >= 25:
                        component_count += 1
                        candidates.append(contour)
        objects = []
        retained_area = 0.0
        if candidates:
            contour = max(candidates, key=cv2.contourArea)
            retained_area = float(cv2.contourArea(contour))
            simplified = cv2.approxPolyDP(contour, epsilon=1.0, closed=True).reshape(-1, 2)
            if len(simplified) >= 3:
                objects.append({"class_id": 0, "label": "lumen", "points": simplified.astype(float).tolist()})
        if not objects:
            warnings.append("유효한 내강 후보를 찾지 못했습니다. 사각형이나 전경/배경 점을 바꾸세요.")
        if component_count > 1:
            warnings.append(f"분리된 영역 {component_count}개 중 가장 큰 영역만 제안했습니다. 누락된 영역 여부를 확인하세요.")
        if hole_count:
            warnings.append(f"마스크 내부 구멍 {hole_count}개가 있으나 polygon 내보내기에서는 채워집니다. 경계를 확인하세요.")
    return {
        "objects": objects, "model": "SAM 2.1 tiny / sam2.1_t.pt",
        "status": "draft", "approved": False, "warnings": warnings,
        "width": w, "height": h,
        "diagnostics": {"components": component_count, "holes": hole_count, "raw_mask_pixels": raw_area, "retained_polygon_area": retained_area},
        "prompt": {"box": box, "points": points, "labels": labels},
    }


def main():
    parser = argparse.ArgumentParser(description="원본 픽셀 사각형/점으로 SAM 레이블 초안 생성")
    parser.add_argument("--image", required=True)
    parser.add_argument("--box", type=float, nargs=4)
    parser.add_argument("--points", help="JSON [[x,y],...] 문자열")
    parser.add_argument("--labels", help="JSON [1,0,...] 문자열")
    parser.add_argument("--output", help="미승인 초안 JSON을 저장할 경로")
    args = parser.parse_args()
    draft = propose(args.image, args.box, json.loads(args.points) if args.points else None, json.loads(args.labels) if args.labels else None)
    payload = json.dumps(draft, ensure_ascii=False, indent=2)
    if args.output:
        output = resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
