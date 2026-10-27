# 모델과 학습

## 모델과 초기화

Ultralytics **YOLO11s-seg**의 공식 yolo11s-seg.pt 사전학습 가중치에서 lumen 한 클래스에 전이학습했습니다. 학습 후 파라미터는 10,082,675개입니다. 참고 논문의 YOLOv12 가중치를 사용한 실험은 아닙니다.

이전 사용자 학습 가중치에서 이어 학습하지 않고 원본 사전학습 모델을 초기값으로 새 실행을 시작했습니다. fresh_from_original_pretrained=true, resume=false입니다. 무작위 가중치에서 시작한 학습과도 다릅니다.

| 항목 | 실제 설정 |
|---|---|
| 실행 | fresh_lumen_20261008T142258Z |
| 승인 자료 | 215장, 양성 200 / 음성 15 |
| 훈련 / 검증 / 시험 | 162 / 50 / 3 |
| imgsz / batch | 640 / 4 |
| epoch | 요청 100, 완료 100 |
| optimizer | AdamW |
| 초기 학습률 / weight decay | 0.001 / 0.0005 |
| seed / deterministic | 42 / true |
| patience / 실제 조기 종료 | 25 / 없음 |
| AMP / cache | false / false |
| mask_ratio / overlap_mask | 1 / true |
| GPU | NVIDIA RTX 5070 Ti |
| 환경 | Python 3.13.11, torch 2.13.0+cu130, torchvision 0.28.0+cu130, Ultralytics 8.4.110 |

## 분할 정책

영상 그룹을 분리했습니다. 분할 seed는 28이며 학습 seed 42와 다릅니다. 승인 수가 가장 큰 그룹은 훈련, 두 번째는 검증, 나머지 중 가장 작은 그룹은 시험에 배정하도록 학습 전에 정했습니다.

훈련 v01=0002, v02=0004, v05=0007, v06=0008, v07=0009. 검증 v03=0005. 시험 v04=0006입니다. 같은 원본 영상은 여러 분할에 중복되지 않습니다.

제공자는 자신의 영상이라고 설명했고 patient_id는 unassigned입니다. **영상 분리는 대상자 독립 검증이 아닙니다.** 시험 3장으로 일반화 성능을 안정적으로 평가할 수 없습니다.

## 전처리와 증강

원본 전체 프레임 640×448을 사용했습니다. 선별 ROI로 이미지를 잘라 학습하지 않았습니다. 명시적 CLAHE, speckle 제거, 추가 대비 보정은 적용하지 않았습니다. 라이브러리의 비율 유지 리사이즈·패딩과 0–1 입력 스케일링을 사용합니다. 추론은 retina_masks=true로 원본 해상도 마스크를 받습니다.

회전 ±3도, 이동 0.02, scale 0.1을 사용했습니다. HSV 변화, 상하·좌우 반전, mosaic, mixup, cutmix, copy-paste는 껐습니다. [전체 학습 인수](../research/training/args.yaml)에는 다른 작업에도 공통인 기본 인수가 있으므로 모두 분할 학습에 적용한 것으로 단정하지 않습니다.

## 최적 체크포인트

best.pt 내부 지표는 results.csv의 **87 epoch** 행과 일치합니다. 그 시점 mask mAP@0.50:0.95는 0.90387, 최종 100 epoch에서는 0.89432입니다. 마지막 가중치와 최적 가중치를 구별합니다. 검증은 모델 선택에 사용한 자료입니다.

- 시작 가중치 SHA-256: 1caa81c0195412efa411b632bcfb8c184939dddb6ae41f6a80c41b211ff257c3
- 실사용 best.pt SHA-256: 8ad1e5f0a47e033db4115b4e0895abda7175ae0c09e08597f1d38c6bcd187d9a

[학습 로그](../research/training/results.csv) / [학습 설정](../pipeline/config/train.yaml) / [공식 YOLO11 문서](https://docs.ultralytics.com/models/yolo11/) / [분할 모델 문서](https://docs.ultralytics.com/tasks/segment/)
