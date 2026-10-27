# 경동맥 초음파 내강 분할 및 직경 측정

권혁진의 운동심혈관생리학 발표를 위한 탐색적 구현 자료입니다. 사용자가 승인한 내강 주석으로 모델을 학습하고, 예측한 내강으로 프레임별 직경 **D (mm)**를 계산했습니다. 현재 결과는 직경이며 FMD 변화율은 계산하지 않았습니다.

## 자료 바로가기

- [분석 영상 7개](https://kwon-hj402.github.io/hjk-d-video-review/)
- [학습·성능·주석 자료 안내](https://kwon-hj402.github.io/hjk-d-video-review/research.html)
- [원본·승인 주석·마스크·예측 갤러리 215장](https://kwon-hj402.github.io/hjk-d-video-review/gallery.html)
- [초기 선별 후보 400장](https://kwon-hj402.github.io/hjk-d-video-review/gallery.html?set=candidates)
- [발표용 PowerPoint](presentation/carotid_segmentation_methods.pptx)
- [슬라이드 미리보기](https://kwon-hj402.github.io/hjk-d-video-review/presentation/)

## 방법과 근거 자료

| 문서 | 내용 |
|---|---|
| [데이터와 레이블링](docs/01_data_labeling.md) | 추출, 후보 선별, 수동 다각형, 마스크와 YOLO 좌표 |
| [모델과 학습 설정](docs/02_training.md) | 원본 YOLO11s-seg, 새 학습, 분할, 증강과 하이퍼파라미터 |
| [성능과 해석](docs/03_evaluation.md) | Dice·IoU·mask mAP, 음성 처리, 시험 자료의 한계 |
| [직경 계산 방법](docs/04_diameter.md) | 국소 중심선 수직 거리, 끝 구간 제외, 눈금 보정 |
| [재현 및 자료 구조](docs/05_reproduction.md) | 실행 방법과 포함한 자료 범위 |
| [발표 구성과 설명 원고](docs/presentation_notes.md) | 슬라이드별 설명과 교수님께 논의할 사항 |

## 학습 자료

승인된 **215장**만 사용했습니다. 양성 200장과 내강 없음으로 승인한 음성 15장입니다. 미승인 초안 5장은 제외했습니다.

| 분할 | 원본 영상 그룹 | 전체 | 양성 | 음성 |
|---|---|---:|---:|---:|
| 훈련 | 0002, 0004, 0007, 0008, 0009 | 162 | 149 | 13 |
| 검증 | 0005 | 50 | 49 | 1 |
| 시험 | 0006 | 3 | 2 | 1 |
| 합계 | 7개 영상 | 215 | 200 | 15 |

동일 영상의 프레임이 분할 사이에 섞이지 않도록 영상 단위로 나눴습니다. 분할 seed는 28, 학습 seed는 42입니다. 제공자가 자신의 경동맥 영상이라고 설명한 자료이며 대상자 독립 성능 평가가 아닙니다. 초기 후보 400장과 최종 승인 215장은 별도 집합으로, 교집합은 27장입니다.

[원본 PNG·YOLO 좌표·0/1 마스크](pipeline/data/export/current), [승인 JSON](pipeline/data/annotations), [파일별 SHA-256](research/dataset_integrity.json)을 제공합니다.

## 확인한 성능

| 지표 | 검증 | 시험 |
|---|---:|---:|
| 평가 프레임 | 50 | 3 |
| 양성 평균 Dice | 0.9653 (49장) | 0.8567 (2장) |
| 양성 평균 IoU | 0.9330 (49장) | 0.7633 (2장) |
| 음성에서 빈 예측 | 1/1 | 1/1 |

best.pt의 지표와 일치하는 epoch는 87입니다. 검증 mask mAP@0.50은 0.9950, mask mAP@0.50:0.95는 0.9039입니다. 검증 자료는 모델 선택에 사용한 내부 자료입니다. **정확도 96.5%나 직경 측정 정확도로 해석하면 안 됩니다.** 시험 자료가 매우 적고 대상자 독립 검증 및 수동 직경 대비 오차 검증을 수행하지 않았습니다.

- [지표 JSON](research/evaluation/metrics.json), [53장 프레임별 CSV](research/evaluation/per_frame_metrics.csv)
- [100 epoch 학습 로그](research/training/results.csv), [학습 그래프](research/training/results.png), [실제 인수](research/training/args.yaml)
- [실사용 best.pt](pipeline/models/best.pt), [시작 사전학습 가중치](pipeline/models/yolo11s-seg.pt)
- [한글 주석 도구와 재현 코드](pipeline/README.md)

## 전체 영상 결과

15,133프레임을 모두 보존했습니다. D 산출 12,276 / 미산출 2,857프레임입니다. 이 산출 비율은 모델 정확도가 아닙니다. [프레임별 D와 영상](videos), [영상별 처리 보고서](research/video_reports)를 제공합니다.

## 참고 논문

Chen WD, Kao YC, Chiu CH, Huang CC, Tsai MW. *Reliability of Handheld Ultrasound Assessment of Brachial Artery Flow-Mediated Dilation Using AI-Assisted Automated Analysis in Postmenopausal Women*. Medicina. 2026;62:181. [DOI](https://doi.org/10.3390/medicina62010181), [PDF](references/Chen_2026_Medicina_FMD.pdf).

참고 논문은 상완동맥·YOLOv12 기반 FMD 연구입니다. 현재 구현은 사용자 제공 경동맥 영상·YOLO11s-seg 기반 직경 측정입니다. 논문 저자의 모델이나 신뢰도 결과를 재현한 것으로 제시하지 않습니다.

## 공개 범위

소유자의 요청으로 영상·이미지·승인 레이블을 공개했습니다. 원본 AVI 전체, 15,133장 전체 추출 PNG, 실행 가상환경, 미승인 자동 초안은 포함하지 않았습니다. 학습에 사용한 215장과 초기 후보 400장의 PNG는 포함합니다.

참고 논문은 CC BY 4.0, Ultralytics 소프트웨어와 모델은 [별도 라이선스 안내](THIRD_PARTY_NOTICES.md)를 확인합니다.
