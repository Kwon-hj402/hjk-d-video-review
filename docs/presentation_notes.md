# 발표 구성과 설명 원고

## 1. 경동맥 초음파 내강 분할 및 직경 측정

이 발표는 참고 논문의 AI 혈관 분석 접근을 바탕으로, 실제 사용자 영상에서 내강 분할과 직경 계산을 구현한 과정을 설명한다. 임상 FMD 신뢰도 연구를 재현했다는 주장은 하지 않는다. 발표자는 운동심혈관생리학 수강생 권혁진이며, 대학원 체육학과 석박사 통합과정이다.

근거: https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/README.md

## 2. 참고 논문과 현재 구현

참고 논문은 Chen 등의 2026년 Medicina 논문으로, 폐경 후 여성 17명의 상완동맥 FMD를 YOLOv12 기반 분석으로 두 날짜에 평가하였다. 논문에서 모델 개발용 영상과 신뢰도 평가 영상을 분리한 점이 중요하다. 현재 구현은 사용자가 제공한 경동맥 영상에서 YOLO11s-seg를 학습하여 D만 계산한 탐색 실험이다. 논문의 ICC나 FMD 신뢰도 수치를 현재 모델 성능으로 사용하지 않는다. 저자의 가중치나 동일 프로토콜을 그대로 재현한 실험도 아니다. 논문의 개발 이미지 수는 훈련 360, 독립 시험 40으로 설명되지만, 우리의 후보 400장은 별도의 자체 선별 집합이다.

근거: https://doi.org/10.3390/medicina62010181 · https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/references/Chen_2026_Medicina_FMD.pdf · https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/docs/02_training.md

## 3. 영상 자료와 학습 분할

원본은 640×448, 28.775fps의 AVI 7개로 총 15,133프레임이다. 모든 프레임을 추출했고, 학습에는 승인된 215장만 사용하였다. 양성은 200장, 내강 없음으로 승인한 음성은 15장이다. 같은 영상의 인접 프레임이 훈련과 검증에 동시에 들어가지 않도록 원본 영상 그룹을 분리했다. 분할 seed는 28이며 학습 seed 42와 별개다. 제공자가 자신의 영상이라고 설명한 자료이고 patient_id도 미지정이므로 대상자 독립 성능을 평가한 것으로 해석하지 않는다.

근거: https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/pipeline/data/export/current/split_report.json · https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/research/dataset_integrity.json

## 4. 프레임 추출과 초기 후보 선별

PNG는 AVI에서 디코딩한 픽셀을 전체 크기로 보존한다. 초기 400장 후보를 고를 때 선명도, 대비, 어두운 띠의 규칙성과 시간·모양의 다양성을 화질 대리지표로 사용하였다. 화질 ROI는 100,125,540,365이며 저장 이미지의 crop이 아니다. 초기 후보와 최종 승인 스냅샷은 같은 집합이 아니다. 최종 215장 중 후보 400장과 겹치는 것은 27장이고, 나머지는 사용자가 전체 영상에서 추가로 선택하여 주석한 프레임이다. 초기 후보를 곧바로 정답 레이블로 사용하지 않았다.

근거: https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/research/candidate400/manifest.json · https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/research/selection_summary.json · https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/docs/01_data_labeling.md

## 5. 한글 작업대에서 내강 다각형 주석

실제 공개 스냅샷을 로컬 주석 도구에서 연 화면이다. 승인 레이블은 215장이고, 왼쪽의 초기 선별 교집합 27장은 별도 카운터다. 전체 프레임 탭에서 215장을 확인할 수 있다. 내강 안쪽 경계를 따라 꼭짓점을 클릭하고 Enter로 닫는다. 편집 모드에서 꼭짓점을 이동하거나 추가·삭제한 뒤 승인한다. 내강이 없다고 판단한 프레임은 음성으로 승인한다. 그림의 자동 제안 버튼은 별도 SAM 가중치가 필요한 선택 기능으로, 현재 학습에는 미승인 자동 초안을 쓰지 않았다. 주석은 사용자 승인 참조이며 전문가 합의 주석은 아니다.

근거: https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/research/figures/labeling_workbench.jpg · https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/pipeline/app/static/app.js · https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/pipeline/app/server.py

## 6. 주석 좌표와 이진 마스크

예시는 v01_f000117이다. 왼쪽은 원본 PNG, 가운데는 승인 다각형과 꼭짓점의 표시 자료, 오른쪽은 승인 다각형에서 채운 이진 마스크다. 실제 학습은 색칠 이미지가 아니라 원본 PNG와 YOLO 다각형 TXT를 사용한다. YOLO TXT의 클래스 번호 0은 lumen이다. 한편 참조 PNG는 배경 0, 내강 1이며 보기용 PNG만 255로 확대했다. YOLO가 내부에서 학습 마스크를 만드는 방식과 Pillow 내보내기 마스크의 경계 픽셀은 미세하게 다를 수 있다. 음성 프레임은 빈 좌표 파일과 모든 픽셀이 0인 마스크로 대응한다.

근거: https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/pipeline/data/annotations/v01_f000117.json · https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/pipeline/data/export/current/labels/train/v01_f000117.txt · https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/pipeline/data/export/current/masks/train/v01_f000117.png · https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/pipeline/scripts/export_dataset.py

## 7. 원본 사전학습 모델에서 새 학습

공식 YOLO11s-seg COCO 사전학습 가중치에서 lumen 한 클래스로 전이학습했다. 이전 사용자 학습 모델을 이어 학습한 것이 아니고 무작위 초기화도 아니다. fresh_from_original_pretrained=true, resume=false를 기록했다. 요청한 100 epoch를 모두 완료했으며 early stopping은 발생하지 않았다. 최적 가중치와 마지막 가중치는 구별한다. 실제 1클래스 모델의 파라미터는 10,082,675개다. 전체 증강과 라이브러리 기본 인수는 args.yaml에 남겨 두었다.

근거: https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/research/training/args.yaml · https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/research/training/training_metadata.json · https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/research/training/completion_verification.json · https://docs.ultralytics.com/models/yolo11/

## 8. 전처리와 데이터 증강

학습에는 crop하지 않은 640×448 전체 프레임을 사용했다. 후보 화질 점수의 ROI와 직경 측정 ROI는 학습 crop과 다르다. 모델 입력은 Ultralytics의 비율 유지 resize와 padding 및 0–1 스케일링을 거친다. 학습 rect=false와 추론 최소 패딩 동작을 혼동하여 항상 640×640으로 추론한다고 설명하지 않는다. 별도의 CLAHE, speckle 제거 또는 대비 보정을 하지 않았다. 증강은 작은 회전, 이동, 크기 변화만 허용했다. retina_masks=true로 추론한 마스크는 원본 640×448에서 평가하고 직경을 구한다.

근거: https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/pipeline/config/train.yaml · https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/research/training/args.yaml · https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/pipeline/scripts/predict.py · https://docs.ultralytics.com/tasks/segment/

## 9. 학습 곡선과 가중치 선택

왼쪽은 100 epoch의 훈련 및 검증 segmentation loss, 오른쪽은 검증 mask mAP@0.50:0.95이다. 표와 그래프의 숫자는 results.csv의 실제 100행을 사용했다. best.pt 내부 train_metrics는 87 epoch와 일치하며 mask mAP는 0.90387이다. 100 epoch 마지막 행은 0.89432이다. 검증 지표를 이용해 최적 가중치를 선택했으므로 같은 검증 자료의 수치를 최종 외부 성능으로 해석하지 않는다. PPT 그래프와 데이터는 편집 가능한 차트다.

근거: https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/research/training/results.csv · https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/research/training/results.png · https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/pipeline/models/best.pt

## 10. 원본 해상도 마스크 일치도

승인 참조 마스크와 예측 마스크를 각 프레임의 원본 해상도에서 비교하였다. Dice는 2배 교집합을 두 마스크 면적 합으로 나눈 값이고, IoU는 교집합/합집합이다. 표는 양성 프레임의 평균이며 음성은 별도 집계한다. 검증은 49양성·1음성, 시험은 2양성·1음성이다. 두 마스크가 모두 빈 경우 점수를 임의로 0 또는 1로 합치지 않는다. 기존 review.json은 두 빈 마스크를 0으로 처리해 검증 전체 Dice 0.94596, 시험 전체 Dice 0.57112를 기록했다. 현재 지표는 양성 일치도와 음성 탐지를 분리한 정의다. 공개 체크포인트로 53장을 재추론하여 동일 수치를 확인했다. 시험 3장으로 일반화 정확도를 단정할 수 없다.

근거: https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/research/evaluation/metrics.json · https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/research/evaluation/per_frame_metrics.csv · https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/pipeline/scripts/evaluate_model.py

## 11. 예측 사례와 경계 오차

왼쪽은 검증 영상 v03_f001276로 Dice 0.9580이다. 오른쪽은 시험 영상 v04_f001982로 Dice 0.7548이다. 청록은 사용자 승인 주석이고 주황은 모델 예측이다. 오른쪽에서는 예측이 승인 주석보다 넓게 퍼져 같은 영상에서도 오차가 커질 수 있음을 보여준다. 좋은 예시만 제시하지 않으며, 현재 시험 자료에는 양성 2장뿐이라는 점을 함께 설명한다. 이 그림은 사용자 승인 주석에 대한 픽셀 일치도이고 해부학적 정답이나 직경 정확도를 검증한 그림이 아니다.

근거: https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/research/evaluation/overlays/v03_f001276.png · https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/research/evaluation/overlays/v04_f001982.png · https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/research/evaluation/per_frame_metrics.csv

## 12. 국소 중심선에 수직인 내강 직경

혈관이 기울어졌다고 단순 세로 폭을 계산하면 직경이 과대평가될 수 있다. 현재 구현은 가장 큰 연결 내강에서 PCA로 종축 좌표계를 잡고, 두 경계의 중간 위치를 평활화하여 국소 중심선을 구한다. 양끝 15%씩을 제외한 중앙 70%에서 최대 81지점의 접선 수직 법선을 따라 두 벽 사이 거리를 구한다. PCA의 전역 축 폭 자체를 직경으로 쓰지 않고, 실제 이미지를 수평 회전하여 저장하지도 않는다. 그림은 최종 0008 분석 영상의 실제 대표 프레임이다.

근거: https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/pipeline/scripts/measurement.py · https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/pipeline/scripts/diameter_quality.py · https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/research/video_reports/EBL_20260904_CAROTID_0008.json · https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/research/figures/diameter_geometry_0008.jpg

## 13. 눈금 보정과 미산출 기준

사용자가 인접 눈금 한 칸을 0.5cm라고 확인해 5mm를 적용하였다. 원본 왼쪽 눈금에서 4개 이상 균일한 간격을 확인했고, 평균 64.1픽셀에 해당하여 0.07800312mm/픽셀이다. 가로·세로 표시 픽셀이 등방성이라는 가정을 사용한다. 다른 장비나 깊이 설정에서 같은 값을 자동 대입해서는 안 된다. 내강 조각이 너무 짧거나 경계가 불명확하면 D를 비워 둔다. 기준은 현재 자료에 적용한 경험적 규칙으로 외부 검증한 최적 임계값이 아니다.

근거: https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/pipeline/scripts/calibration.py · https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/pipeline/scripts/measurement.py · https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/pipeline/scripts/diameter_quality.py · https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/research/figures/depth_tick_crop.png

## 14. 7개 전체 영상의 직경 산출 결과

전체 영상 결과는 15,133프레임을 원본 순서와 fps 그대로 보존한다. 12,276프레임에서 직경을 계산했고 2,857프레임은 산출 불가로 남겼다. 종축 길이 기준을 추가해 짧은 조각 87프레임을 제외한 최종 검토 결과다. 그래프는 각 영상의 산출 프레임 비율이며 정답 대비 정확도가 아니다. 훈련에 사용한 영상도 이 7개에 포함되어 전체 영상 산출 비율을 독립 성능처럼 보고하지 않는다. 원본·분석 좌우 비교 MP4, 분석 MP4와 프레임별 CSV가 저장소에 있다.

근거: https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/summary.json · https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/research/video_reports/ · https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/videos/

## 15. 해석의 한계와 후속 검증

현재 결과는 소규모 탐색 실험이다. 동일 제공자의 영상에서 주석과 예측이 얼마나 일치하는지 확인했지만 환자 독립 검증이나 전문가 간 주석 신뢰도를 평가하지 않았다. 시험 3장 중 양성은 2장이다. 직경을 실제 수동 측정과 비교한 MAE나 bias, ICC, Bland–Altman 분석이 없으므로 FMD 측정의 정확도나 신뢰도를 주장할 수 없다. 교수님께는 독립 대상자·장비를 나누는 검증 설계, 반복 주석 및 수동 직경 참조, 상완동맥 FMD 촬영 조건과 커프·심장주기 정보 확보 순서에 대해 의견을 구하고자 한다.

근거: https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/docs/03_evaluation.md · https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/docs/04_diameter.md · https://doi.org/10.3390/medicina62010181

## 16. 공개 자료와 재현 경로

저장소에는 발표 PPT, 승인 원본 215장과 다각형·마스크, 초기 후보 400장, 검증·시험 53장 예측과 지표, 100 epoch 학습 로그, 실제 모델과 시작 가중치, 한글 주석 도구 및 영상 처리 코드를 함께 제공한다. 원본 AVI 전체와 15,133장 전체 추출 PNG는 별도 원본 자료이므로 포함하지 않았다. 학습과 주석 저장은 로컬 Python에서 실행하며 Pages에서는 영상과 갤러리를 읽기 전용으로 확인한다. 모델과 참고 논문의 별도 라이선스도 저장소에 명시했다.

근거: https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/README.md · https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/pipeline/README.md · https://github.com/Kwon-hj402/hjk-d-video-review/blob/main/research/dataset_integrity.json
