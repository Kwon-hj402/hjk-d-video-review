# 재현 가능한 학습·주석·영상 처리 코드

모든 명령은 이 `pipeline` 폴더에서 실행합니다. 원본 작업대와 분리한 공개 스냅샷이므로 여기서 레이블을 수정해도 기존 로컬 작업 파일은 바뀌지 않습니다.

## 환경

기록된 환경은 Python 3.13.11, PyTorch 2.13.0+cu130, torchvision 0.28.0+cu130, Ultralytics 8.4.110, RTX 5070 Ti입니다. GPU 패키지는 PyTorch의 공식 CUDA 패키지 저장소에서 설치한 뒤 나머지 의존성을 설치합니다. 실제 사용 버전은 `requirements.txt`와 `requirements-local.lock`에 있습니다. `video_pipeline.py`의 MP4 출력에는 `imageio-ffmpeg`가 필요합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install ultralytics==8.4.110 numpy==2.4.4 opencv-python==5.0.0.93 Pillow==12.2.0 PyYAML==6.0.3 fastapi==0.143.0 uvicorn==0.54.0 mcp==1.30.0 imageio-ffmpeg==0.6.0
```

GPU 사용 가능 여부와 OS에 맞는 PyTorch 설치는 [공식 설치 안내](https://pytorch.org/get-started/locally/)에서 확인합니다. GPU가 없으면 추론·평가는 `--device cpu`로 실행할 수 있습니다.

## 원본 이미지·주석·마스크 확인

```powershell
python scripts/verify_published_dataset.py
python scripts/train.py --check-only --allow-exploratory
```

`data/export/current/images/<split>`은 원본 크기 PNG, `labels/<split>`은 YOLO segmentation 좌표, `masks/<split>`은 0/1 이진 마스크입니다. `data/annotations`는 승인된 원본 다각형 JSON입니다. 같은 대상자의 영상이 여러 분할에 포함될 수 있는 탐색 실험이며, 대상자 독립 검증은 수행하지 않았습니다.

## 로컬 한글 레이블링 도구

```powershell
python -m uvicorn app.server:app --host 127.0.0.1 --port 8765
```

`http://127.0.0.1:8765/?view=all&manual=1`에서 215개 프레임을 확인합니다. 꼭짓점을 클릭하고 Enter로 윤곽을 닫은 후, 꼭짓점 편집과 승인 저장을 사용할 수 있습니다. 자동 경계 제안은 별도 SAM 가중치가 필요한 선택 기능이며, 이 공개 스냅샷의 승인 데이터에는 미승인 자동 초안을 포함하지 않았습니다. MCP 연결용 코드는 `app/mcp_server.py`에 있습니다. GitHub Pages의 갤러리는 읽기 전용이며 저장 기능은 로컬 도구에서 제공합니다.

## 원본 사전학습 모델에서 새로 학습

```powershell
python scripts/train.py --allow-exploratory --model models/yolo11s-seg.pt --epochs 100 --batch 4 --device 0 --name reproduced_lumen
```

`config/train.yaml`을 사용합니다. `resume`으로 이전 학습을 이어 가지 않고 원본 가중치에서 전이학습합니다. 분할은 제공한 스냅샷을 그대로 사용합니다. 새 실행의 수치는 라이브러리·GPU 연산 환경에 따라 기록된 실행과 달라질 수 있습니다. 현재 사용한 가중치는 `models/best.pt`이며 원래 실행 결과입니다.

## 성능 재평가

```powershell
python scripts/evaluate_model.py --output reports/evaluation_new --device cpu
```

기존 폴더를 덮어쓰지 않도록 새 출력 경로를 지정합니다. 검증 50장과 시험 3장의 원본 해상도 마스크를 평가합니다. 두 마스크가 모두 빈 음성 프레임은 Dice/IoU를 비워 두고, 음성 프레임의 예측 유무를 별도로 집계합니다.

## 다른 영상의 내강 직경 처리

```powershell
python scripts/process_videos.py --source "D:\source_videos" --output "D:\diameter_results_new" --device 0 --tick-mm 5 --roi "94,40,545,382"
```

폴더 안의 원본 영상을 재귀 검색하여 원본·분석 비교 MP4, 분석 MP4, 프레임별 CSV와 보고서를 생성합니다. `--roi`는 현재 장비의 640×448 촬영 영역 예시입니다. 다른 장비에서는 촬영 영역을 직접 확인한 좌표로 지정하거나 이 인수를 생략합니다. `--tick-mm 5`도 이 영상의 인접 눈금이 5 mm라는 사용자 확인에 근거한 값입니다. 눈금 검출에 실패하면 임의의 mm 값을 대입하지 않습니다.

현재 공개한 7개 결과를 재현할 때는 원본 AVI가 별도로 필요합니다. 저장소에는 결과 MP4 및 학습/후보 PNG를 포함하고, 원본 AVI 전체와 15,133장의 전체 추출 PNG는 포함하지 않았습니다.
