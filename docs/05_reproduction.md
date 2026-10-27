# 자료 구조와 재현

실행 방법은 [pipeline/README.md](../pipeline/README.md)에 있습니다. 발표 표·그래프의 근거는 학습 로그, 분할 기록과 평가 마스크입니다.

| 경로 | 자료 |
|---|---|
| videos | 비교·분석 MP4, D CSV, 대표 이미지 |
| pipeline/data/export/current/images | 학습 원본 PNG 215장 |
| pipeline/data/export/current/labels | YOLO 다각형 215개 |
| pipeline/data/export/current/masks | 0/1 참조 마스크 215개 |
| pipeline/data/annotations | 승인 JSON 215개 |
| research/annotations, research/mask_display | 전체 승인 프레임의 보기용 주석·마스크 |
| research/evaluation | 검증·시험 53장 예측과 Dice·IoU |
| research/candidate400 | 초기 후보 400장과 학습 포함 여부 |
| research/training | 학습 CSV·그래프·설정·완료 기록 |
| pipeline/models | 실사용 best.pt와 원본 사전학습 가중치 |
| pipeline/app, pipeline/scripts | 한글 주석, MCP, 추출·학습·평가·직경 계산 코드 |
| references | 인용 정보와 참고 논문 |
| presentation | 편집 가능한 PPTX와 슬라이드 미리보기 |

[SHA-256](../research/dataset_integrity.json)으로 원본 이미지·0/1 마스크·승인 JSON의 무결성을 확인합니다. 색칠 오버레이와 0/255 표시 마스크는 별도 시각화 자료입니다.

원본 AVI 전체와 15,133장 전체 추출 PNG는 포함하지 않았습니다. 공개 자료에 학습 PNG 215장과 후보 PNG 400장을 포함합니다. 7개 전체 영상 분석을 다시 실행하려면 원본 AVI를 --source로 지정해야 합니다.

원래 로컬 작업대에서 코드를 복사하고 공개용 경로·재평가 도구를 추가했습니다. 사용자 작업대의 승인 파일이나 모델을 덮어쓰지 않았습니다. GitHub Pages는 결과를 보여주는 정적 페이지이고 실제 학습·레이블 저장은 Python 환경에서 실행합니다.
