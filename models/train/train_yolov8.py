from ultralytics import YOLO

# 모델 생성
model = YOLO('yolov8n.pt')  # 'n'은 nano 버전, 다른 옵션: 's'(small), 'm'(medium), 'l'(large), 'x'(xlarge)

# 학습 설정
results = model.train(
    data='data/dataset/data.yaml',         # 데이터셋 설정 파일 경로
    epochs=100,                            # 학습 에포크 수
    imgsz=640,                            # 입력 이미지 크기
    batch=16,                             # 배치 크기
    device='0',                           # GPU 장치 번호 (CPU만 사용 시 'cpu')
    workers=8,                            # 데이터 로딩 워커 수
    project='models/weights',             # 결과 저장 프로젝트 폴더
    name='yolov8n_custom',               # 결과 모델 이름
    exist_ok=True                         # 기존 모델이 있어도 덮어쓰기
)

# 학습된 모델로 검증 수행
results = model.val()