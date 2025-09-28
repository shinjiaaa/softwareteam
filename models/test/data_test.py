from ultralytics import YOLO

# 학습된 모델 불러오기
model = YOLO("models/weights/yolov8n_custom.pt")

# 테스트 이미지에서 객체 감지
results = model.predict(
    source="data/dataset/valid/images",  # 전체 검증 데이터셋 사용
    conf=0.25,
    save=True  # 결과 저장
)

# 결과 이미지 확인
results[0].show()
