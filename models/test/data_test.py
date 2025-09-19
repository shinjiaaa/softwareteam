from ultralytics import YOLO

# 학습된 모델 불러오기
model = YOLO("runs/detect/train5/weights/best.pt")

# 테스트 이미지에서 객체 감지
results = model.predict(
    source="./dataset/valid/images/Drone-Crash-Compilation-VOL-4_0_0-10_0_mp4-2_jpg.rf.183f13076b52ad2cb8170abb1533b72a.jpg",
    conf=0.25,  
)

# 결과 이미지 확인
results[0].show()
