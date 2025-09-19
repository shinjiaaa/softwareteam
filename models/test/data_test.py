from ultralytics import YOLO

# 학습된 모델 불러오기
model = YOLO("data/runs/detect/train5/weights/best.pt")

# 테스트 이미지에서 객체 감지
results = model.predict(
    source=".../data/dataset/valid/images/Drone-Crash-Compilation-VOL-4_184_0-210_0_mp4-3_jpg.rf.dd8264825613b9a5986fcf2f69691044.jpg",
    conf=0.25,  
)

# 결과 이미지 확인
results[0].show()
