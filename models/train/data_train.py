from ultralytics import YOLO

if __name__ == "__main__":
    # 모델 불러오기
    model = YOLO("yolov8n.pt")

    # 학습 진행
    model.train(
        data="./dataset/data.yaml",
        epochs=50,
        imgsz=640,
        device=0 # GPU 0번 사용
    )
