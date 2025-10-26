from ultralytics import YOLO

def main():
    # YOLO11 pretrained 모델 불러오기
    model = YOLO("yolo11n.pt")

    # 학습 실행
    results = model.train(
        data="VisDrone.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        name="VisDrone_train"
    )

if __name__ == "__main__":
    main()
