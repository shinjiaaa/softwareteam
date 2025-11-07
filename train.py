from ultralytics import YOLO
import torch
from multiprocessing import freeze_support

def main():
    # GPU 확인
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Using device: {device}")

    # 기존 학습 모델 불러오기
    model = YOLO("YOLO-Continued/train9_finetune/weights/best.pt")  # 경로를 네 모델에 맞게 수정

    # Recall 보강 + Precision 유지용 학습
    model.train(
        data="data.yaml",          # 데이터셋 경로
        epochs=30,                 # 추가 학습 30 epoch
        batch=16,
        imgsz=640,
        lr0=0.004,                 # 약간 높여서 빠른 수렴
        optimizer="SGD",
        device=device,
        augment=True,
        mosaic=1.0,                # 작은 객체 잘 잡히도록
        mixup=0.2,                 # 이미지 혼합으로 다양한 배경 학습
        copy_paste=0.3,            # 객체 복사-붙여넣기 (Recall 향상)
        cache=True,
        patience=15,               # 조기 종료 기준
        project="YOLO-RecallBoost",
        name="train_recall_boost",
        exist_ok=True
    )

    # 평가
    metrics = model.val(data="data.yaml", device=device)
    f1_score = 2 * metrics.box.p * metrics.box.r / (metrics.box.p + metrics.box.r + 1e-6)

    print("\n=== 🔍 평가 결과 ===")
    print(f"mAP50        : {metrics.box.map50:.4f}")
    print(f"mAP50-95     : {metrics.box.map:.4f}")
    print(f"Precision_mean: {metrics.box.p.mean():.4f}")
    print(f"Recall_mean   : {metrics.box.r.mean():.4f}")
    print(f"F1_mean       : {f1_score.mean():.4f}")

if __name__ == "__main__":
    freeze_support()
    main()
