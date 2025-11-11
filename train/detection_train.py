from ultralytics import YOLO
import torch
import numpy as np

def main():
    # GPU 사용 여부 확인
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Using device: {device}")

    # GPU 캐시 비우기 (MemoryError 방지)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 기존 학습 모델 불러오기
    model = YOLO("YOLO-Continued/train9_finetune/weights/best.pt")

    # 학습 설정
    model.train(
        data="data.yaml",
        epochs=80,
        batch=8,                # 메모리 절약
        imgsz=640,
        lr0=0.0015,             # 안정적 학습률
        optimizer="SGD",
        device=device,

        # 데이터 증강 설정
        augment=True,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10, translate=0.1, scale=0.5, shear=2.0,
        flipud=0.1, fliplr=0.5,
        mosaic=0.5,             # mosaic 줄임 (메모리 절약)
        mixup=0.1,
        copy_paste=0.3,

        # 학습 안정화 관련 옵션
        cache=False,            # 메모리 절약
        workers=0,              # ⚡ Windows에서 필수
        patience=20,
        project="YOLO-Continued",
        name="train_merge_finetune",
        exist_ok=True
    )

    # 평가 수행
    metrics = model.val(data="data.yaml", device=device)

    # F1 계산
    f1 = 2 * metrics.box.p * metrics.box.r / (metrics.box.p + metrics.box.r + 1e-6)

    # 평가 결과 출력
    print("\n=== 평가 결과 ===")
    print(f"mAP50        : {metrics.box.map50:.4f}")
    print(f"mAP50-95     : {metrics.box.map:.4f}")
    print(f"Precision_mean: {np.mean(metrics.box.p):.4f}")
    print(f"Recall_mean   : {np.mean(metrics.box.r):.4f}")
    print(f"F1_mean       : {np.mean(f1):.4f}")

if __name__ == "__main__":
    main()
