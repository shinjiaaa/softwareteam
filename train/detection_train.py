<<<<<<< HEAD:train.py
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
=======
import os
import numpy as np
from ultralytics import YOLO
import multiprocessing

# =========================
# 환경 설정
# =========================
MODEL_PATH = "YOLO-Continued/train9_finetune/weights/best.pt"  # 이어서 학습할 기존 weight
DATA_YAML = "data.yaml"  # train/val/test 정의된 YAML

EPOCHS = 20          # 추가 학습
BATCH_SIZE = 16
IMG_SIZE = 640
LR = 0.002           # 이어 학습용 안정적 LR
DEVICE = 0           # GPU 번호

# =========================
# 클래스별 Augmentation 가중치 설정
# =========================
# Recall 낮은 클래스 인덱스를 지정 (예: 2: building, 4: other)
LOW_RECALL_CLASSES = [2, 4]  

# =========================
# 모델 학습 및 평가 함수
# =========================
def train_and_evaluate():
    # 모델 불러오기
    model = YOLO(MODEL_PATH)
    print(f"\n📌 이어 학습 시작: {MODEL_PATH}")

    # 추가 학습
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        lr0=LR,
        optimizer="SGD",
        device=DEVICE,
        augment=True,   # 일반 Augmentation
        cache=True,
        project="YOLO-Continued",
        name="finetune_recall_boost",
        exist_ok=True
    )

    # 학습 완료 후 평가
    metrics = model.val(data=DATA_YAML, device=DEVICE)
    f1_score = 2 * metrics.box.p * metrics.box.r / (metrics.box.p + metrics.box.r + 1e-6)

    print("\n=== 평가 결과 ===")
    print(f"mAP50        : {metrics.box.map50:.4f}")
    print(f"mAP50-95     : {metrics.box.map:.4f}")
    print(f"Precision_mean: {np.mean(metrics.box.p):.4f}")
    print(f"Recall_mean   : {np.mean(metrics.box.r):.4f}")
    print(f"F1_mean       : {np.mean(f1_score):.4f}")

    # 클래스별 Precision/Recall
    print("\n=== 클래스별 Precision / Recall ===")
    print("Precision:", np.round(metrics.box.p, 4))
    print("Recall   :", np.round(metrics.box.r, 4))


# =========================
# 메인 실행
# =========================
if __name__ == "__main__":
    multiprocessing.freeze_support()
    train_and_evaluate()
>>>>>>> d2ef84aa78a2db5ae5e43e568e1a46fea6939113:train/detection_train.py
