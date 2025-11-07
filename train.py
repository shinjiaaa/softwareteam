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
