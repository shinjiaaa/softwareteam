# from ultralytics import YOLO
# import torch
# import numpy as np

# def main():
#     # GPU 사용 여부 확인
#     device = 0 if torch.cuda.is_available() else 'cpu'
#     print(f"🚀 Using device: {device}")

#     # GPU 캐시 비우기 (MemoryError 방지)
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

#     # 기존 학습 모델 불러오기
#     model = YOLO("YOLO-Continued/train9_finetune/weights/best.pt")

#     # 학습 설정
#     model.train(
#         data="data.yaml",
#         epochs=70,
#         batch=8,                # 메모리 절약
#         imgsz=640,
#         lr0=0.003,             # 안정적 학습률
#         optimizer="SGD",
#         device=device,

#         # 데이터 증강 설정
#         augment=True,
#         hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
#         degrees=10, translate=0.1, scale=0.5, shear=2.0,
#         flipud=0.1, fliplr=0.5,
#         mosaic=0.5,             # mosaic 줄임 (메모리 절약)
#         mixup=0.1,
#         copy_paste=0.3,

#         # 학습 안정화 관련 옵션
#         cache=False,            # 메모리 절약
#         workers=0,              # 멀티스레드 비활성화
#         patience=20,
#         project="YOLO-Continued",
#         name="train_merge_finetune",
#         exist_ok=True
#     )

#     # 평가 수행
#     metrics = model.val(data="data.yaml", device=device)

#     # F1 계산
#     f1 = 2 * metrics.box.p * metrics.box.r / (metrics.box.p + metrics.box.r + 1e-6)

#     # 평가 결과 출력
#     print("\n=== 평가 결과 ===")
#     print(f"mAP50        : {metrics.box.map50:.4f}")
#     print(f"mAP50-95     : {metrics.box.map:.4f}")
#     print(f"Precision_mean: {np.mean(metrics.box.p):.4f}")
#     print(f"Recall_mean   : {np.mean(metrics.box.r):.4f}")
#     print(f"F1_mean       : {np.mean(f1):.4f}")

# if __name__ == "__main__":
#     main()





# 라벨 변환
# import os

# # YOLO 데이터셋 기본 경로
# base_path = r"C:/Users/lab/softwareteam/datasets"

# # 라벨 폴더 경로들 (train, val, test)
# label_dirs = [
#     os.path.join(base_path, "labels", "train"),
#     os.path.join(base_path, "labels", "val"),
#     os.path.join(base_path, "labels", "test"),
# ]

# # 변환할 클래스 번호
# old_class = "4"  # building
# new_class = "3"  # other

# for label_dir in label_dirs:
#     print(f"=== Checking folder: {label_dir} ===")
#     for root, _, files in os.walk(label_dir):
#         for file in files:
#             if not file.endswith(".txt"):
#                 continue

#             file_path = os.path.join(root, file)

#             with open(file_path, "r") as f:
#                 lines = f.readlines()

#             new_lines = []
#             changed = False
#             for line in lines:
#                 parts = line.strip().split()
#                 if len(parts) == 0:
#                     continue

#                 if parts[0] == old_class:
#                     parts[0] = new_class
#                     changed = True

#                 new_lines.append(" ".join(parts) + "\n")

#             # 변경된 경우에만 파일 덮어쓰기
#             if changed:
#                 with open(file_path, "w") as f:
#                     f.writelines(new_lines)
#                 print(f"Updated: {file_path}")

# print("\n✅ 모든 2번 클래스가 4번으로 변경 완료!")




# 클래스 라벨 확인
import os
from collections import Counter

# YOLO 데이터셋 기본 경로
base_path = r"C:/Users/lab/softwareteam/datasets"

# 라벨 폴더 경로들 (train, val, test)
label_dirs = [
    os.path.join(base_path, "labels", "train"),
    os.path.join(base_path, "labels", "val"),
    os.path.join(base_path, "labels", "test"),
]

# 전체 클래스 개수 저장용
total_counts = Counter()

for label_dir in label_dirs:
    class_counts = Counter()
    for root, _, files in os.walk(label_dir):
        for file in files:
            if not file.endswith(".txt"):
                continue

            file_path = os.path.join(root, file)

            with open(file_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) == 0:
                    continue
                cls = parts[0]
                class_counts[cls] += 1
                total_counts[cls] += 1

    print(f"\n📁 폴더: {label_dir}")
    if class_counts:
        for cls, cnt in sorted(class_counts.items(), key=lambda x: int(x[0])):
            print(f"  클래스 {cls}: {cnt}개")
    else:
        print("  ⚠️ 라벨 파일이 없음")

print("\n📊 전체 데이터 합계:")
for cls, cnt in sorted(total_counts.items(), key=lambda x: int(x[0])):
    print(f"  클래스 {cls}: {cnt}개")

print("\n✅ 클래스별 객체 개수 계산 완료!")
