from datasets import Dataset
import os
import cv2

# 🔹 1. 데이터셋 경로
base_dir = "C:/Users/lab/.cache/huggingface/datasets/nexar-ai___nexar_collision_prediction"

# 🔹 2. arrow 파일 경로 (네 PC에 실제로 있는 파일명 확인 필요)
train_arrow = os.path.join(base_dir, "nexar_collision_prediction-train.arrow")

if not os.path.exists(train_arrow):
    raise FileNotFoundError(f"❌ dataset.arrow 파일을 찾을 수 없습니다: {train_arrow}")

train_dataset = Dataset.from_file(train_arrow)
print(f"✅ Dataset loaded successfully with {len(train_dataset)} samples")

# 🔹 3. 첫 번째 샘플 확인
sample = train_dataset[0]
print("Sample keys:", sample.keys())

# 🔹 4. video_path 확인
video_path = sample.get("video_path", None)
if not video_path:
    raise KeyError("❌ video_path 키를 찾을 수 없습니다.")

# 🔹 5. 실제 비디오 파일 경로
video_full_path = os.path.join(base_dir, "data", video_path)
if not os.path.exists(video_full_path):
    raise FileNotFoundError(f"❌ 비디오 파일이 존재하지 않습니다: {video_full_path}")

print(f"🎥 Video path: {video_full_path}")

# 🔹 6. OpenCV로 프레임 추출
cap = cv2.VideoCapture(video_full_path)
if not cap.isOpened():
    raise RuntimeError(f"❌ 비디오 파일을 열 수 없습니다: {video_full_path}")

os.makedirs("frames", exist_ok=True)
i = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(f"frames/frame_{i:04d}.jpg", frame)
    i += 1

cap.release()
print(f"✅ {i} frames extracted successfully from {video_full_path}")
