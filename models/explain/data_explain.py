from ultralytics import YOLO
import cv2
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from plyer import notification
import winsound
import threading
import time

def play_alarm_sound():
    """Play emergency alarm sound"""
    while True:
        winsound.Beep(1000, 500)  # 1000Hz for 500ms
        time.sleep(0.5)  # Pause between beeps
        winsound.Beep(800, 500)   # 800Hz for 500ms
        time.sleep(0.5)  # Pause between beeps

# 학습된 모델 불러오기
model = YOLO("runs/detect/train5/weights/best.pt")


# LIME을 적용할 함수 정의
def predict_for_lime(images):
    results = []
    for img in images:
        r = model.predict(img, verbose=False)
        if len(r[0].boxes) > 0:
            conf = r[0].boxes.conf[0].item()
        else:
            conf = 0.0
        # 충돌 가능성 / 비충돌 가능성
        results.append([conf, 1 - conf])
    return np.array(results)


# 테스트 이미지 불러오기
img_path = "./dataset/valid/images/Drone-Crash-Compilation-VOL-4_0_0-10_0_mp4-2_jpg.rf.183f13076b52ad2cb8170abb1533b72a.jpg"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 충돌 확률 임계값 설정
conf_threshold = 0.5

# 이미지별 충돌 확률 계산
conf = predict_for_lime([img])[0][0]

# Check collision risk and set alarm
if conf >= conf_threshold:
    color_map = "Reds"
    
    # Start alarm sound in a separate thread
    alarm_thread = threading.Thread(target=play_alarm_sound)
    alarm_thread.daemon = True  # Thread will stop when main program stops
    alarm_thread.start()
    
    # Show Windows notification
    notification.notify(
        title='Drone Collision Warning!',
        message=f'Collision Risk Detected!\nProbability: {conf*100:.1f}%\nTake immediate action!',
        app_icon=None,
        timeout=10,
    )
else:
    color_map = "Blues"

# LIME explainer
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(
    img, classifier_fn=predict_for_lime, top_labels=1, hide_color=0, num_samples=1000
)

# 충돌 기여도 가져오기
label = explanation.top_labels[0]
# 전체 슈퍼픽셀 개수로 시각화
num_features = len(explanation.local_exp[label])
temp, mask = explanation.get_image_and_mask(
    label=label,
    positive_only=False,  # 긍정/부정 모두
    num_features=num_features,  # 전체 슈퍼픽셀 시각화
    hide_rest=False,
)

# 각 슈퍼픽셀별 기여도 합산
positive_sum = sum(weight for _, weight in explanation.local_exp[label] if weight > 0)
negative_sum = sum(weight for _, weight in explanation.local_exp[label] if weight < 0)
total = positive_sum + abs(negative_sum)

# 통계 정보만 출력
print("\n=== 충돌 위험 분석 결과 ===")
print(f"충돌 확률: {conf*100:.1f}%")
print(f"충돌 위험 영역 기여도: {positive_sum / total * 100:.2f}%")
print(f"안전 영역 기여도: {abs(negative_sum) / total * 100:.2f}%")
print("==========================")

# 결과 시각화: 원본과 설명 시각화를 한 화면에 표시
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# 1. Original Image
axes[0].imshow(img)
axes[0].axis("off")
axes[0].set_title("Original Image")

# 2. LIME 설명 시각화
# LIME 설명자의 파라미터 조정
# 긍정적 기여도(초록색)
temp_pos, mask_pos = explanation.get_image_and_mask(
    label=explanation.top_labels[0],
    positive_only=True,  # 긍정적 기여만
    num_features=5,  # 상위 5개
    hide_rest=False,  # 전체 영역 표시
    min_weight=0.01,  # 최소 가중치 임계값을 낮게 설정
)

# 부정적 기여도(빨간색)
temp_neg, mask_neg = explanation.get_image_and_mask(
    label=explanation.top_labels[0],
    positive_only=False,  # 부정적 기여만
    negative_only=True,  # 부정적 기여만
    num_features=5,
    hide_rest=False,
    min_weight=-0.01,
)

# 마스크 합성하여 표시
base_img = img.copy() / 255.0

# 긍정적 영역 (초록색)
pos_overlay = base_img.copy()
pos_overlay[mask_pos] = [0, 1, 0]  # 초록색
axes[1].imshow(pos_overlay, alpha=0.3)  # 영역 투명도 감소

# 부정적 영역 (빨간색)
neg_overlay = base_img.copy()
neg_overlay[mask_neg] = [1, 0, 0]  # 빨간색
axes[1].imshow(neg_overlay, alpha=0.3)  # 영역 투명도 감소

# 경계선 표시 - 두꺼운 테두리와 높은 투명도
boundary_pos = mark_boundaries(
    base_img, mask_pos, color=(0, 1, 0), outline_color=(0, 1, 0), mode="thick"
)
boundary_neg = mark_boundaries(
    base_img, mask_neg, color=(1, 0, 0), outline_color=(1, 0, 0), mode="thick"
)

axes[1].imshow(boundary_pos, alpha=0.8)  # 초록색 경계 - 높은 투명도
axes[1].imshow(boundary_neg, alpha=0.8)  # 빨간색 경계 - 높은 투명도

axes[1].axis("off")
axes[1].set_title("LIME Visualization\n(Green: Collision Risk / Red: Safe Area)")

# Display guidance/contribution information as text at the top of the screen
fig.suptitle(
    f"{'[WARNING] Collision Risk! Probability' if conf >= conf_threshold else '[INFO] Low Collision Risk. Probability'}: {conf*100:.1f}%\n"
    f"Collision Risk Area Contribution: {positive_sum / total * 100:.2f}%\n"
    f"Safe Area Contribution: {abs(negative_sum) / total * 100:.2f}%",
    fontsize=18,
    color="red" if conf >= conf_threshold else "blue",
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
