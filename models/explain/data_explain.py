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

# 다양한 경로에서 모델 찾기
model_paths = [
    # 상대 경로들 (models/explain 기준)
    "../../data/runs/detect/visdrone_safe/weights/best.pt",
    "../../data/runs/detect/continue_training/weights/best.pt", 
    "../../data/runs/detect/train5/weights/best.pt",
    # 절대 경로들 (프로젝트 루트 기준)
    "data/runs/detect/visdrone_safe/weights/best.pt",
    "data/runs/detect/continue_training/weights/best.pt",
    "data/runs/detect/train5/weights/best.pt",
    # Windows 절대 경로
    "C:/Users/lab/softwareteam/data/runs/detect/visdrone_safe/weights/best.pt",
    "C:/Users/lab/softwareteam/data/runs/detect/continue_training/weights/best.pt",
    "C:/Users/lab/softwareteam/data/runs/detect/train5/weights/best.pt"
]

model = None
for i, path in enumerate(model_paths):
    try:
        model = YOLO(path)
        if i <= 2:
            print(f"모델 로드 완료: {['VisDrone 영상 학습', '향상', '기본'][i]} 모델")
        else:
            print(f"모델 로드 완료: {path}")
        break
    except Exception as e:
        continue

if model is None:
    print("❌ 모든 경로에서 모델을 찾을 수 없습니다!")
    exit(1)


# LIME을 적용할 함수 정의
def predict_for_lime(images):
    results = []
    for img in images:
        r = model.predict(img, verbose=False)
        if len(r[0].boxes) > 0:
            # 가장 높은 신뢰도를 가진 객체 선택
            conf = r[0].boxes.conf.max().item()
        else:
            conf = 0.0
        # 충돌 가능성 / 비충돌 가능성
        results.append([conf, 1 - conf])
    return np.array(results)

# 객체가 탐지된 영역만 마스킹하는 함수 추가
def get_object_mask(img, model):
    results = model.predict(img, verbose=False)
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
    
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            # 바운딩 박스 영역을 마스크에 추가
            mask[y1:y2, x1:x2] = True
    
    return mask

# 객체 영역에만 초점을 맞춘 LIME 예측 함수
def predict_for_lime_focused(images, object_mask):
    results = []
    for img in images:
        # 객체 영역 외부를 검은색으로 마스킹
        masked_img = img.copy()
        masked_img[~object_mask] = [0, 0, 0]  # 배경을 검은색으로
        
        r = model.predict(masked_img, verbose=False)
        if len(r[0].boxes) > 0:
            # 가장 높은 신뢰도를 가진 객체 선택
            conf = r[0].boxes.conf.max().item()
        else:
            conf = 0.0
        results.append([conf, 1 - conf])
    return np.array(results)


# 테스트 이미지 경로들 시도
image_paths = [
    "../../data/dataset/valid/images/Drone-Crash-Compilation-VOL-4_184_0-210_0_mp4-3_jpg.rf.dd8264825613b9a5986fcf2f69691044.jpg",
    "data/dataset/valid/images/Drone-Crash-Compilation-VOL-4_184_0-210_0_mp4-3_jpg.rf.dd8264825613b9a5986fcf2f69691044.jpg",
    "C:/Users/lab/softwareteam/data/dataset/valid/images/Drone-Crash-Compilation-VOL-4_184_0-210_0_mp4-3_jpg.rf.dd8264825613b9a5986fcf2f69691044.jpg"
]

img = None
for path in image_paths:
    try:
        img = cv2.imread(path)
        if img is not None:
            print(f"이미지 로드 완료: {path}")
            break
    except:
        continue

if img is None:
    print("❌ 테스트 이미지를 찾을 수 없습니다!")
    exit(1)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 충돌 확률 임계값 설정
conf_threshold = 0.5

# 이미지별 충돌 확률 계산
conf = predict_for_lime([img])[0][0]

# 객체 탐지 영역 마스크 생성
object_mask = get_object_mask(img, model)

# Check collision risk and set alarm
if conf >= conf_threshold:
    color_map = "Reds"
    
    # Start alarm sound in a separate thread
    alarm_thread = threading.Thread(target=play_alarm_sound)
    alarm_thread.daemon = True  # Thread will stop when main program stops
    alarm_thread.start()
    
    # 알림 시각화 (Windows)
    notification.notify(
        title='Drone Collision Warning!',
        message=f'Collision Risk Detected!\nProbability: {conf*100:.1f}%\nTake immediate action!',
        app_icon=None,
        timeout=10,
    )
else:
    color_map = "Blues"

# LIME explainer - 객체 영역에 집중
explainer = lime_image.LimeImageExplainer()

# 개선된 세그멘테이션 파라미터 사용
explanation = explainer.explain_instance(
    img, 
    classifier_fn=lambda images: predict_for_lime_focused(images, object_mask),
    top_labels=1, 
    hide_color=0, 
    num_samples=1000,
    segmentation_fn=lambda x: lime_image.SegmentationAlgorithm('quickshift', 
                                                              kernel_size=4,
                                                              max_dist=200, 
                                                              ratio=0.2)(x)
)

# 충돌 기여도 가져오기
label = explanation.top_labels[0]

# 신뢰도 기반 필터링 - 낮은 기여도 영역 제거
min_contribution_threshold = 0.05  # 5% 미만 기여도는 제거
filtered_exp = [(feature, weight) for feature, weight in explanation.local_exp[label] 
                if abs(weight) >= min_contribution_threshold]

print(f"원본 영역 수: {len(explanation.local_exp[label])}")
print(f"필터링 후 영역 수: {len(filtered_exp)}")

# 전체 슈퍼픽셀 개수로 시각화 (필터링된 것만)
if filtered_exp:
    # 필터링된 기여도만으로 마스크 생성
    temp, mask = explanation.get_image_and_mask(
        label=label,
        positive_only=False,
        num_features=len(filtered_exp),
        hide_rest=False,
    )
else:
    # 필터링된 영역이 없으면 기본값 사용
    temp, mask = explanation.get_image_and_mask(
        label=label,
        positive_only=False,
        num_features=5,
        hide_rest=False,
    )

# 각 슈퍼픽셀별 기여도 합산 (필터링된 것만)
if filtered_exp:
    positive_sum = sum(weight for _, weight in filtered_exp if weight > 0)
    negative_sum = sum(weight for _, weight in filtered_exp if weight < 0)
else:
    positive_sum = sum(weight for _, weight in explanation.local_exp[label] if weight > 0)
    negative_sum = sum(weight for _, weight in explanation.local_exp[label] if weight < 0)

total = positive_sum + abs(negative_sum)

# 통계 정보만 출력
print("\n=== 충돌 위험 분석 결과 ===")
print(f"충돌 확률: {conf*100:.1f}%")
if total > 0:
    print(f"충돌 위험 영역 기여도: {positive_sum / total * 100:.2f}%")
    print(f"안전 영역 기여도: {abs(negative_sum) / total * 100:.2f}%")
else:
    print("기여도 분석 불가 (탐지된 객체 없음)")
print("==========================================")

# 객체 탐지 결과도 표시
results = model.predict(img, verbose=False)
class_names = ['Building', 'Person', 'Pole', 'Tree', 'Vehicle']

if len(results[0].boxes) > 0:
    print("\n=== 탐지된 객체들 ===")
    max_conf = 0
    best_obj_info = ""
    
    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf_obj = box.conf[0].item()
        class_id = int(box.cls[0].item())
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
        
        print(f"객체 {i+1}: {class_name}, 좌표=({x1},{y1},{x2},{y2}), 신뢰도={conf_obj:.3f}")
        
        if conf_obj > max_conf:
            max_conf = conf_obj
            best_obj_info = f"{class_name} 신뢰도: {conf_obj:.3f}"
    
    print(f"\nLIME 분석 기준: {best_obj_info}")
else:
    print("탐지된 객체 없음")

# 결과 시각화: 원본, 객체 탐지, LIME 설명을 한 화면에 표시
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# 1. Original Image
axes[0].imshow(img)
axes[0].axis("off")
axes[0].set_title("Original Image")

# 2. Object Detection Result
img_with_boxes = img.copy()
results = model.predict(img, verbose=False)
if len(results[0].boxes) > 0:
    import matplotlib.patches as patches
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf_obj = box.conf[0].item()
        
        # 바운딩 박스 그리기
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=3, edgecolor='yellow', facecolor='none')
        axes[1].add_patch(rect)
        axes[1].text(x1, y1-10, f'Conf: {conf_obj:.2f}', 
                    bbox=dict(facecolor='yellow', alpha=0.8), fontsize=10)

axes[1].imshow(img_with_boxes)
axes[1].axis("off")
axes[1].set_title("YOLO Detection Results")

# 3. 개선된 LIME 설명 시각화
# 더 엄격한 기준으로 긍정적/부정적 기여도 분리
temp_pos, mask_pos = explanation.get_image_and_mask(
    label=explanation.top_labels[0],
    positive_only=True,
    num_features=3,  # 상위 3개만
    hide_rest=False,
    min_weight=0.05,  # 최소 가중치 임계값을 높게 설정
)

temp_neg, mask_neg = explanation.get_image_and_mask(
    label=explanation.top_labels[0],
    positive_only=False,
    negative_only=True,
    num_features=3,
    hide_rest=False,
    min_weight=-0.05,  # 최소 가중치 임계값을 높게 설정
)

# 객체 영역 마스크와 교집합만 표시
if len(results[0].boxes) > 0:
    # 객체 영역 내에서만 LIME 결과 표시
    mask_pos = mask_pos & object_mask
    mask_neg = mask_neg & object_mask

# 마스크 합성하여 표시
base_img = img.copy() / 255.0

# 긍정적 영역 (초록색) - 충돌 위험
pos_overlay = base_img.copy()
if np.any(mask_pos):
    pos_overlay[mask_pos] = [0, 1, 0]
    axes[2].imshow(pos_overlay, alpha=0.6)

# 부정적 영역 (빨간색) - 안전 영역
neg_overlay = base_img.copy()
if np.any(mask_neg):
    neg_overlay[mask_neg] = [1, 0, 0]
    axes[2].imshow(neg_overlay, alpha=0.6)

# 경계선 표시
if np.any(mask_pos):
    boundary_pos = mark_boundaries(base_img, mask_pos, color=(0, 1, 0), outline_color=(0, 1, 0), mode="thick")
    axes[2].imshow(boundary_pos, alpha=0.8)

if np.any(mask_neg):
    boundary_neg = mark_boundaries(base_img, mask_neg, color=(1, 0, 0), outline_color=(1, 0, 0), mode="thick")
    axes[2].imshow(boundary_neg, alpha=0.8)

axes[2].axis("off")
axes[2].set_title("Improved LIME Visualization\n(Green: Collision Risk / Red: Safe Area)\nFiltered & Object-Focused")

# Display guidance/contribution information as text at the top of the screen
fig.suptitle(
    f"{'[WARNING] Collision Risk! Probability' if conf >= conf_threshold else '[INFO] Low Collision Risk. Probability'}: {conf*100:.1f}%\n"
    f"Collision Risk Area Contribution: {positive_sum / total * 100:.2f}% | "
    f"Safe Area Contribution: {abs(negative_sum) / total * 100:.2f}%\n"
    f"Object-Focused Analysis | Filtered Low-Contribution Areas",
    fontsize=16,
    color="red" if conf >= conf_threshold else "blue",
)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()
