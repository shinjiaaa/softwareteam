import cv2
import pygame
from djitellopy import Tello
import numpy as np
from ultralytics import YOLO
from lime import lime_image
import threading
import time
import winsound
from queue import Queue, Empty
import os
import torch # GPU 확인을 위해 torch 임포트

# === Configuration (설정) ===
MODEL_PATHS = [
    # 상대 경로들 (models/explain 기준) 
    #"../../data/runs/detect/train5/weights/best.pt",
    # 절대 경로들 (프로젝트 루트 기준)
    "data/runs/detect/train5/weights/best.pt",
    "D:/SWPBL/softwareteam-KGH/data/runs/detect/train5/weights/best.pt"
]
CONF_THRESHOLD = 0.5

# --- 성능 최적화 설정 (Performance Optimization Settings) ---
# [최적화 1] LIME 샘플 수 감소 (기존 400 -> 100)
LIME_NUM_SAMPLES = 100
# [최적화 2] 분석 해상도 감소 (기존 640x480 -> 320x240)
ANALYSIS_WIDTH = 320
ANALYSIS_HEIGHT = 240
# [최적화 3] LIME 배치 처리 크기 (메모리 부족 시 감소, 예: 32 -> 16)
LIME_BATCH_SIZE = 32
# -----------------------------------------------------------

CONTROL_SPEED = 50
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 720
# =====================

# Global state
running = True
model = None
device = 'cpu' # 기본 연산 장치
alarm_event = threading.Event()

# Thread communication Queues
lime_input_queue = Queue(maxsize=1)
lime_output_queue = Queue(maxsize=1)

# === 1. Initialization (초기화) ===

def load_yolo_model(paths):
    """YOLO 모델을 로드하고 GPU 설정을 최적화합니다."""
    global device
    
    # [최적화 4] GPU(CUDA) 사용 가능 여부 확인
    if torch.cuda.is_available():
        device = 'cuda'
        print("✅ NVIDIA GPU(CUDA) 감지됨. GPU 가속을 사용합니다.")
    else:
        device = 'cpu'
        print("⚠️ GPU를 찾을 수 없습니다. CPU를 사용합니다. (성능이 느릴 수 있음)")

    for path in paths:
        if os.path.exists(path):
            try:
                m = YOLO(path)
                m.to(device) # 모델을 지정된 장치로 이동
                print(f"✅ 모델 로드 성공: {path}")
                
                # 모델 워밍업 (첫 실행 속도 향상)
                print("모델 워밍업 중...")
                m.predict(np.zeros((ANALYSIS_HEIGHT, ANALYSIS_WIDTH, 3), dtype=np.uint8), verbose=False, device=device)
                print("모델 워밍업 완료.")
                return m
            except Exception as e:
                print(f"❌ 모델 로드 실패 {path}: {e}")
    print("❌ 오류: YOLO 모델을 찾을 수 없습니다.")
    return None

# === 2. Utility Functions (유틸리티 함수) ===

def get_object_mask(img_rgb, model):
    """탐지된 객체 영역 마스킹"""
    # 예측 시 device 명시
    results = model.predict(img_rgb, verbose=False, imgsz=(ANALYSIS_HEIGHT, ANALYSIS_WIDTH), device=device)
    mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=bool)
    
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            # GPU 사용 시 .cpu() 호출 필요
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            mask[y1:y2, x1:x2] = True
    return mask

def alarm_worker():
    """알람 소리 재생 스레드"""
    while running:
        if alarm_event.wait(timeout=0.5):
            try:
                winsound.Beep(1000, 100)
                time.sleep(0.05)
            except Exception:
                time.sleep(0.1)

# === 3. LIME Analysis Worker Thread (LIME 분석 스레드 - 최적화됨) ===

def lime_analysis_worker():
    """백그라운드에서 LIME 분석을 수행 (배치 처리 및 벡터화 적용)"""
    explainer = lime_image.LimeImageExplainer()
    print("LIME 분석 워커 스레드 시작됨.")

    # [핵심 최적화] 배치 처리 및 벡터화된 마스킹을 지원하는 예측 함수
    def predict_for_lime_focused_vectorized(images, object_mask):
        """
        LIME 예측 함수. NumPy 벡터화를 통해 마스킹하고 배치로 추론합니다.
        """
        # LIME은 (N, H, W, C) 형태의 NumPy 배열을 전달합니다.
        
        # 1. 마스크 준비
        img_h, img_w = images[0].shape[:2]
        if object_mask.shape != (img_h, img_w):
             resized_mask = cv2.resize(object_mask.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            resized_mask = object_mask
            
        # 2. 벡터화된 마스킹 (Vectorized Masking)
        # NumPy 브로드캐스팅을 위해 마스크 형태 변형: (H, W) -> (1, H, W, 1)
        # 이 마스크는 (N, H, W, C) 형태의 배치 이미지 전체에 동시에 적용될 수 있습니다.
        mask_broadcastable = resized_mask[np.newaxis, :, :, np.newaxis]

        # 이미지 복사 후 마스킹 적용
        masked_images = images.copy()
        # ~mask_broadcastable이 True인 영역(배경)을 0으로 설정합니다. (데이터 타입 무관하게 효율적)
        masked_images[~mask_broadcastable] = 0

        # 3. 배치 추론 (Batch Inference)
        try:
            # YOLOv8 배치 추론 실행
            results_batch = model.predict(
                masked_images,
                verbose=False,
                imgsz=(ANALYSIS_HEIGHT, ANALYSIS_WIDTH),
                batch=LIME_BATCH_SIZE,
                device=device,
                stream=False # 모든 결과를 한 번에 받음
            )
        except Exception as e:
            print(f"❌ 배치 예측 중 오류 발생: {e}")
            return np.zeros((len(images), 2))

        # 4. 결과 처리
        final_results = []
        for r in results_batch:
            # 각 이미지의 최대 신뢰도를 충돌 확률로 사용 (GPU 사용 시 .cpu().item() 필요)
            conf = r.boxes.conf.max().cpu().item() if len(r.boxes) > 0 else 0.0
            # [충돌 확률, 비충돌 확률] 형태로 반환
            final_results.append([conf, 1 - conf])
            
        return np.array(final_results)

    # 분석 루프
    while running:
        try:
            img_rgb, object_mask = lime_input_queue.get(timeout=1)
            
            print(f"⏳ LIME 분석 중 (Samples: {LIME_NUM_SAMPLES}, Res: {ANALYSIS_WIDTH}x{ANALYSIS_HEIGHT}, Device: {device})...")
            start_time = time.time()
            
            # 분석 대상 프레임의 실제 충돌 확률 계산 (배치 함수 사용)
            # 입력 이미지를 배치 형태로 (1, H, W, C) 만들어 전달
            analyzed_conf = predict_for_lime_focused_vectorized(np.array([img_rgb]), object_mask)[0][0]

            # LIME 설명 생성 (최적화된 예측 함수 사용)
            explanation = explainer.explain_instance(
                img_rgb, 
                # classifier_fn에 최적화된 함수 전달
                classifier_fn=lambda images: predict_for_lime_focused_vectorized(images, object_mask),
                top_labels=1, 
                hide_color=0, 
                num_samples=LIME_NUM_SAMPLES,
                # 해상도가 낮아졌으므로 세그멘테이션 파라미터 조정
                segmentation_fn=lime_image.SegmentationAlgorithm('quickshift', 
                                                              kernel_size=3,
                                                              max_dist=100, 
                                                              ratio=0.2)
            )

            # 결과 추출 및 시각화 마스크 생성 (이하 동일)
            label = explanation.top_labels[0]
            
            # 기여도 계산
            positive_sum = sum(weight for _, weight in explanation.local_exp[label] if weight > 0)
            negative_sum = sum(weight for _, weight in explanation.local_exp[label] if weight < 0)
            total = positive_sum + abs(negative_sum)
            
            pos_contrib = (positive_sum / total * 100) if total > 0 else 0
            neg_contrib = (abs(negative_sum) / total * 100) if total > 0 else 0

            # 시각화 마스크 생성
            try:
                # 긍정적 기여도 (충돌 위험 영역)
                _, mask_pos = explanation.get_image_and_mask(
                    label=label, positive_only=True, num_features=3, hide_rest=False, min_weight=0.05
                )
                # 부정적 기여도 (안전 영역)
                _, mask_neg = explanation.get_image_and_mask(
                    label=label, positive_only=False, negative_only=True, num_features=3, hide_rest=False, min_weight=0.01
                )
            except (KeyError, IndexError):
                # 특징 추출 실패 시 빈 마스크 사용
                mask_pos = np.zeros_like(object_mask)
                mask_neg = np.zeros_like(object_mask)
            
            # 객체 마스크와 교집합 처리
            mask_pos = mask_pos & object_mask
            mask_neg = mask_neg & object_mask

            # 결과를 출력 큐에 전송
            if lime_output_queue.full():
                try: lime_output_queue.get_nowait()
                except Empty: pass
            lime_output_queue.put((mask_pos, mask_neg, analyzed_conf, pos_contrib, neg_contrib))

            end_time = time.time()
            print(f"✅ LIME 분석 완료 (소요 시간: {end_time - start_time:.2f}s)")

        except Empty:
            continue
        except Exception as e:
            print(f"❌ LIME 워커 오류: {e}")
            time.sleep(1)

# === 4. Drone Control (드론 제어) ===
# (이전 코드와 동일)

def handle_keyboard_control(tello):
    """Handles Pygame keyboard events for responsive, speed-based control."""
    lr, fb, ud, yv = 0, 0, 0, 0
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            global running
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_t:
                print("이륙 시도...")
                try: tello.takeoff()
                except Exception as e: print(f"이륙 실패: {e}")
            elif event.key == pygame.K_l:
                print("착륙 시도...")
                try: tello.land()
                except Exception as e: print(f"착륙 실패: {e}")

    keys = pygame.key.get_pressed()
    if keys[pygame.K_a]: lr = -CONTROL_SPEED
    elif keys[pygame.K_d]: lr = CONTROL_SPEED
    
    if keys[pygame.K_w]: fb = CONTROL_SPEED
    elif keys[pygame.K_s]: fb = -CONTROL_SPEED
    
    if keys[pygame.K_UP]: ud = CONTROL_SPEED
    elif keys[pygame.K_DOWN]: ud = -CONTROL_SPEED
    
    if keys[pygame.K_LEFT]: yv = -CONTROL_SPEED
    elif keys[pygame.K_RIGHT]: yv = CONTROL_SPEED
        
    try:
        if tello.is_flying:
            tello.send_rc_control(lr, fb, ud, yv)
    except Exception as e:
        # print(f"Failed to send RC control: {e}")
        pass
        
# === 5. Visualization (시각화) ===
def visualize(display_frame, results, max_conf, lime_result):
    """Draws the HUD, YOLO detections, and LIME explanations."""
    
    # 1. LIME Overlay
    if lime_result:
        mask_pos, mask_neg, analyzed_conf, pos_contrib, neg_contrib = lime_result
        
        # Resize masks from ANALYSIS_SIZE to DISPLAY_SIZE
        mask_pos_resized = cv2.resize(mask_pos.astype(np.uint8), (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST)
        mask_neg_resized = cv2.resize(mask_neg.astype(np.uint8), (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST)

        overlay = display_frame.copy()
        
        # Positive contribution (Red - Risk Factor)
        overlay[mask_pos_resized > 0] = [0, 0, 255] # BGR format
        
        # Negative contribution (Green - Safety Factor)
        overlay[mask_neg_resized > 0] = [0, 255, 0]
        
        # Blend overlay with the image (40% opacity)
        cv2.addWeighted(overlay, 0.4, display_frame, 0.6, 0, display_frame)

        # LIME Info Text
        cv2.putText(display_frame, "LIME Explanation (Optimized)", 
                    (10, DISPLAY_HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        contrib_text = f"Risk Contrib: {pos_contrib:.1f}%, Safety Contrib: {neg_contrib:.1f}% (Analyzed @ {analyzed_conf*100:.1f}%)"
        cv2.putText(display_frame, contrib_text, 
                    (10, DISPLAY_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 2. YOLO Bounding Boxes
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            # Coordinates need scaling (GPU 사용 시 .cpu() 필요)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1 = int(x1 * DISPLAY_WIDTH / ANALYSIS_WIDTH)
            x2 = int(x2 * DISPLAY_WIDTH / ANALYSIS_WIDTH)
            y1 = int(y1 * DISPLAY_HEIGHT / ANALYSIS_HEIGHT)
            y2 = int(y2 * DISPLAY_HEIGHT / ANALYSIS_HEIGHT)

            conf = box.conf[0].cpu().item()
            color = (0, 0, 255) if conf >= CONF_THRESHOLD else (0, 255, 255) # Red or Yellow
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, f"{conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # 3. HUD (Status Text)
    is_high_risk = max_conf >= CONF_THRESHOLD
    status_color = (0, 0, 255) if is_high_risk else (0, 255, 0)
    status_text = f"Real-Time Collision Risk: {max_conf*100:.1f}% (Device: {device})"
    
    if is_high_risk:
            cv2.putText(display_frame, "WARNING! " + status_text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
    else:
        cv2.putText(display_frame, status_text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
        
    return display_frame

# === 6. Main Application Loop (메인 애플리케이션 루프) ===

def main():
    global model, running
    
    # Initialize Model (GPU 설정 포함)
    model = load_yolo_model(MODEL_PATHS)
    if not model: return
    
    # Initialize Pygame and Tello
    pygame.init()
    screen = pygame.display.set_mode((300, 100))
    pygame.display.set_caption("Tello Control (Focus Here)")

    tello = Tello()
    try:
        tello.connect()
        print(f"✅ Tello 연결됨. 배터리: {tello.get_battery()}%")
        tello.streamon()
        frame_read = tello.get_frame_read()
    except Exception as e:
        print(f"❌ Tello 연결 실패: {e}. Wi-Fi 연결을 확인하세요.")
        pygame.quit()
        return

    # Start worker threads
    alarm_thread = threading.Thread(target=alarm_worker)
    alarm_thread.start()
    lime_thread = threading.Thread(target=lime_analysis_worker)
    lime_thread.start()

    print("🚀 시스템 준비 완료. 제어를 위해 Pygame 창을 활성화하세요.")
    
    current_lime_result = None

    while running:
        # 1. Control
        handle_keyboard_control(tello)

        # 2. Get Frame
        frame_bgr = frame_read.frame
        if frame_bgr is None:
            time.sleep(0.01)
            continue
            
        # Prepare frames: Display (High resolution) and Analysis (Low resolution)
        display_frame = cv2.resize(frame_bgr, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        analysis_frame_bgr = cv2.resize(frame_bgr, (ANALYSIS_WIDTH, ANALYSIS_HEIGHT))
        
        # Convert BGR to RGB for model processing
        analysis_frame_rgb = cv2.cvtColor(analysis_frame_bgr, cv2.COLOR_BGR2RGB)

        # 3. Real-time YOLO Prediction (Fast)
        # 메인 루프 예측 시에도 device 명시
        results = model.predict(analysis_frame_rgb, verbose=False, imgsz=(ANALYSIS_HEIGHT, ANALYSIS_WIDTH), device=device)
        
        max_conf = 0.0
        if len(results[0].boxes) > 0:
            # GPU 사용 시 .cpu().item() 필요
            max_conf = results[0].boxes.conf.max().cpu().item()

        # 4. Risk Assessment and Alarm
        is_high_risk = max_conf >= CONF_THRESHOLD
        alarm_event.set() if is_high_risk else alarm_event.clear()

        # 5. LIME Analysis Request (Non-blocking)
        if lime_input_queue.empty():
            # Generate object mask
            object_mask = get_object_mask(analysis_frame_rgb, model)
            # Optimization: Only run LIME if there's a significant object detected (임계값 0.3)
            if np.any(object_mask) and max_conf > 0.3:
                lime_input_queue.put((analysis_frame_rgb.copy(), object_mask))

        # 6. Update LIME Results (Non-blocking check)
        try:
            current_lime_result = lime_output_queue.get_nowait() 
        except Empty:
            pass

        # 7. Visualization
        display_frame = visualize(display_frame, results, max_conf, current_lime_result)

        # 8. Display Frame
        cv2.imshow("Tello Collision Detection & LIME Explanation (Optimized)", display_frame)

        if cv2.waitKey(1) & 0xFF == 27: # ESC 키
            running = False

    # === 7. Cleanup (종료 및 정리) ===
    print("🛑 시스템 종료 중...")
    running = False
    alarm_event.set()
    
    if alarm_thread.is_alive():
        alarm_thread.join(timeout=2)
    if lime_thread.is_alive():
        # LIME 스레드가 배치 처리 중일 수 있으므로 넉넉히 대기
        lime_thread.join(timeout=15)

    try:
        if tello.is_flying:
            print("드론 착륙 중...")
            tello.land()
        tello.streamoff()
        tello.end()
    except Exception as e:
        print(f"Tello 정리 중 오류 발생: {e}")

    cv2.destroyAllWindows()
    pygame.quit()
    print("✅ 종료 완료.")

if __name__ == '__main__':
    main()
