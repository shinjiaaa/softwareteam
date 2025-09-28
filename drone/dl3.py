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
import torch
import traceback

# === Configuration (설정) ===
MODEL_PATHS = [
    # 학습된 커스텀 모델 (안정화된 버전)
    "models/weights/yolov8n_custom.pt",
    # 기본 사전학습 모델 (자동 다운로드)
    "yolov8n.pt"
]
CONF_THRESHOLD = 0.5

# --- 성능 및 안정성 설정 (Performance and Stability Settings) ---
LIME_NUM_SAMPLES = 100

# [중요 수정 1] 분석 해상도 설정. YOLO 모델은 높이/너비가 32의 배수여야 합니다.
# 기존 240은 32의 배수가 아니어서 오류 발생 가능성 -> 256으로 수정.
ANALYSIS_WIDTH = 320  # (32 * 10)
ANALYSIS_HEIGHT = 256 # (32 * 8)

LIME_BATCH_SIZE = 16 # 메모리 부족 시 8로 감소
LIME_MIN_CONFIDENCE = 0.35
# -----------------------------------------------------------

CONTROL_SPEED = 50
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 720
# =====================

# Global state
running = True
model = None
device = 'cpu'
alarm_event = threading.Event()

# Thread communication Queues
lime_input_queue = Queue(maxsize=1)
lime_output_queue = Queue(maxsize=1)

# === 1. Initialization (초기화) ===

def load_yolo_model(paths):
    """YOLO 모델을 로드하고 GPU 설정을 최적화합니다."""
    global device
    
    if torch.cuda.is_available():
        device = 'cuda'
        print("✅ NVIDIA GPU(CUDA) 감지됨. GPU 가속을 사용합니다.")
    else:
        device = 'cpu'
        print(f"⚠️ GPU를 찾을 수 없습니다. CPU를 사용합니다. (Python 버전: {os.sys.version.split()[0]})")

    for path in paths:
        if os.path.exists(path):
            try:
                m = YOLO(path)
                m.to(device)
                print(f"✅ 모델 로드 성공: {path}")
                
                # 모델 워밍업
                print("모델 워밍업 중...")
                try:
                    # 수정된 해상도로 워밍업 (imgsz 명시하여 경고 방지)
                    m.predict(np.zeros((ANALYSIS_HEIGHT, ANALYSIS_WIDTH, 3), dtype=np.uint8), verbose=False, device=device, imgsz=(ANALYSIS_HEIGHT, ANALYSIS_WIDTH))
                    print("모델 워밍업 완료.")
                except Exception as e:
                    print(f"⚠️ 모델 워밍업 중 경고 발생: {e}")
                return m
            except Exception as e:
                print(f"❌ 모델 로드 실패 {path}: {e}")
    print("❌ 오류: YOLO 모델을 찾을 수 없습니다.")
    return None

# === 2. Utility Functions (유틸리티 함수) ===

def get_object_mask(img_rgb, model):
    """탐지된 객체 영역 마스킹"""
    results = model.predict(img_rgb, verbose=False, imgsz=(ANALYSIS_HEIGHT, ANALYSIS_WIDTH), device=device)
    mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=bool)
    
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            mask[y1:y2, x1:x2] = True
    return mask

def alarm_worker():
    """알람 소리 재생 스레드"""
    while running:
        if alarm_event.wait(timeout=0.5):
            try:
                winsound.Beep(1000, 80)
                time.sleep(0.05)
            except Exception:
                time.sleep(0.1)

# === 3. LIME Analysis Worker Thread (LIME 분석 스레드 - 종합 안정화 적용) ===

def lime_analysis_worker():
    """백그라운드에서 LIME 분석을 수행 (데이터 타입 표준화, 빈 이미지 필터링 적용)"""
    explainer = lime_image.LimeImageExplainer()
    print("LIME 분석 워커 스레드 시작됨.")

    # [종합 수정] 안정화 및 최적화된 예측 함수
    def predict_for_lime_focused_robust(images, object_mask):
        """
        LIME 예측 함수. 데이터 타입 표준화, 마스킹, 빈 이미지 필터링, 배치 추론을 수행합니다.
        """
        try:
            # 1. [중요 수정 2] 데이터 타입 표준화 (Data Type Standardization)
            # LIME이 float 타입을 제공할 경우 uint8(0-255)로 변환하여 OpenCV 오류 방지
            if images.dtype in [np.float64, np.float32, np.float16]:
                # 픽셀 값이 0.0-1.0 범위인지 확인 후 변환
                if images.max() <= 1.0 + 1e-6: 
                     standardized_images = (images * 255).astype(np.uint8)
                else:
                    standardized_images = images.astype(np.uint8)
            else:
                standardized_images = images.astype(np.uint8)

            # 2. 마스크 준비
            img_h, img_w = standardized_images[0].shape[:2]
            if object_mask.shape != (img_h, img_w):
                 resized_mask = cv2.resize(object_mask.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                resized_mask = object_mask
                
            # 3. 벡터화된 마스킹 (Vectorized Masking - 곱셈 방식)
            mask_broadcastable = resized_mask[np.newaxis, :, :, np.newaxis]
            masked_images = (standardized_images * mask_broadcastable.astype(standardized_images.dtype))

            # 4. [중요 수정 3] 빈 이미지 필터링 (Filtering empty images)
            # 완전히 검은색 이미지 식별 (NumPy 벡터 연산 사용)
            image_sums = np.sum(masked_images, axis=(1, 2, 3))
            is_empty = (image_sums == 0)
            
            # 결과 배열 초기화 (기본값: 충돌 확률 0%)
            final_results = np.zeros((len(images), 2), dtype=np.float32)
            final_results[:, 1] = 1.0 # 비충돌 확률 100%

            # 빈 이미지가 아닌 것들만 필터링
            non_empty_indices = np.where(~is_empty)[0]
            images_to_process = masked_images[non_empty_indices]

            # 5. 배치 추론 (Batch Inference)
            if len(images_to_process) > 0:
                
                # 참고: Python 3.13 호환성을 위해 리스트로 변환하는 것이 더 안전할 수 있으나, 
                # NumPy 배열로 유지하여 성능을 우선합니다. 문제 지속 시 list(images_to_process)로 변경 고려.
                
                results_batch = model.predict(
                    images_to_process,
                    verbose=False,
                    imgsz=(ANALYSIS_HEIGHT, ANALYSIS_WIDTH), # 수정된 해상도 사용
                    batch=LIME_BATCH_SIZE,
                    device=device,
                    stream=False
                )

                # 6. 결과 병합
                batch_confidences = []
                for r in results_batch:
                    conf = r.boxes.conf.max().cpu().item() if len(r.boxes) > 0 else 0.0
                    batch_confidences.append([conf, 1.0 - conf])
                
                # 처리된 결과를 올바른 위치에 삽입
                final_results[non_empty_indices] = np.array(batch_confidences)
                
            return final_results

        except Exception as e:
            # 예외 발생 시 상세 로그 출력
            print("\n" + "="*50)
            print(f"❌ 배치 예측 중 심각한 오류 발생: {e}")
            print(f"Device: {device}, Batch Size: {LIME_BATCH_SIZE}, Input LIME dtype: {images.dtype}")
            print("--- 상세 오류 내용 (Traceback) ---")
            traceback.print_exc()
            print("="*50 + "\n")
            return np.zeros((len(images), 2))

    # 분석 루프
    while running:
        try:
            img_rgb, object_mask = lime_input_queue.get(timeout=1)
            
            print(f"⏳ LIME 분석 중 (Samples: {LIME_NUM_SAMPLES}, Res: {ANALYSIS_WIDTH}x{ANALYSIS_HEIGHT}, Device: {device})...")
            start_time = time.time()
            
            # 분석 대상 프레임의 실제 충돌 확률 계산
            analyzed_conf = predict_for_lime_focused_robust(np.array([img_rgb]), object_mask)[0][0]

            # LIME 설명 생성
            explanation = explainer.explain_instance(
                img_rgb, 
                classifier_fn=lambda images: predict_for_lime_focused_robust(images, object_mask),
                top_labels=1, 
                hide_color=0, 
                num_samples=LIME_NUM_SAMPLES,
                segmentation_fn=lime_image.SegmentationAlgorithm('quickshift', 
                                                              kernel_size=3,
                                                              max_dist=100, 
                                                              ratio=0.2)
            )

            # 결과 추출 및 시각화 마스크 생성
            try:
                label = explanation.top_labels[0]
                
                # 기여도 계산
                positive_sum = sum(weight for _, weight in explanation.local_exp[label] if weight > 0)
                negative_sum = sum(weight for _, weight in explanation.local_exp[label] if weight < 0)
                total = positive_sum + abs(negative_sum)
                
                pos_contrib = (positive_sum / total * 100) if total > 0 else 0
                neg_contrib = (abs(negative_sum) / total * 100) if total > 0 else 0

                # 긍정적 기여도 (충돌 위험 영역)
                _, mask_pos = explanation.get_image_and_mask(
                    label=label, positive_only=True, num_features=3, hide_rest=False, min_weight=0.05
                )
                # 부정적 기여도 (안전 영역)
                _, mask_neg = explanation.get_image_and_mask(
                    label=label, positive_only=False, negative_only=True, num_features=3, hide_rest=False, min_weight=0.01
                )
                
            except Exception as e:
                print(f"⚠️ LIME 결과 처리 중 경고: {e}")
                mask_pos = np.zeros_like(object_mask)
                mask_neg = np.zeros_like(object_mask)
                pos_contrib = 0
                neg_contrib = 0

            
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
            # LIME 워커 루프 전체의 예외 처리
            print("\n" + "="*50)
            print(f"❌ LIME 워커 오류 발생: {e}")
            print("--- 상세 오류 내용 (Traceback) ---")
            traceback.print_exc()
            print("="*50 + "\n")
            time.sleep(2)

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
        if np.any(mask_pos_resized > 0):
            overlay[mask_pos_resized > 0] = [0, 0, 255] # BGR format
        
        # Negative contribution (Green - Safety Factor)
        if np.any(mask_neg_resized > 0):
            overlay[mask_neg_resized > 0] = [0, 255, 0]
        
        # Blend overlay with the image (40% opacity)
        cv2.addWeighted(overlay, 0.4, display_frame, 0.6, 0, display_frame)

        # LIME Info Text
        cv2.putText(display_frame, "LIME Explanation (Final)", 
                    (10, DISPLAY_HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        contrib_text = f"Risk Contrib: {pos_contrib:.1f}%, Safety Contrib: {neg_contrib:.1f}% (Analyzed @ {analyzed_conf*100:.1f}%)"
        cv2.putText(display_frame, contrib_text, 
                    (10, DISPLAY_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 2. YOLO Bounding Boxes
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            # Coordinates scaling
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1 = int(x1 * DISPLAY_WIDTH / ANALYSIS_WIDTH)
            x2 = int(x2 * DISPLAY_WIDTH / ANALYSIS_WIDTH)
            y1 = int(y1 * DISPLAY_HEIGHT / ANALYSIS_HEIGHT)
            y2 = int(y2 * DISPLAY_HEIGHT / ANALYSIS_HEIGHT)

            conf = box.conf[0].cpu().item()
            color = (0, 0, 255) if conf >= CONF_THRESHOLD else (0, 255, 255)
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
    
    # Initialize Model
    model = load_yolo_model(MODEL_PATHS)
    if not model: return
    
    # Initialize Pygame and Tello
    pygame.init()
    screen = pygame.display.set_mode((300, 100))
    pygame.display.set_caption("Tello Control (Focus Here)")

    tello = Tello()
    try:
        tello.connect()
        try:
            battery = tello.get_battery()
            print(f"✅ Tello 연결됨. 배터리: {battery}%")
        except Exception as e:
            print(f"✅ Tello 연결됨. 배터리 확인 실패: {e}")
            
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
        try:
            # 1. Control
            handle_keyboard_control(tello)

            # 2. Get Frame
            frame_bgr = frame_read.frame
            if frame_bgr is None:
                time.sleep(0.01)
                continue
                
            # Prepare frames
            display_frame = cv2.resize(frame_bgr, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            # 수정된 해상도로 분석 프레임 생성
            analysis_frame_bgr = cv2.resize(frame_bgr, (ANALYSIS_WIDTH, ANALYSIS_HEIGHT))
            # 분석용 프레임은 항상 uint8 타입 보장
            analysis_frame_rgb = cv2.cvtColor(analysis_frame_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)

            # 3. Real-time YOLO Prediction
            results = model.predict(analysis_frame_rgb, verbose=False, imgsz=(ANALYSIS_HEIGHT, ANALYSIS_WIDTH), device=device)
            
            max_conf = 0.0
            if len(results[0].boxes) > 0:
                max_conf = results[0].boxes.conf.max().cpu().item()

            # 4. Risk Assessment and Alarm
            is_high_risk = max_conf >= CONF_THRESHOLD
            alarm_event.set() if is_high_risk else alarm_event.clear()

            # 5. LIME Analysis Request
            if lime_input_queue.empty():
                if max_conf >= LIME_MIN_CONFIDENCE:
                    object_mask = get_object_mask(analysis_frame_rgb, model)
                    if np.any(object_mask):
                        lime_input_queue.put((analysis_frame_rgb.copy(), object_mask))

            # 6. Update LIME Results
            try:
                current_lime_result = lime_output_queue.get_nowait() 
            except Empty:
                pass

            # 7. Visualization
            display_frame = visualize(display_frame, results, max_conf, current_lime_result)

            # 8. Display Frame
            cv2.imshow("Tello Collision Detection & LIME Explanation (Final)", display_frame)

            if cv2.waitKey(1) & 0xFF == 27: # ESC 키
                running = False
        
        except Exception as e:
            print(f"❌ 메인 루프에서 심각한 오류 발생: {e}")
            traceback.print_exc()
            running = False

    # === 7. Cleanup (종료 및 정리) ===
    print("🛑 시스템 종료 중...")
    running = False
    alarm_event.set()
    
    if alarm_thread.is_alive():
        alarm_thread.join(timeout=2)
    if lime_thread.is_alive():
        lime_thread.join(timeout=20) # 배치 처리 시간을 고려하여 넉넉히 대기

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