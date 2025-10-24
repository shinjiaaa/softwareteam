# 필요한 라이브러리 임포트
import cv2                      # 영상 처리 및 시각화 (OpenCV)
import pygame                   # 키보드 입력을 통한 드론 제어 인터페이스
from djitellopy import Tello    # Tello 드론 제어 SDK
import numpy as np              # 수치 연산
from ultralytics import YOLO    # 객체 탐지 모델 (YOLOv8)
from lime import lime_image     # 설명 가능한 AI (LIME) 라이브러리
import threading                # 병렬 처리를 위한 스레딩 (알람, LIME 분석)
import time                     # 시간 관련 함수
import winsound                 # Windows 환경에서의 알람 소리 재생
from queue import Queue, Empty  # 스레드 간 안전한 데이터 교환을 위한 큐
import os                       # 파일 시스템 경로 확인

# === Configuration (설정) ===
# 모델 경로 설정: 학습된 커스텀 모델 또는 기본 모델 사용
MODEL_PATHS = [
    # 커스텀 모델 (우선 사용)
    "models/weights/yolov8n_custom.pt",
    # 기본 사전학습 모델 (자동 다운로드)
    "yolov8n.pt"
]
CONF_THRESHOLD = 0.5        # 충돌 위험으로 판단할 신뢰도 임계값 (50%)
LIME_NUM_SAMPLES = 400      # LIME 분석 샘플 수 (값이 클수록 정확하지만 느려짐. 실시간성을 위해 400으로 설정)
CONTROL_SPEED = 50          # 드론 제어 속도 (0~100)
DISPLAY_WIDTH = 960         # 화면에 표시될 영상의 가로 크기
DISPLAY_HEIGHT = 720        # 화면에 표시될 영상의 세로 크기
ANALYSIS_WIDTH = 640        # YOLO 및 LIME 분석에 사용될 내부 이미지 크기 (작을수록 속도 향상)
ANALYSIS_HEIGHT = 480
# =====================

# Global state (전역 상태 변수)
running = True # 프로그램 실행 상태를 제어하는 플래그
model = None   # YOLO 모델 객체를 저장할 변수
# 알람 스레드 제어를 위한 이벤트 객체 (효율적인 신호 전달)
alarm_event = threading.Event()

# 스레드 간 통신을 위한 큐 (Queue)
# maxsize=1로 설정하여 LIME이 항상 가장 최신 프레임만 분석하도록 보장

# LIME 입력 큐: 메인 스레드 -> LIME 스레드 (분석용 RGB 프레임, 객체 마스크)
lime_input_queue = Queue(maxsize=1)
# LIME 출력 큐: LIME 스레드 -> 메인 스레드 (긍정 마스크, 부정 마스크, 분석된 신뢰도, 기여도 등)
lime_output_queue = Queue(maxsize=1)

# === 1. Initialization (초기화) ===

def load_yolo_model(paths):
    """여러 경로에서 YOLO 모델을 로드하는 함수"""
    for path in paths:
        if os.path.exists(path): # 파일 존재 여부 확인
            try:
                m = YOLO(path)
                print(f"✅ 모델 로드 성공: {path}")
                return m
            except Exception as e:
                print(f"❌ 모델 로드 실패 {path}: {e}")
    print("❌ 오류: YOLO 모델을 찾을 수 없습니다. MODEL_PATHS 설정을 확인하세요.")
    return None

# === 2. Utility Functions (유틸리티 함수) ===

def get_object_mask(img_rgb, model):
    """YOLO를 사용하여 이미지에서 탐지된 객체 영역만 마스킹하는 함수 (LIME 집중 분석용)"""
    # 모델 예측 수행 (verbose=False로 로그 출력 억제)
    results = model.predict(img_rgb, verbose=False, imgsz=(ANALYSIS_HEIGHT, ANALYSIS_WIDTH))
    # 이미지와 동일한 크기의 빈 마스크 생성 (False로 채워짐)
    mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=bool)
    
    # 탐지된 객체가 있다면
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            # 바운딩 박스 좌표 추출 및 정수 변환
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            # 해당 영역을 마스크에서 True로 설정
            mask[y1:y2, x1:x2] = True
    return mask

def alarm_worker():
    """알람 소리를 재생하는 백그라운드 스레드 함수"""
    while running:
        # alarm_event가 설정될 때까지 효율적으로 대기 (timeout은 'running' 상태 확인용)
        if alarm_event.wait(timeout=0.5):
            try:
                # 충돌 위험 시 경고음 재생 (Windows용)
                winsound.Beep(1000, 100)
                time.sleep(0.05)
                winsound.Beep(800, 100)
                time.sleep(0.1)
            except Exception:
                # winsound 실패 시 대기
                time.sleep(0.2)

# === 3. LIME Analysis Worker Thread (LIME 분석 스레드 - 핵심 로직) ===
# 이 부분은 연산량이 많아 메인 스레드와 분리하여 비동기로 실행됩니다.

def lime_analysis_worker():
    """백그라운드에서 LIME 분석을 수행하는 함수"""
    explainer = lime_image.LimeImageExplainer() # LIME 설명자 초기화
    print("LIME 분석 워커 스레드 시작됨.")

    # LIME 예측 함수 정의 (클로저 형태)
    def predict_for_lime_focused(images, object_mask):
        """
        LIME이 모델의 예측을 호출할 때 사용하는 함수.
        객체 영역에만 집중하여 분석하도록 마스킹을 적용합니다.
        """
        results = []
        for img in images: # LIME은 여러 개의 변형된 이미지를 입력으로 제공
            # img는 RGB 형식
            masked_img = img.copy()
            
            # 중요: LIME이 제공하는 이미지 크기에 맞춰 마스크 크기를 조정해야 함
            img_h, img_w = img.shape[:2]
            if object_mask.shape != (img_h, img_w):
                 # 보간법(INTER_NEAREST)을 사용하여 마스크 크기 조정
                 resized_mask = cv2.resize(object_mask.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                resized_mask = object_mask

            # 배경(객체 외 영역)을 검은색으로 마스킹
            # 이미지 타입(float 또는 int)에 따라 처리
            if masked_img.dtype in [np.float64, np.float32]:
                 masked_img[~resized_mask] = 0.0
            else:
                 masked_img[~resized_mask] = 0
            
            # 마스킹된 이미지로 YOLO 예측 수행
            r = model.predict(masked_img, verbose=False, imgsz=(ANALYSIS_HEIGHT, ANALYSIS_WIDTH))
            
            # 가장 높은 신뢰도를 충돌 확률로 사용
            conf = r[0].boxes.conf.max().item() if len(r[0].boxes) > 0 else 0.0
            # 결과는 [충돌 확률, 비충돌 확률] 형태로 반환해야 함
            results.append([conf, 1 - conf])
        return np.array(results)

    # 분석 루프
    while running:
        try:
            # 입력 큐에서 다음 분석할 프레임과 마스크를 가져옴 (최대 1초 대기)
            img_rgb, object_mask = lime_input_queue.get(timeout=1)
            
            print("⏳ LIME 프레임 분석 중...")
            start_time = time.time()
            
            # 분석 대상 프레임의 실제 충돌 확률 계산
            analyzed_conf = predict_for_lime_focused([img_rgb], object_mask)[0][0]

            # LIME 설명 생성 (핵심 연산)
            explanation = explainer.explain_instance(
                img_rgb, 
                classifier_fn=lambda images: predict_for_lime_focused(images, object_mask), # 예측 함수 지정
                top_labels=1, 
                hide_color=0, 
                num_samples=LIME_NUM_SAMPLES, # 샘플 수 설정
                # 세그멘테이션 알고리즘 설정 (이미지를 슈퍼픽셀로 나누는 방법)
                segmentation_fn=lime_image.SegmentationAlgorithm('quickshift', 
                                                              kernel_size=4,
                                                              max_dist=200, 
                                                              ratio=0.2)
            )

            # 결과 추출 및 시각화 마스크 생성
            label = explanation.top_labels[0]
            
            # 기여도 계산 (각 슈퍼픽셀의 가중치 합산)
            positive_sum = sum(weight for _, weight in explanation.local_exp[label] if weight > 0)
            negative_sum = sum(weight for _, weight in explanation.local_exp[label] if weight < 0)
            total = positive_sum + abs(negative_sum)
            
            # 백분율로 변환
            pos_contrib = (positive_sum / total * 100) if total > 0 else 0
            neg_contrib = (abs(negative_sum) / total * 100) if total > 0 else 0

            # 시각화 마스크 생성
            try:
                # 긍정적 기여도 (충돌 위험 영역) - 상위 3개 특징, 최소 가중치 0.05 이상
                _, mask_pos = explanation.get_image_and_mask(
                    label=label, positive_only=True, num_features=3, hide_rest=False, min_weight=0.05
                )
                # 부정적 기여도 (안전 영역) - 상위 3개 특징, 최소 가중치 0.01 이상
                _, mask_neg = explanation.get_image_and_mask(
                    label=label, positive_only=False, negative_only=True, num_features=3, hide_rest=False, min_weight=0.01
                )
            except KeyError:
                # LIME이 유의미한 특징을 찾지 못했을 경우 빈 마스크 사용
                mask_pos = np.zeros_like(object_mask)
                mask_neg = np.zeros_like(object_mask)

            
            # 객체 마스크와 교집합 처리 (객체 영역 내에서만 설명 표시)
            mask_pos = mask_pos & object_mask
            mask_neg = mask_neg & object_mask

            # 결과를 출력 큐에 전송 (메인 스레드에서 사용)
            # 큐가 가득 차 있으면 이전 결과를 버리고 최신 결과로 갱신
            if lime_output_queue.full():
                try: lime_output_queue.get_nowait()
                except Empty: pass
            lime_output_queue.put((mask_pos, mask_neg, analyzed_conf, pos_contrib, neg_contrib))

            end_time = time.time()
            print(f"✅ LIME 분석 완료 (소요 시간: {end_time - start_time:.2f}s)")

        except Empty:
            continue # 큐 타임아웃 발생 시, 'running' 상태 확인 후 계속 진행
        except Exception as e:
            print(f"❌ LIME 워커 오류: {e}")
            time.sleep(1)

# === 4. Drone Control (드론 제어) ===

def handle_keyboard_control(tello):
    """Pygame을 사용하여 키보드 입력을 처리하고 드론을 제어하는 함수 (속도 기반 제어)"""
    # 제어 값 초기화 (left/right, forward/backward, up/down, yaw velocity)
    lr, fb, ud, yv = 0, 0, 0, 0
    
    # Pygame 이벤트 처리 (이륙, 착륙, 종료 등 단일 이벤트)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            global running
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_t: # T: 이륙
                print("이륙 시도...")
                try: tello.takeoff()
                except Exception as e: print(f"이륙 실패: {e}")
            elif event.key == pygame.K_l: # L: 착륙
                print("착륙 시도...")
                try: tello.land()
                except Exception as e: print(f"착륙 실패: {e}")

    # 키 눌림 상태 확인 (이동 제어 - 지속적인 속도 명령)
    keys = pygame.key.get_pressed()
    # 좌/우 이동 (A/D)
    if keys[pygame.K_a]: lr = -CONTROL_SPEED
    elif keys[pygame.K_d]: lr = CONTROL_SPEED
    
    # 전/후진 이동 (W/S)
    if keys[pygame.K_w]: fb = CONTROL_SPEED
    elif keys[pygame.K_s]: fb = -CONTROL_SPEED
    
    # 상승/하강 (화살표 위/아래)
    if keys[pygame.K_UP]: ud = CONTROL_SPEED
    elif keys[pygame.K_DOWN]: ud = -CONTROL_SPEED
    
    # 회전 (화살표 좌/우)
    if keys[pygame.K_LEFT]: yv = -CONTROL_SPEED
    elif keys[pygame.K_RIGHT]: yv = CONTROL_SPEED
        
    # RC 제어 명령 전송
    try:
        # 드론이 비행 중일 때만 명령 전송
        if tello.is_flying:
            tello.send_rc_control(lr, fb, ud, yv)
    except Exception as e:
        print(f"RC 제어 명령 전송 실패: {e}")
        
# === 5. Visualization (시각화) ===
def visualize(display_frame, results, max_conf, lime_result):
    """OpenCV를 사용하여 실시간 영상 위에 HUD, YOLO 탐지 결과, LIME 설명을 그리는 함수"""
    
    # 1. LIME 오버레이
    if lime_result:
        # LIME 결과 언패킹
        mask_pos, mask_neg, analyzed_conf, pos_contrib, neg_contrib = lime_result
        
        # LIME 마스크 크기를 분석 크기(ANALYSIS_SIZE)에서 표시 크기(DISPLAY_SIZE)로 조정
        mask_pos_resized = cv2.resize(mask_pos.astype(np.uint8), (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST)
        mask_neg_resized = cv2.resize(mask_neg.astype(np.uint8), (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST)

        overlay = display_frame.copy()
        
        # 긍정적 기여도 (빨간색 - 충돌 위험 요인) (BGR 형식)
        overlay[mask_pos_resized > 0] = [0, 0, 255] 
        
        # 부정적 기여도 (초록색 - 안전 요인)
        overlay[mask_neg_resized > 0] = [0, 255, 0]
        
        # 오버레이와 원본 이미지를 합성 (오버레이 투명도 40%)
        cv2.addWeighted(overlay, 0.4, display_frame, 0.6, 0, display_frame)

        # LIME 정보 텍스트 표시 (화면 하단)
        cv2.putText(display_frame, "LIME Explanation (Red: Risk Factor, Green: Safety Factor)", 
                    (10, DISPLAY_HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        contrib_text = f"Risk Contribution: {pos_contrib:.1f}%, Safety Contribution: {neg_contrib:.1f}% (Analyzed @ {analyzed_conf*100:.1f}%)"
        cv2.putText(display_frame, contrib_text, 
                    (10, DISPLAY_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 2. YOLO 바운딩 박스
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            # 좌표를 분석 크기에서 표시 크기로 스케일링
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1 = int(x1 * DISPLAY_WIDTH / ANALYSIS_WIDTH)
            x2 = int(x2 * DISPLAY_WIDTH / ANALYSIS_WIDTH)
            y1 = int(y1 * DISPLAY_HEIGHT / ANALYSIS_HEIGHT)
            y2 = int(y2 * DISPLAY_HEIGHT / ANALYSIS_HEIGHT)

            conf = box.conf[0].item()
            # 신뢰도에 따라 색상 변경 (임계값 이상: 빨강, 미만: 노랑)
            color = (0, 0, 255) if conf >= CONF_THRESHOLD else (0, 255, 255)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, f"{conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # 3. HUD (상태 텍스트 - 화면 상단)
    is_high_risk = max_conf >= CONF_THRESHOLD
    status_color = (0, 0, 255) if is_high_risk else (0, 255, 0) # 빨강 또는 초록
    status_text = f"Real-Time Collision Risk: {max_conf*100:.1f}%"
    
    # 위험 상태에 따라 텍스트 강조
    if is_high_risk:
            cv2.putText(display_frame, "WARNING! " + status_text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
    else:
        cv2.putText(display_frame, status_text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
        
    return display_frame

# === 6. Main Application Loop (메인 애플리케이션 루프) ===

def main():
    global model, running
    
    # 모델 초기화
    model = load_yolo_model(MODEL_PATHS)
    if not model: return
    
    # Pygame 초기화 (제어용 창)
    pygame.init()
    screen = pygame.display.set_mode((300, 100))
    pygame.display.set_caption("Tello Control (Focus Here)")

    # Tello 드론 연결 및 초기화
    tello = Tello()
    try:
        tello.connect()
        print(f"✅ Tello 연결됨. 배터리: {tello.get_battery()}%")
        tello.streamon() # 영상 스트리밍 시작
        frame_read = tello.get_frame_read() # 영상 프레임 리더 객체 확보
    except Exception as e:
        print(f"❌ Tello 연결 실패: {e}. Wi-Fi 연결을 확인하세요.")
        pygame.quit()
        return

    # 워커 스레드 시작 (알람 및 LIME 분석)
    alarm_thread = threading.Thread(target=alarm_worker)
    alarm_thread.start()
    lime_thread = threading.Thread(target=lime_analysis_worker)
    lime_thread.start()

    print("🚀 시스템 준비 완료. 제어를 위해 Pygame 창을 활성화하세요 (T: 이륙, L: 착륙, WASD/화살표).")
    
    # 현재 표시 중인 LIME 결과를 저장하는 변수
    current_lime_result = None

    # 메인 루프
    while running:
        # 1. 제어 입력 처리
        handle_keyboard_control(tello)

        # 2. 영상 프레임 가져오기
        frame_bgr = frame_read.frame
        if frame_bgr is None:
            time.sleep(0.01)
            continue
            
        # 프레임 크기 조정 (표시용 및 분석용)
        display_frame = cv2.resize(frame_bgr, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        analysis_frame_bgr = cv2.resize(frame_bgr, (ANALYSIS_WIDTH, ANALYSIS_HEIGHT))
        
        # 분석을 위해 BGR을 RGB로 변환 (YOLO, LIME은 RGB 사용)
        analysis_frame_rgb = cv2.cvtColor(analysis_frame_bgr, cv2.COLOR_BGR2RGB)

        # 3. 실시간 YOLO 예측 (고속 처리)
        results = model.predict(analysis_frame_rgb, verbose=False, imgsz=(ANALYSIS_HEIGHT, ANALYSIS_WIDTH))
        
        # 최대 충돌 확률 계산
        max_conf = 0.0
        if len(results[0].boxes) > 0:
            max_conf = results[0].boxes.conf.max().item()

        # 4. 위험 평가 및 알람 제어
        is_high_risk = max_conf >= CONF_THRESHOLD
        # 위험 상태에 따라 알람 이벤트 설정/해제
        alarm_event.set() if is_high_risk else alarm_event.clear()

        # 5. LIME 분석 요청 (비차단 방식)
        # LIME 워커가 대기 중일 때만(입력 큐가 비어 있을 때) 분석 요청
        if lime_input_queue.empty():
            # 집중 분석을 위한 객체 마스크 생성
            object_mask = get_object_mask(analysis_frame_rgb, model)
            # 최적화: 유의미한 객체(신뢰도 0.2 이상)가 탐지되었을 때만 LIME 실행
            if np.any(object_mask) and max_conf > 0.2: 
                # 분석할 데이터(이미지 복사본, 마스크)를 큐에 넣음
                lime_input_queue.put((analysis_frame_rgb.copy(), object_mask))

        # 6. LIME 결과 업데이트 (비차단 방식)
        try:
            # 새로운 LIME 분석 결과가 도착했는지 확인 (대기 없이 즉시 확인)
            current_lime_result = lime_output_queue.get_nowait() 
        except Empty:
            pass # 새 결과가 없으면 이전 결과를 계속 표시

        # 7. 시각화 처리
        display_frame = visualize(display_frame, results, max_conf, current_lime_result)

        # 8. 화면 표시
        cv2.imshow("Tello Collision Detection & LIME Explanation System", display_frame)

        # OpenCV 창에서 ESC 키 입력 시 종료
        if cv2.waitKey(1) & 0xFF == 27:
            running = False

    # === 7. Cleanup (종료 및 정리) ===
    print("🛑 시스템 종료 중...")
    running = False
    alarm_event.set() # 알람 스레드가 종료될 수 있도록 신호 전달
    
    # 스레드 종료 대기
    if alarm_thread.is_alive():
        alarm_thread.join(timeout=2)
    if lime_thread.is_alive():
        lime_thread.join(timeout=5)

    # Tello 정리
    try:
        if tello.is_flying:
            print("드론 착륙 중...")
            tello.land()
        tello.streamoff() # 스트리밍 종료
        tello.end()       # 연결 종료
    except Exception as e:
        print(f"Tello 정리 중 오류 발생: {e}")

    cv2.destroyAllWindows()
    pygame.quit()
    print("✅ 종료 완료.")

# 스크립트 직접 실행 시 main 함수 호출
if __name__ == '__main__':
    main()