# tello_video.py
import cv2
import pygame
import time
from djitellopy import Tello
# data_explain.py에서 클래스 임포트
try:
    from data_explain import CollisionDetectorLIME
except ImportError as e:
    print(f"[ERROR] data_explain.py를 찾을 수 없습니다. {e}")
    exit()

# --- 설정값 ---
# TODO: YOLO 가중치 파일 경로 설정 (None이면 자동 탐색/기본 모델 사용)
WEIGHTS_PATH = None 
# 드론 제어 속도 설정 (0 ~ 100)
DRONE_SPEED = 50
# 메인 루프 목표 FPS (안정적인 실행을 위함)
TARGET_FPS = 30
# Tello 기본 해상도
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
# ---------------

# 1. Pygame 초기화 (키보드 입력용)
pygame.init()
# 키 입력을 받기 위한 작은 창 생성. 이 창이 활성화되어 있어야 조종 가능.
screen = pygame.display.set_mode((300, 100))
pygame.display.set_caption("Tello Control (Focus here)")
clock = pygame.time.Clock()

# 2. Tello 드론 연결
print("[INFO] Connecting to Tello...")
tello = Tello()
try:
    tello.connect()
    print(f"[INFO] Connected. Battery: {tello.get_battery()}%")
    tello.streamon()
    frame_reader = tello.get_frame_read()
    tello.send_rc_control(0, 0, 0, 0) # 초기화 (속도 0)
except Exception as e:
    print(f"[ERROR] Failed to connect to Tello: {e}")
    pygame.quit()
    exit()

# 3. 충돌 감지 시스템 초기화
print("[INFO] Initializing Collision Detector (YOLO+LIME)...")
try:
    detector = CollisionDetectorLIME(
        weights_path=WEIGHTS_PATH,
        imgsz=512,              # YOLO 입력 크기 (성능에 따라 조절, 예: 640, 512, 320)
        conf_thres=0.4,         # 탐지 임계값
        min_conf_for_lime=0.6,  # LIME 적용 최소 임계값
        warning_threshold=0.75, # 경고 발생 임계값 (게이지에 표시됨)
        topk=1                  # 가장 위험한 1개 객체만 LIME 분석
    )
    # LIME 워커 스레드 시작
    detector.start_worker()
except Exception as e:
    print(f"[ERROR] Failed to initialize Collision Detector: {e}")
    tello.end()
    pygame.quit()
    exit()


# Tello 제어 함수 (Non-blocking 방식)
def get_keyboard_control(tello, speed):
    """Pygame 입력을 받아 RC 컨트롤 값(속도 벡터)을 계산합니다."""
    # lr: 좌우, fb: 전후, ud: 상하, yv: 회전 속도
    lr, fb, ud, yv = 0, 0, 0, 0
    
    # 이벤트 처리 (이륙/착륙/종료 등 단발성 명령)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return None # 종료 신호
        if event.type == pygame.KEYDOWN:
            try:
                if event.key == pygame.K_t: # 이륙 (T)
                    print("Control: Takeoff...")
                    tello.takeoff()
                elif event.key == pygame.K_l: # 착륙 (L)
                    print("Control: Landing...")
                    tello.land()
                elif event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                     return None # 종료 신호
            except Exception as e:
                print(f"[ERROR] Drone command failed: {e}")

    # 키 눌림 상태 확인 (연속 이동 제어)
    keys = pygame.key.get_pressed()

    # WASD 이동
    if keys[pygame.K_a]: lr = -speed
    if keys[pygame.K_d]: lr = speed
    if keys[pygame.K_w]: fb = speed
    if keys[pygame.K_s]: fb = -speed

    # 화살표 상/하 (고도)
    if keys[pygame.K_UP]: ud = speed
    if keys[pygame.K_DOWN]: ud = -speed

    # 화살표 좌/우 (회전)
    if keys[pygame.K_LEFT]: yv = -speed
    if keys[pygame.K_RIGHT]: yv = speed

    return lr, fb, ud, yv


# 메인 루프
running = True
print("[INFO] Main loop started. Keep Pygame window focused for control.")
# OpenCV 창 이름 설정
CV_WINDOW_NAME = "Tello Camera - Integrated Collision Explanation System"
cv2.namedWindow(CV_WINDOW_NAME, cv2.WINDOW_NORMAL)

try:
    while running:
        # 1. 키보드 입력 처리
        controls = get_keyboard_control(tello, DRONE_SPEED)

        if controls is None: # 종료 신호 감지
            running = False
            break

        # 2. 드론 제어 신호 전송 (Non-blocking)
        # send_rc_control은 지속적으로 호출되어야 함
        try:
            tello.send_rc_control(*controls)
        except Exception as e:
             print(f"[WARN] Send RC control failed: {e}")

        # 3. 영상 프레임 가져오기 (djitellopy는 RGB 형식 반환)
        frame_rgb = frame_reader.frame

        if frame_rgb is not None:
            # 해상도 표준화
            if frame_rgb.shape[0] != FRAME_HEIGHT or frame_rgb.shape[1] != FRAME_WIDTH:
                frame_rgb = cv2.resize(frame_rgb, (FRAME_WIDTH, FRAME_HEIGHT))

            # [중요] RGB를 BGR로 변환 (OpenCV, YOLO 모델 호환성)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # 4. 장애물 감지 및 LIME 시각화 처리
            # 모든 시각화는 이 함수 내에서 완료됨
            processed_frame = detector.process_frame(frame_bgr)

            # 5. 화면에 영상 표시
            cv2.imshow(CV_WINDOW_NAME, processed_frame)

        # OpenCV 대기 (종료 키 확인용)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            running = False
        
        # 목표 FPS 유지
        clock.tick(TARGET_FPS)

except KeyboardInterrupt:
    print("[INFO] Keyboard interrupt received (Ctrl+C).")

finally:
    # 프로그램 종료 시 안전한 처리 (매우 중요)
    print("[INFO] Shutting down...")

    # LIME 워커 스레드 정지
    if 'detector' in locals() and detector:
        detector.stop_worker()

    # 드론 안전 착륙
    try:
        print("Landing drone safely...")
        tello.send_rc_control(0, 0, 0, 0) # 이동 중이라면 정지 명령 전송
        tello.land()
        time.sleep(3) # 착륙 완료 대기
    except Exception as e:
        print(f"[WARN] Landing command failed during shutdown: {e}")

    # 연결 종료
    if 'tello' in locals() and tello:
        tello.streamoff()
        tello.end()
    print("Tello connection closed.")

    # 창 닫기
    cv2.destroyAllWindows()
    pygame.quit()
    print("[INFO] Shutdown complete.")