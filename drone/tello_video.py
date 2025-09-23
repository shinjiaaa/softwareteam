import cv2
import pygame
from djitellopy import Tello

# pygame 초기화
pygame.init()

screen_width, screen_height = 100, 100
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Tello Control")

# Tello 드론 객체 생성
tello = Tello()

# 드론에 연결
tello.connect()
print(f"배터리 잔량: {tello.get_battery()}%")

# 영상 스트리밍 시작
tello.streamon()

# Tello 제어 함수 정의
def get_keyboard_input(tello):
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_t: # 이륙
                print("이륙...")
                tello.takeoff()
            elif event.key == pygame.K_l: # 착륙
                print("착륙...")
                tello.land()
            elif event.key == pygame.K_w: # 앞으로 이동
                tello.move_forward(15)
            elif event.key == pygame.K_s: # 뒤로 이동
                tello.move_back(15)
            elif event.key == pygame.K_a: # 왼쪽으로 이동
                tello.move_left(15)
            elif event.key == pygame.K_d: # 오른쪽으로 이동
                tello.move_right(15)
            elif event.key == pygame.K_UP: # 위로 상승
                tello.move_up(15)
            elif event.key == pygame.K_DOWN: # 아래로 하강
                tello.move_down(15)
            elif event.key == pygame.K_LEFT: # 반시계 방향 회전
                tello.rotate_counter_clockwise(15)
            elif event.key == pygame.K_RIGHT: # 시계 방향 회전
                tello.rotate_clockwise(15)

# 메인 루프
running = True
while running:
    # 1. 키보드 입력 처리
    get_keyboard_input(tello)

    # 2. 영상 프레임 가져오기
    frame = tello.get_frame_read().frame

    # 3. 화면에 영상 표시
    if frame is not None:
        cv2.imshow("Tello Camera Feed", frame)

    # 'q' 키를 누르면 프로그램 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

# 착륙
print("프로그램 종료 및 착륙...")
tello.land()

# 연결 종료
tello.end()

# 창 닫기
cv2.destroyAllWindows()
pygame.quit()