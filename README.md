# softwareteam - LIME & TELLO SDK 기반 충돌 감지 모니터링 및 알림 시스템

## 개발 수칙
1. directory를 구조화하여 개발을 진행할 것 (객체지향적 개발 지향)
2. repository에 각자 branch 생성
   1. 1차로 branch에 본인이 작업한 코드를 올리기
   2. 그 후 main pull 받은 후 충돌이 존재할 시 자신의 branch에서 해결
   3. 해결 후 main에 merge 진행 (main에서의 충돌을 막기 위함)
4. commit message 상세하게 작성할 것 (ex. model 데이터셋 추가)
5. 주석을 상세히 작성할 것 - 유지보수를 원활하게 진행하기 위함
6. 함수명, 변수명 snake_case 형식으로 통일할 것
7. main에 변경 사항이 있을 경우, 터미널에 git pull origin main 입력 후 개발 진행 (가장 최신 코드에서 작업하기 위함)

## Directory
1. model 경로: runs/detect/exp/weights/best.pt
2. dataset/: train, valid, test 폴더 내 각 image, lebel 존재함
3. static/: UI
4. app.py: 엔드포인트 정의 및 UI 실행 함수
5. detector.py: 실시간 영상 사용하여 yolo, lime 적용
6. drone_manager.py: 드론 제어

## 출처
1. dataset
   1. https://universe.roboflow.com/tylervisimoai/drone-crash-avoidance
   2. https://github.com/VisDrone/VisDrone-Dataset (Task 2)

## 대시보드 실행법
대시보드 접속 
http://127.0.0.1:8000

드론 연결 후 python app.py 실행
