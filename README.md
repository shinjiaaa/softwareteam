# softwareteam - 장애물 충돌 감지 모니터링 및 알림 시스템

## 개발 수칙
1. directory를 구조화하여 개발을 진행할 것 (객체지향적 개발 지향)
2. repository에 각자 branch 생성
   1. 1차로 branch에 본인이 작업한 코드를 올리기
   2. 그 후 main pull 받은 후 충돌이 존재할 시 자신의 branch에서 해결
   3. 해결 후 main에 merge 진행 (main에서의 충돌을 막기 위함)
4. commit message 상세하게 작성할 것 (본인 이름 및 수정한 부분 ex. shinjia - model 데이터 셋 추가)
5. 주석을 상세히 작성할 것 - 유지보수를 원활하게 진행하기 위함
6. 함수명, 변수명 snake_case 형식으로 통일할 것

## Directory
1. drone: TELLO SDK 이착륙 제어
2. data: 원본 데이터(드론 충돌 영상 & 이미지), YOLOv8 학습용 파일(.yaml)
    1. dataset
        1. https://universe.roboflow.com/tylervisimoai/drone-crash-avoidance
        2. https://github.com/VisDrone/VisDrone-Dataset
    2. model 위치: data/runs/detect/train5/weights/best.pt
3. alert: 알림 & TTS 제어
4. models: Model Train, Test, LIME 적용
    1. data_train.py: model train file
    2. data_test.py: model이 image를 보고 충돌 가능성이 있는 객체인지 확인하는 파일
    3. data_explain.py: (LIME) model이 왜 충돌 가능성이 있는 객체라고 판단했는지 설명하는 파일
5. ui: 유저 인터페이스 (frontend)
