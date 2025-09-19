# softwareteam - LIME 기반의 장애물 충돌 감지 드론 시스템 개발

## 개발 수칙
1. 디렉토리를 구조화하여 개발을 진행할 것 (객체지향적 개발 지향)
2. 레포지토리에 각자 브랜치를 생성한 후, 각자 브랜치에 1차적으로 본인이 작업한 코드를 올리기 -> 그 후 본인의 브랜치에서 메인을 pull 받은 후 충돌이 존재할 시 각자 브랜치에서 해결 후 메인에 업데이트 할 것 (메인 코드에서의 충돌을 막기 위함)
3. 커밋 메시지를 상세하게 작성할 것 (본인 이름 및 수정한 부분 ex. shinjia - model 가중치 수정)
4. 코드 주석을 상세히 작성할 것 - 유지보수를 원활하게 진행하기 위함
5. 함수명, 변수명 snake_case 형식으로 통일

## directory
1. drone: TELLO SDK 이착륙 제어
2. data: 원본 데이터(드론 충돌 영상 & 이미지), YOLOv8 학습용 파일(.yaml)
    1. dataset
        1. https://universe.roboflow.com/tylervisimoai/drone-crash-avoidance
        2. https://github.com/VisDrone/VisDrone-Dataset
    2. model 위치: data/dataset/runs/detect/train5/weights/best.pt
3. alert: 알림 & TTS 제어
4. models: Model Train, Test, LIME 적용
    1. data_train.py: model train file
    2. data_test.py: model이 image를 보고 충돌 가능성이 있는 객체인지 확인하는 파일
    3. data_explain.py: (LIME) model이 왜 충돌 가능성이 있는 객체라고 판단했는지 설명하는 파일
5. ui: 유저 인터페이스 (front-end)

