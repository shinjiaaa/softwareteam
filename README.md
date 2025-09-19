# softwareteam
소프트웨어개발론PBL - LIME 기반의 장애물 충돌 감지 드론 시스템 개발
-
## directory
1. drone: TELLO SDK 이착륙 제어
2. data: 원본 데이터(드론 충돌 영상 & 이미지), YOLOv8 학습용 파일(.yaml)
3. alert: 알림 & TTS 제어
4. models: Model Train, Test, LIME 적용
    1. data_train.py -> model train file
    2. data_test.py -> model이 image를 보고 충돌 가능성이 있는 객체인지 확인하는 파일
    3. data_explain.py -> (LIME) model이 왜 충돌 가능성이 있는 객체라고 판단했는지 설명하는 파일
5. ui: 드론 충돌 감지 모니터링 및 알림 시스템 (유저 인터페이스)

