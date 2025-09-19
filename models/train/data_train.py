from ultralytics import YOLO
from pathlib import Path
import torch

def continue_training():
    print("기존 모델 추가 학습 (50 epochs)")
    print("="*40)
    
    # 기존 모델 경로
    existing_model = "../../data/runs/detect/train5/weights/best.pt"
    
    if not Path(existing_model).exists():
        print(f"모델을 찾을 수 없습니다: {existing_model}")
        print("먼저 새 모델 학습을 진행하세요.")
        return None
    
    print(f"기존 모델 로드: {existing_model}")
    model = YOLO(existing_model)
    
    print(f"데이터셋: ../../data/dataset/data.yaml")
    print(f"=== 추가 학습 시작 (50 epochs) ===")
    
    try:
        results = model.train(
            data="../../data/dataset/data.yaml",
            epochs=50,
            imgsz=640,
            batch=16,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            project="../../data/runs/detect",
            name="continue_training",
            exist_ok=True,
            resume=False,
            patience=5,
            lr0=0.001,
            warmup_epochs=3,
        )
        
        print(f"\n추가 학습 완료!")
        print(f"새 모델: ../../data/runs/detect/continue_training/weights/best.pt")
        return results
        
    except Exception as e:
        print(f"학습 중 오류: {e}")
        return None

def train_with_visdrone():
    """VisDrone 데이터셋으로 추가 학습하는 함수"""
    print("🚁 VisDrone 영상 데이터 학습 (안전 모드)")
    print("="*50)
    
    # 기존 향상된 모델 사용
    existing_model = "../../data/runs/detect/continue_training/weights/best.pt"
    
    if not Path(existing_model).exists():
        print(f"❌ 향상된 모델을 찾을 수 없습니다: {existing_model}")
        print("기본 모델 사용...")
        existing_model = "../../data/runs/detect/train5/weights/best.pt"
    
    # VisDrone 변환 데이터 확인
    visdrone_data = "../../data/visdrone_converted/data.yaml"
    
    if not Path(visdrone_data).exists():
        print(f"❌ VisDrone 데이터를 찾을 수 없습니다: {visdrone_data}")
        return None
    
    print(f"📂 기존 모델: {existing_model}")
    print(f"📂 VisDrone 데이터: {visdrone_data}")
    print(f"📊 이미지: 6,471개")
    print(f"🎯 클래스: Person, Vehicle, Building, Tree, Pole")
    
    model = YOLO(existing_model)
    
    print(f"\n🔥 VisDrone 영상 데이터 학습 시작!")
    print(f"   - 안전 설정: 작은 배치, 낮은 학습률")
    print(f"   - GPU 메모리 최적화")
    
    try:
        results = model.train(
            data=visdrone_data,
            epochs=15,      # 적당한 epochs
            imgsz=640,
            batch=4,        # 작은 배치로 메모리 안전
            device='cuda' if torch.cuda.is_available() else 'cpu',
            project="../../data/runs/detect",
            name="visdrone_safe",
            exist_ok=True,
            resume=False,
            patience=8,     # 충분한 patience
            lr0=0.0001,     # 낮은 학습률로 안정성
            warmup_epochs=1,
            save_period=5,
            cache=False,    # 메모리 절약을 위해 캐시 비활성화
            workers=4,      # 워커 수 줄임
            verbose=True
        )
        
        print(f"\n✅ VisDrone 영상 학습 완료!")
        print(f"🎯 새 모델: ../../data/runs/detect/visdrone_safe/weights/best.pt")
        print(f"📈 이제 더 많은 차량과 사람을 정확히 감지할 수 있습니다!")
        return results
        
    except Exception as e:
        print(f"❌ 학습 중 오류: {e}")
        print("\n🔧 더 안전한 설정으로 재시도...")
        
        # 더 안전한 설정으로 재시도
        try:
            results = model.train(
                data=visdrone_data,
                epochs=10,
                imgsz=416,      # 이미지 크기 줄임
                batch=2,        # 배치 더 줄임
                device='cuda' if torch.cuda.is_available() else 'cpu',
                project="../../data/runs/detect",
                name="visdrone_ultra_safe",
                exist_ok=True,
                resume=False,
                patience=5,
                lr0=0.00005,    # 더 낮은 학습률
                warmup_epochs=1,
                cache=False,
                workers=2,      # 워커 최소화
                verbose=True
            )
            
            print(f"\n✅ 안전 모드로 VisDrone 학습 완료!")
            print(f"🎯 새 모델: ../../data/runs/detect/visdrone_ultra_safe/weights/best.pt")
            return results
            
        except Exception as e2:
            print(f"❌ 안전 모드도 실패: {e2}")
            return None

if __name__ == "__main__":
    print("🚁 드론 충돌 감지 모델 학습")
    print("="*40)
    print("1. 기존 모델 추가 학습")
    print("2. VisDrone 데이터 추가 학습")
    print("="*40)
    
    choice = input("선택하세요 (1/2): ")
    
    if choice == "1":
        continue_training()
    elif choice == "2":
        train_with_visdrone()
    else:
        print("기본적으로 기존 모델 추가 학습을 실행합니다.")
        continue_training()
