"""
드론 충돌 감지를 위한 알림 시스템
- 시스템 알림 (Windows Toast, macOS Notification Center)
- 경고음
- 로깅
"""

import subprocess
import logging
import platform
from typing import Optional
from plyer import notification

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def notify(title: str, message: str, timeout: int = 5):
    """
    시스템 알림 표시
    Args:
        title: 알림 제목
        message: 알림 내용
        timeout: 알림 표시 시간(초)
    """
    try:
        notification.notify(
            title=title,
            message=message,
            timeout=timeout,
            app_icon=None  # TODO: 드론 아이콘 추가
        )
        logging.info(f"알림 발송: {title} - {message}")
    except Exception as e:
        logging.error(f"알림 실패: {str(e)}")
        # 실패 시 터미널에라도 출력
        print(f"\n[!] {title}: {message}")


def beep(frequency: int = 1000, duration: int = 500):
    """
    시스템 경고음 재생
    Args:
        frequency: 경고음 주파수 (Windows 전용, Hz)
        duration: 경고음 지속 시간 (Windows 전용, ms)
    """
    try:
        system = platform.system()
        if system == "Darwin":  # macOS
            subprocess.Popen(
                ["afplay", "/System/Library/Sounds/Basso.aiff"],  # 더 경고성이 있는 사운드로 변경
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
        elif system == "Windows":
            import winsound
            winsound.Beep(frequency, duration)
        else:  # Linux
            print("\a", end="", flush=True)
        logging.debug("경고음 재생 완료")
    except Exception as e:
        logging.error(f"경고음 재생 실패: {str(e)}")

def alert_collision(confidence: float, danger_level: str = "주의"):
    """
    드론 충돌 위험 알림
    Args:
        confidence: 감지 신뢰도 (0.0 ~ 1.0)
        danger_level: 위험 수준 ("주의", "경고", "위험")
    """
    levels = {
        "주의": {"timeout": 3, "freq": 800},
        "경고": {"timeout": 5, "freq": 1000},
        "위험": {"timeout": 7, "freq": 1200}
    }
    level_info = levels.get(danger_level, levels["주의"])
    
    message = f"충돌 위험 감지 ({confidence*100:.1f}%)"
    if danger_level == "위험":
        message = "⚠️ " + message
    
    notify(
        f"드론 충돌 {danger_level}!", 
        message,
        timeout=level_info["timeout"]
    )
    beep(frequency=level_info["freq"])

if __name__ == "__main__":
    # 알림 시스템 테스트
    print("알림 시스템 테스트 시작")
    
    # 1. 기본 알림 테스트
    notify("테스트 알림", "알림 시스템이 정상 작동 중입니다.")
    beep()
    
    # 2. 다양한 위험 수준 테스트
    for level in ["주의", "경고", "위험"]:
        print(f"{level} 수준 알림 테스트...")
        alert_collision(0.75, level)
        import time
        time.sleep(2)  # 알림 간 간격
    
    print("알림 시스템 테스트 완료.")
