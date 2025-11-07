import time, threading, cv2
from typing import Optional, Dict, Any
from detector import CollisionDetectorLIME

try:
    from djitellopy import Tello
    TELLO_AVAILABLE=True
except ImportError:
    Tello=None; TELLO_AVAILABLE=False
    print("[Manager] djitellopy not found. Tello control disabled.")

class DroneManager:
    def __init__(self, detector:CollisionDetectorLIME,use_webcam=False):
        self.detector=detector; self.use_webcam=use_webcam
        if not TELLO_AVAILABLE and not use_webcam:
            print("[Manager] Tello requested but library missing. Forcing webcam mode."); self.use_webcam=True
        self.tello:Optional[Tello]=None; self.webcam:Optional[cv2.VideoCapture]=None
        self.FRAME_WIDTH=960; self.FRAME_HEIGHT=720; self.DRONE_SPEED=50
        self.is_connected=False; self.is_streaming=False; self.battery_level=0
        self.latest_processed_data={"frame":None,"risk":None}
        self.lock=threading.Lock(); self.stop_event=threading.Event()
        self.latest_telemetry=self._generate_telemetry_snapshot(None)

    def connect(self):
        if self.is_connected: return True
        try:
            if self.use_webcam: self._connect_webcam()
            else:
                try: self._connect_tello()
                except Exception as e:
                    print(f"[Manager] Tello connection failed: {e}. Falling back to Webcam."); self.use_webcam=True; self._connect_webcam()
            self.is_connected=True
            self.detector.start_worker()
            self.processing_thread=threading.Thread(target=self._main_loop,daemon=True)
            self.processing_thread.start()
            print("[Manager] System started successfully."); return True
        except Exception as e:
            print(f"[Manager] Final connection failed: {e}"); self.is_connected=False; return False

    def _connect_tello(self):
        print("[Manager] Connecting to Tello...")
        self.tello=Tello(); self.tello.connect(); self.battery_level=self.tello.get_battery()
        if self.battery_level==0: raise RuntimeError("Tello connection failed.")
        print(f"[Manager] Tello connected. Battery: {self.battery_level}%")
        self.tello.streamon(); self.frame_reader=self.tello.get_frame_read(); self.is_streaming=True

    def _connect_webcam(self):
        print("[Manager] Connecting to Webcam...")
        self.webcam=cv2.VideoCapture(0)
        if not self.webcam.isOpened(): raise RuntimeError("Could not open webcam.")
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH,self.FRAME_WIDTH)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT,self.FRAME_HEIGHT)
        self.battery_level=100; self.is_streaming=True

    def shutdown(self):
        print("[Manager] Shutting down..."); self.stop_event.set()
        if hasattr(self,'processing_thread') and self.processing_thread.is_alive(): self.processing_thread.join(timeout=5.0)
        if self.detector: self.detector.stop_worker()
        if self.tello and not self.use_webcam:
            try: self.tello.send_rc_control(0,0,0,0); self.tello.land(); time.sleep(3)
            except: pass
            try: self.tello.streamoff(); self.tello.end()
            except: pass
        if self.webcam: self.webcam.release()
        self.is_connected=False; print("[Manager] Shutdown complete.")

    def _main_loop(self):
        last_battery_check=time.time()
        while not self.stop_event.is_set():
            if not self.is_streaming: time.sleep(0.1); continue
            try:
                frame_bgr=self._get_frame(); 
                if frame_bgr is None: time.sleep(0.01); continue
                processed_frame,risk_data=self.detector.process_frame(frame_bgr)
                collision_conf=risk_data.get("max_conf",0.0)
                telemetry=self._generate_telemetry_snapshot(risk_data)
                with self.lock:
                    self.latest_processed_data={"frame":processed_frame,"risk":risk_data}
                    self.latest_telemetry=telemetry
                if not self.use_webcam and time.time()-last_battery_check>5:
                    self.battery_level=self.tello.get_battery(); last_battery_check=time.time()
            except Exception as e: print(f"[Manager DEBUG] Loop error: {e}"); time.sleep(0.05)

    def _get_frame(self):
        if self.use_webcam and self.webcam:
            ret,frame=self.webcam.read(); return frame if ret else None
        elif self.tello and self.tello.get_frame_read():
            return self.tello.get_frame_read().frame
        return None

    def _generate_telemetry_snapshot(self,risk_data:Optional[dict]):
        return {
            "type":"telemetry",
            "connected":self.is_connected,
            "battery":self.battery_level,
            "fps":self.detector.fps,
            "risk":risk_data or {"max_conf":0.0,"level":"safe"},
        }

    def get_latest_frame(self):
        with self.lock: return self.latest_processed_data.get("frame")

    def get_latest_telemetry(self):
        with self.lock: return self.latest_telemetry


    def send_command(self, command: str, data: Optional[Dict[str, Any]] = None):
        # (이전과 동일)
        if self.use_webcam:
             return
        if not self.tello: return
        try:
            if command == "rc_control":
                if data:
                    lr = int(data.get("lr", 0) * self.DRONE_SPEED)
                    fb = int(data.get("fb", 0) * self.DRONE_SPEED)
                    ud = int(data.get("ud", 0) * self.DRONE_SPEED)
                    yv = int(data.get("yv", 0) * self.DRONE_SPEED)
                    self.tello.send_rc_control(lr, fb, ud, yv)
            elif command == "takeoff":
                self.tello.takeoff()
            elif command == "land":
                self.tello.land()
        except Exception as e:
            print(f"[Manager] Failed to send command '{command}': {e}")

# 싱글톤 인스턴스 초기화
try:
    initial_detector = CollisionDetectorLIME(weights_path=None)
    # 중요: 실제 드론 사용 시 use_webcam=False로 설정
    # 테스트를 위해 웹캠을 사용하려면 True로 변경하세요.
    drone_manager = DroneManager(detector=initial_detector, use_webcam=False) 
except Exception as e:
    print(f"[System Init Error] Failed to initialize Detector or Manager: {e}")
    exit(1)