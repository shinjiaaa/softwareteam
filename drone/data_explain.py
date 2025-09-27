# data_explain.py
import os
import time
import subprocess
import sys
from typing import Optional, Tuple
from threading import Thread, Lock, Event

import cv2
import numpy as np
from ultralytics import YOLO

# LIME
from lime import lime_image
from skimage.segmentation import slic

# ---------- 알림 및 유틸리티 (이전과 동일) ----------
try:
    from plyer import notification
    def notify(title: str, message: str):
        try:
            Thread(target=notification.notify, kwargs={'title': title, 'message': message, 'timeout': 1}, daemon=True).start()
        except Exception:
            pass
except ImportError:
    def notify(title: str, message: str):
        print(f"[NOTIFICATION] {title}: {message}")

def beep():
    """시스템 사운드 재생."""
    def _beep():
        try:
            if os.name == 'nt': # Windows
                import winsound
                winsound.Beep(1000, 200)
            elif sys.platform == "darwin": # macOS
                subprocess.Popen(["afplay", "/System/Library/Sounds/Glass.aiff"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    Thread(target=_beep, daemon=True).start()

def put_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

# ---------- 시각화 함수들 (이전과 동일) ----------

def draw_risk_indicator(frame, max_conf, warning_threshold):
    """화면 상단에 충돌 위험도(최대 신뢰도) 게이지를 표시합니다."""
    h, w = frame.shape[:2]
    bar_height = 20
    cv2.rectangle(frame, (0, 0), (w, bar_height), (50, 50, 50), -1)

    risk_width = int(w * max_conf)
    
    # 색상 그라데이션: Green(0.0) -> Yellow(0.5) -> Red(1.0) (BGR)
    if max_conf < 0.5:
        r, g = int(255 * (max_conf * 2)), 255
    else:
        r, g = 255, int(255 * (1 - (max_conf-0.5) * 2))
    color = (0, g, r)

    if risk_width > 0:
        cv2.rectangle(frame, (0, 0), (risk_width, bar_height), color, -1)

    thresh_x = int(w * warning_threshold)
    if 0 <= thresh_x < w:
        cv2.line(frame, (thresh_x, 0), (thresh_x, bar_height), (255, 255, 255), 2)

    text = f"RISK LEVEL: {max_conf*100:.1f}%"
    text_color = (255, 255, 255) if max_conf < 0.6 else (10, 10, 10)
    cv2.putText(frame, text, (10, bar_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)


def draw_boxes(frame, results, conf_thres=0.35, names=None):
    """YOLO 박스를 그리고 신뢰도 순으로 정렬된 리스트 반환."""
    if not results or getattr(results[0], "boxes", None) is None:
        return frame, []

    sorted_boxes = sorted(results[0].boxes, key=lambda b: float(b.conf[0]) if b.conf is not None else 0.0, reverse=True)
    
    boxes_info = []
    for b in sorted_boxes:
        if b.conf is None or b.xyxy is None: continue
        conf = float(b.conf[0])
        if conf < conf_thres: break

        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
        cls = int(b.cls[0]) if b.cls is not None else -1
        
        color = (0, int(255 * (1-conf)), int(255 * conf))
        label = names[cls] if names and 0 <= cls < len(names) else str(cls)
        label = f"{label} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (w_text, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_y = max(y1, h+10)
        cv2.rectangle(frame, (x1, text_y-h-10), (x1+w_text, text_y), color, -1)
        cv2.putText(frame, label, (x1, text_y-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        boxes_info.append((x1, y1, x2, y2, cls, conf))
    return frame, boxes_info

def _blend_single(bg: np.ndarray, fg_color_bgr: Tuple[int, int, int], mask: np.ndarray, alpha: float) -> np.ndarray:
    """단일 마스크 블렌딩 헬퍼."""
    if mask is None or np.max(mask) == 0:
        return bg
    
    m = cv2.GaussianBlur(mask, (0, 0), 2.5)
    m3 = cv2.merge([m, m, m])
    
    fg = np.zeros_like(bg)
    fg[:] = fg_color_bgr

    out = (bg.astype(np.float32) * (1.0 - alpha*m3)
           + fg.astype(np.float32) * (alpha*m3))
    
    return np.clip(out, 0, 255).astype(np.uint8)

def blend_dual_mask_sequential(frame_bgr: np.ndarray, pos_mask01: np.ndarray, neg_mask01: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    """긍정(Red) 및 부정(Green) 마스크를 순차적으로 합성하여 Red를 우선 표시."""
    h, w = frame_bgr.shape[:2]
    if pos_mask01 is None or neg_mask01 is None or pos_mask01.shape != (h, w):
        return frame_bgr

    COLOR_GREEN = (0, 255, 0)
    COLOR_RED = (0, 0, 255)

    # 1. Green (안전 기여) 합성
    out = _blend_single(frame_bgr, COLOR_GREEN, neg_mask01, alpha * 0.9)
    
    # 2. Red (충돌 기여) 합성
    out = _blend_single(out, COLOR_RED, pos_mask01, alpha)
    
    return out


# ---------- LIME 핵심 함수들 ----------

def make_predict_fn_for_roi(model: YOLO, class_id: int):
    """LIME 샘플들을 배치로 처리하여 예측 속도 향상."""
    def predict_proba(batch_rgb: np.ndarray) -> np.ndarray:
        bgr_batch = [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) for rgb in batch_rgb]
        try:
            # 배치 예측 (imgsz=320 권장)
            results = model.predict(source=bgr_batch, verbose=False, imgsz=320)
        except Exception as e:
            print(f"[WARN] YOLO batch prediction error in LIME: {e}")
            return np.array([[1.0, 0.0]] * len(batch_rgb), dtype=np.float32)

        probs = []
        for res in results:
            score = 0.0
            if getattr(res, "boxes", None) is not None:
                for bx in res.boxes:
                    if bx.conf is None or bx.cls is None: continue
                    if int(bx.cls[0]) == class_id:
                        score = max(score, float(bx.conf[0]))
            pos = float(np.clip(score, 0.0, 1.0))
            probs.append([1.0 - pos, pos]) # [부정 확률, 긍정 확률]
        return np.array(probs, dtype=np.float32)
    return predict_proba

# [수정] num_samples를 필수 인자로 변경
def lime_mask_on_roi_weighted(roi_bgr: np.ndarray, model: YOLO, class_id: int,
                              num_samples: int, n_segments=70, num_features=10, compactness=10.0) -> Tuple[np.ndarray, np.ndarray]:
    """ROI에서 LIME Positive/Negative 마스크를 가중치 기반 강도로 생성."""
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    h, w = roi_bgr.shape[:2]

    def segmenter(img): return slic(img, n_segments=n_segments, compactness=compactness, sigma=1, start_label=0)
    explainer = lime_image.LimeImageExplainer()
    predict_fn = make_predict_fn_for_roi(model, class_id)

    try:
        # num_samples 사용
        explanation = explainer.explain_instance(
            roi_rgb, classifier_fn=predict_fn, top_labels=1, hide_color=0,
            num_samples=num_samples, segmentation_fn=segmenter
        )
        
        # 설명 데이터 추출
        label = explanation.top_labels[0]
        segments = explanation.segments
        local_exp = explanation.local_exp[label] # (segment_id, weight) 리스트

        # 중요도(Weight 절대값) 순으로 정렬하여 상위 N개 특징 선택
        sorted_exp = sorted(local_exp, key=lambda item: abs(item[1]), reverse=True)[:num_features]

        pos_mask = np.zeros((h, w), dtype=np.float32)
        neg_mask = np.zeros((h, w), dtype=np.float32)

        if not sorted_exp:
            return pos_mask, neg_mask

        # 마스크 생성 (가중치를 강도로 사용)
        for segment_id, weight in sorted_exp:
            mask_area = (segments == segment_id)
            if weight > 0:
                pos_mask[mask_area] = weight 
            elif weight < 0:
                neg_mask[mask_area] = abs(weight)

        # 시각화를 위해 마스크 정규화 (0.0 ~ 1.0)
        max_weight = max(abs(w) for _, w in sorted_exp)
        if max_weight > 0:
            pos_mask = np.clip(pos_mask / max_weight, 0.0, 1.0)
            neg_mask = np.clip(neg_mask / max_weight, 0.0, 1.0)
              
        return pos_mask, neg_mask

    except Exception as e:
        print(f"[WARN] LIME explanation failed: {e}")
        return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

# ---------- 메인 클래스 ----------

class CollisionDetectorLIME:
    """실시간 충돌 감지 및 LIME 비동기 설명 클래스."""
    # [수정] 생성자에 lime_samples 추가, 기본값 200 설정
    def __init__(self, weights_path: Optional[str], imgsz=512, conf_thres=0.4,
                 min_conf_for_lime=0.6, warning_threshold=0.75, roi_shrink=192, topk=1,
                 lime_samples=200):
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.min_conf_for_lime = min_conf_for_lime
        self.warning_threshold = warning_threshold
        self.roi_shrink = roi_shrink
        self.topk = topk
        self.lime_samples = lime_samples # LIME 샘플 수 저장
        self.lime_alpha = 0.65

        # 1. 모델 로드
        self.weights = self._find_weights(weights_path)
        print(f"[INFO] Loading YOLO model from: {self.weights}")
        print(f"[INFO] LIME Config: num_samples={self.lime_samples}, roi_shrink={self.roi_shrink}")
        self.yolo = YOLO(self.weights)
        self.names = getattr(self.yolo.model, "names", None)

        # 2. LIME 비동기 처리 관련
        self.last_mask_pos: Optional[np.ndarray] = None
        self.last_mask_neg: Optional[np.ndarray] = None
        self.data_lock = Lock()
        self.cancel_event = Event()
        self.latest_job = {"frame": None, "boxes": None}
        self.worker_thread: Optional[Thread] = None

        # 3. FPS 및 경고 쿨타임
        self.t0, self.cnt, self.fps = time.time(), 0, 0.0
        self.last_warning_time = 0

    def _find_weights(self, path):
        """가중치 파일 경로 탐색."""
        if path and os.path.exists(path): return path
        candidates = ["data/runs/detect/train5/weights/best.pt", "best.pt"]
        found = next((c for c in candidates if os.path.exists(c)), None)
        if not found:
             print("[WARN] Custom weights not found. Falling back to 'yolov8n.pt' (COCO dataset) for testing.")
             return "yolov8n.pt"
        return found

    def start_worker(self):
        if self.worker_thread is None or not self.worker_thread.is_alive():
            print("[INFO] Starting LIME worker thread...")
            self.cancel_event.clear()
            self.worker_thread = Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()

    def stop_worker(self):
        print("[INFO] Stopping LIME worker thread...")
        self.cancel_event.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)

    def _worker_loop(self):
        """비동기 LIME 처리 루프."""
        while not self.cancel_event.is_set():
            job_frame, job_boxes = None, None
            
            # 최신 작업 가져오기
            with self.data_lock:
                if self.latest_job["frame"] is not None and self.latest_job["boxes"]:
                    job_frame = self.latest_job["frame"]
                    job_boxes = self.latest_job["boxes"]
                    self.latest_job["frame"] = None
                    self.latest_job["boxes"] = None

            if job_frame is None:
                time.sleep(0.01)
                continue

            H, W = job_frame.shape[:2]
            sel = job_boxes[:self.topk]

            mask_full_pos = np.zeros((H, W), np.float32)
            mask_full_neg = np.zeros((H, W), np.float32)

            for (x1, y1, x2, y2, cls, conf) in sel:
                if self.cancel_event.is_set(): return
                if conf < self.min_conf_for_lime: continue

                # ROI 추출 및 경계 확인
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W-1, x2), min(H-1, y2)
                if x2 <= x1 or y2 <= y1: continue

                # ROI 추출 및 축소
                roi = job_frame[y1:y2, x1:x2]
                try:
                    roi_small = cv2.resize(roi, (self.roi_shrink, self.roi_shrink), interpolation=cv2.INTER_AREA)
                except cv2.error: continue

                # [수정] LIME 실행 (설정된 self.lime_samples 사용)
                m_small_pos, m_small_neg = lime_mask_on_roi_weighted(
                    roi_small, self.yolo, cls, num_samples=self.lime_samples, num_features=10
                )

                # 마스크 복원 및 합성
                m_roi_pos = cv2.resize(m_small_pos, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)
                m_roi_neg = cv2.resize(m_small_neg, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)

                full_p = np.zeros((H, W), np.float32)
                full_n = np.zeros((H, W), np.float32)
                full_p[y1:y2, x1:x2] = m_roi_pos
                full_n[y1:y2, x1:x2] = m_roi_neg
                
                mask_full_pos = np.maximum(mask_full_pos, full_p)
                mask_full_neg = np.maximum(mask_full_neg, full_n)

            # 최신 마스크로 교체
            with self.data_lock:
                self.last_mask_pos = mask_full_pos
                self.last_mask_neg = mask_full_neg

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """프레임을 받아 모든 시각화를 수행하고 결과를 반환."""
        if frame_bgr is None:
            return frame_bgr

        # 1) YOLO 탐지
        results = self.yolo.predict(source=frame_bgr, imgsz=self.imgsz, verbose=False)
        
        # 박스 정보 추출
        _, boxes = draw_boxes(frame_bgr.copy(), results, conf_thres=self.conf_thres, names=self.names)
        processed_frame = frame_bgr.copy()

        # 2) 최신 LIME 작업 큐 업데이트
        if boxes:
            with self.data_lock:
                self.latest_job["frame"] = frame_bgr.copy()
                self.latest_job["boxes"] = boxes

        # 3) 최신 LIME 마스크 합성
        m_pos, m_neg = None, None
        with self.data_lock:
            if self.last_mask_pos is not None and self.last_mask_neg is not None:
                if self.last_mask_pos.shape[:2] == processed_frame.shape[:2]:
                    m_pos = self.last_mask_pos.copy()
                    m_neg = self.last_mask_neg.copy()
                else:
                    self.last_mask_pos = None
                    self.last_mask_neg = None
        
        if m_pos is not None and m_neg is not None:
            processed_frame = blend_dual_mask_sequential(processed_frame, m_pos, m_neg, alpha=self.lime_alpha)

        # 4) YOLO 바운딩 박스 그리기
        processed_frame, _ = draw_boxes(processed_frame, results, conf_thres=self.conf_thres, names=self.names)

        # 5) 충돌 위험도 게이지 및 경고 알림
        max_conf = boxes[0][5] if boxes else 0.0
        draw_risk_indicator(processed_frame, max_conf, self.warning_threshold)

        if max_conf >= self.warning_threshold:
            now = time.time()
            if now - self.last_warning_time > 2.0: # 2초 쿨타임
                self.last_warning_time = now
                msg = f"충돌 위험 감지: {max_conf*100:.1f}%"
                print(f"[WARNING] {msg}")
                notify("Drone Collision Warning!", msg)
                beep()

        # 6) FPS 표시
        self._calculate_fps()
        put_fps(processed_frame, self.fps)

        return processed_frame

    def _calculate_fps(self):
        self.cnt += 1
        now = time.time()
        if now - self.t0 >= 0.5:
            self.fps = self.cnt / (now - self.t0)
            self.t0, self.cnt = now, 0

# 독립 실행 테스트 코드 (웹캠 사용)
if __name__ == "__main__":
    print("[INFO] Running data_explain.py in test mode (Webcam).")
    # 테스트 모드에서 LIME 샘플 수 설정 가능 (예: 150으로 설정)
    detector = CollisionDetectorLIME(weights_path=None, imgsz=640, warning_threshold=0.7, lime_samples=200)
    detector.start_worker()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam not found.")
        exit()
        
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            processed_frame = detector.process_frame(frame)
            cv2.imshow("Webcam Test Mode - Integrated Visualization (Red: Risk, Green: Safety)", processed_frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27): break
    finally:
        detector.stop_worker()
        cap.release()
        cv2.destroyAllWindows()