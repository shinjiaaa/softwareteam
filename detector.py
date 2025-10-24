# detector.py
import os
import time
from typing import Optional, Tuple, Dict, Any
from threading import Thread, Lock, Event

import cv2
import numpy as np
from ultralytics import YOLO

from lime import lime_image
from skimage.segmentation import slic

# ==============================================================================
#  Visualization Utilities (OpenCV - Server Side Rendering)
# ==============================================================================
# (시각화 함수들은 이전과 동일합니다. 지면상 주요 함수만 표기)


def draw_risk_indicator(frame, max_conf, warning_threshold):
    # (이전과 동일)
    h, w = frame.shape[:2]
    bar_height = 20
    cv2.rectangle(frame, (0, 0), (w, bar_height), (50, 50, 50), -1)
    risk_width = int(w * max_conf)
    if max_conf < 0.5:
        r, g = int(255 * (max_conf * 2)), 255
    else:
        r, g = 255, int(255 * (1 - (max_conf - 0.5) * 2))
    color = (0, g, r)
    if risk_width > 0:
        cv2.rectangle(frame, (0, 0), (risk_width, bar_height), color, -1)
    thresh_x = int(w * warning_threshold)
    if 0 <= thresh_x < w:
        cv2.line(frame, (thresh_x, 0), (thresh_x, bar_height), (255, 255, 255), 2)
    text = f"RISK LEVEL: {max_conf*100:.1f}%"
    text_color = (255, 255, 255) if max_conf < 0.6 else (10, 10, 10)
    cv2.putText(
        frame,
        text,
        (10, bar_height - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        text_color,
        1,
        cv2.LINE_AA,
    )


def draw_boxes(frame, results, conf_thres=0.35, names=None):
    # (이전과 동일)
    if not results or getattr(results[0], "boxes", None) is None:
        return frame, []
    sorted_boxes = sorted(
        results[0].boxes,
        key=lambda b: float(b.conf[0]) if b.conf is not None else 0.0,
        reverse=True,
    )
    boxes_info = []
    for b in sorted_boxes:
        if b.conf is None or b.xyxy is None:
            continue
        conf = float(b.conf[0])
        if conf < conf_thres:
            break
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
        cls = int(b.cls[0]) if b.cls is not None else -1
        color = (0, int(255 * (1 - conf)), int(255 * conf))
        label = names[cls] if names and 0 <= cls < len(names) else str(cls)
        label = f"{label} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (w_text, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_y = max(y1, h + 30)

        # 바운딩 박스 텍스트는 서버 렌더링으로 유지 (가독성 향상 위해 배경 제거)
        cv2.putText(
            frame,
            label,
            (x1, text_y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
        boxes_info.append((x1, y1, x2, y2, cls, conf))
    return frame, boxes_info


def _blend_single(
    bg: np.ndarray, fg_color_bgr: Tuple[int, int, int], mask: np.ndarray, alpha: float
) -> np.ndarray:
    # (이전과 동일)
    if mask is None or np.max(mask) == 0:
        return bg
    m = cv2.GaussianBlur(mask, (0, 0), 2.5)
    m3 = cv2.merge([m, m, m])
    fg = np.zeros_like(bg)
    fg[:] = fg_color_bgr
    out = bg.astype(np.float32) * (1.0 - alpha * m3) + fg.astype(np.float32) * (
        alpha * m3
    )
    return np.clip(out, 0, 255).astype(np.uint8)


def blend_dual_mask_sequential(
    frame_bgr: np.ndarray,
    pos_mask01: np.ndarray,
    neg_mask01: np.ndarray,
    alpha: float = 0.65,
) -> np.ndarray:
    # (이전과 동일)
    h, w = frame_bgr.shape[:2]
    if pos_mask01 is None or neg_mask01 is None or pos_mask01.shape != (h, w):
        return frame_bgr
    COLOR_GREEN = (0, 255, 0)
    COLOR_RED = (0, 0, 255)
    out = _blend_single(frame_bgr, COLOR_GREEN, neg_mask01, alpha * 0.9)
    out = _blend_single(out, COLOR_RED, pos_mask01, alpha)
    return out


# ==============================================================================
#  LIME Core Functions
# ==============================================================================
# (LIME 핵심 함수들은 이전과 동일하므로 생략: make_predict_fn_for_roi, lime_mask_on_roi_weighted)


def make_predict_fn_for_roi(model: YOLO, class_id: int):
    def predict_proba(batch_rgb: np.ndarray) -> np.ndarray:
        bgr_batch = [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) for rgb in batch_rgb]
        try:
            results = model.predict(source=bgr_batch, verbose=False, imgsz=320)
        except Exception as e:
            print(f"[WARN] YOLO batch prediction error: {e}")
            return np.array([[1.0, 0.0]] * len(batch_rgb), dtype=np.float32)
        probs = []
        for res in results:
            score = 0.0
            if getattr(res, "boxes", None) is not None:
                for bx in res.boxes:
                    if bx.conf is None or bx.cls is None:
                        continue
                    if int(bx.cls[0]) == class_id:
                        score = max(score, float(bx.conf[0]))
            pos = float(np.clip(score, 0.0, 1.0))
            probs.append([1.0 - pos, pos])
        return np.array(probs, dtype=np.float32)

    return predict_proba


def lime_mask_on_roi_weighted(
    roi_bgr: np.ndarray,
    model: YOLO,
    class_id: int,
    num_samples: int,
    n_segments=70,
    num_features=10,
    compactness=10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    h, w = roi_bgr.shape[:2]

    def segmenter(img):
        return slic(
            img, n_segments=n_segments, compactness=compactness, sigma=1, start_label=0
        )

    explainer = lime_image.LimeImageExplainer()
    predict_fn = make_predict_fn_for_roi(model, class_id)
    try:
        explanation = explainer.explain_instance(
            roi_rgb,
            classifier_fn=predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=num_samples,
            segmentation_fn=segmenter,
        )
        label = explanation.top_labels[0]
        segments = explanation.segments
        local_exp = explanation.local_exp[label]
        sorted_exp = sorted(local_exp, key=lambda item: abs(item[1]), reverse=True)[
            :num_features
        ]
        pos_mask = np.zeros((h, w), dtype=np.float32)
        neg_mask = np.zeros((h, w), dtype=np.float32)
        if not sorted_exp:
            return pos_mask, neg_mask
        for segment_id, weight in sorted_exp:
            mask_area = segments == segment_id
            if weight > 0:
                pos_mask[mask_area] = weight
            elif weight < 0:
                neg_mask[mask_area] = abs(weight)
        max_weight = max(abs(w) for _, w in sorted_exp)
        if max_weight > 0:
            pos_mask = np.clip(pos_mask / max_weight, 0.0, 1.0)
            neg_mask = np.clip(neg_mask / max_weight, 0.0, 1.0)
        return pos_mask, neg_mask
    except Exception as e:
        return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)


# ==============================================================================
#  CollisionDetectorLIME Class
# ==============================================================================


class CollisionDetectorLIME:
    def __init__(self, weights_path: Optional[str] = None):
        # (초기화는 이전과 동일)
        self.config = {
            "imgsz": 512,
            "conf_thres": 0.4,
            "min_conf_for_lime": 0.6,
            "warning_threshold": 0.75,
            "roi_shrink": 192,
            "topk": 1,
            "lime_samples": 150,
            "lime_alpha": 0.65,
        }
        self.config_lock = Lock()
        self.weights = self._find_weights(weights_path)
        print(f"[Detector] Loading YOLO model from: {self.weights}")
        self.yolo = YOLO(self.weights)
        self.names = getattr(self.yolo.model, "names", None)
        self.last_mask_pos: Optional[np.ndarray] = None
        self.last_mask_neg: Optional[np.ndarray] = None
        self.data_lock = Lock()
        self.cancel_event = Event()
        self.latest_job = {"frame": None, "boxes": None}
        self.worker_thread: Optional[Thread] = None
        self.t0, self.cnt, self.fps = time.time(), 0, 0.0
        self.last_alert_time = 0

    def _find_weights(self, path):
        if path and os.path.exists(path):
            return path
        candidates = ["best.pt", "yolov8n.pt"]
        return next((c for c in candidates if os.path.exists(c)), candidates[-1])

    def get_config(self) -> Dict[str, Any]:
        with self.config_lock:
            return self.config.copy()

    # (워커 스레드 관리 함수들은 이전과 동일하므로 생략: start_worker, stop_worker, _worker_loop)
    def start_worker(self):
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.cancel_event.clear()
            self.worker_thread = Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()

    def stop_worker(self):
        self.cancel_event.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=3.0)

    def _worker_loop(self):
        while not self.cancel_event.is_set():
            job_frame, job_boxes = None, None
            with self.data_lock:
                if self.latest_job["frame"] is not None and self.latest_job["boxes"]:
                    job_frame = self.latest_job["frame"]
                    job_boxes = self.latest_job["boxes"]
                    self.latest_job["frame"] = None
                    self.latest_job["boxes"] = None
            if job_frame is None:
                time.sleep(0.01)
                continue
            cfg = self.get_config()
            H, W = job_frame.shape[:2]
            sel = job_boxes[: cfg["topk"]]
            mask_full_pos = np.zeros((H, W), np.float32)
            mask_full_neg = np.zeros((H, W), np.float32)
            for x1, y1, x2, y2, cls, conf in sel:
                if self.cancel_event.is_set():
                    return
                if conf < cfg["min_conf_for_lime"]:
                    continue
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                roi = job_frame[y1:y2, x1:x2]
                try:
                    roi_small = cv2.resize(
                        roi,
                        (cfg["roi_shrink"], cfg["roi_shrink"]),
                        interpolation=cv2.INTER_AREA,
                    )
                except cv2.error:
                    continue
                m_small_pos, m_small_neg = lime_mask_on_roi_weighted(
                    roi_small,
                    self.yolo,
                    cls,
                    num_samples=cfg["lime_samples"],
                    num_features=10,
                )
                m_roi_pos = cv2.resize(
                    m_small_pos,
                    (roi.shape[1], roi.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                m_roi_neg = cv2.resize(
                    m_small_neg,
                    (roi.shape[1], roi.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                full_p = np.zeros((H, W), np.float32)
                full_n = np.zeros((H, W), np.float32)
                full_p[y1:y2, x1:x2] = m_roi_pos
                full_n[y1:y2, x1:x2] = m_roi_neg
                mask_full_pos = np.maximum(mask_full_pos, full_p)
                mask_full_neg = np.maximum(mask_full_neg, full_n)
            with self.data_lock:
                self.last_mask_pos = mask_full_pos
                self.last_mask_neg = mask_full_neg

    # --- 위험도 평가 및 알림 (로직 검증 및 로깅 추가) ---
    def _evaluate_risk(self, max_conf: float) -> Dict[str, Any]:
        alert_event = None

        # 요구사항에 따른 위험 단계 정의 (Thresholds: 0.5, 0.6, 0.8)
        if max_conf >= 0.80:
            level, text = "danger", "위험"
            sound, tts = "alert_high_repeat", "충돌 위험"
        elif max_conf >= 0.60:
            level, text = "warning", "경고"
            sound, tts = "alert_mid_2", "경고"
        elif max_conf >= 0.50:
            level, text = "caution", "주의"
            sound, tts = "alert_low_1", "주의"
        else:
            # Conf < 0.50
            level, text = "safe", "안전"
            sound, tts = None, None

        # 쿨타임 기반 알림 발생 (2초)
        now = time.time()
        # 위험 상태 조건 (정의된 레벨 기준)
        is_risky = level != "safe"

        if is_risky and now - self.last_alert_time > 2.0:
            self.last_alert_time = now
            alert_event = {
                "level": level,
                "message": f"충돌 위험 감지: {max_conf*100:.1f}%",
                "sound": sound,
                "tts": tts,
            }
            # [추가됨] 백엔드 디버깅 로그 (위험 감지 확인용)
            print(
                f"[Detector DEBUG] Risk Detected: {level} ({max_conf*100:.1f}%). Alert event generated."
            )

        return {
            "max_conf": max_conf,
            "level": level,
            "text": text,
            "alert_event": alert_event,
        }

    # --- 메인 처리 함수 ---
    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        if frame_bgr is None:
            return frame_bgr, self._evaluate_risk(0.0)
        cfg = self.get_config()
        results = self.yolo.predict(source=frame_bgr, imgsz=cfg["imgsz"], verbose=False)
        _, boxes = draw_boxes(
            frame_bgr.copy(), results, conf_thres=cfg["conf_thres"], names=self.names
        )
        processed_frame = frame_bgr.copy()
        if boxes:
            with self.data_lock:
                self.latest_job["frame"] = frame_bgr.copy()
                self.latest_job["boxes"] = boxes
        m_pos, m_neg = None, None

        with self.data_lock:
            if self.last_mask_pos is not None and self.last_mask_neg is not None:
                if self.last_mask_pos.shape[:2] == processed_frame.shape[:2]:
                    m_pos = self.last_mask_pos.copy()
                    m_neg = self.last_mask_neg.copy()

        if m_pos is not None and m_neg is not None:
            processed_frame = blend_dual_mask_sequential(
                processed_frame, m_pos, m_neg, alpha=cfg["lime_alpha"]
            )
        processed_frame, _ = draw_boxes(
            processed_frame, results, conf_thres=cfg["conf_thres"], names=self.names
        )
        max_conf = boxes[0][5] if boxes else 0.0
        # 위험도 평가 (핵심 로직)
        risk_data = self._evaluate_risk(max_conf)
        draw_risk_indicator(processed_frame, max_conf, cfg["warning_threshold"])
        self._calculate_fps()
        return processed_frame, risk_data

    def _calculate_fps(self):
        self.cnt += 1
        now = time.time()
        if now - self.t0 >= 0.5:
            self.fps = self.cnt / (now - self.t0)
            self.t0, self.cnt = now, 0