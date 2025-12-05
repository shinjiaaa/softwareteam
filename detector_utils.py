import os
import time
import cv2
import numpy as np
from threading import Thread, Lock, Event
from typing import Optional, Dict, Any, Tuple
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import json

from detector import (
    estimate_distance,
    draw_risk_indicator,
    draw_boxes,
    blend_red_mask,
    make_predict_fn_for_roi,
    lime_mask_on_roi_weighted,
)

DEFAULT_YOLO = "models/best.pt"
DEFAULT_COLLISION = "models/model_weights.h5"


class CollisionDetectorLIME:
    def __init__(
        self,
        weights_path: Optional[str] = None,
        collision_model_path: Optional[str] = None,
    ):
        self.config = {
            "imgsz": 320,
            "conf_thres": 0.35,
            "min_conf_for_lime": 0.5,  # 위험률 50% 이상일 때만 픽셀 반환
            "warning_threshold": 0.75,
            "roi_shrink": 96,
            "topk": 1,
            "lime_samples": 100,
            "lime_alpha": 0.65,
        }
        self.config_lock = Lock()

        # 모델 가중치 찾기
        if weights_path and os.path.exists(weights_path):
            self.weights = weights_path
        elif os.path.exists(DEFAULT_YOLO):
            self.weights = DEFAULT_YOLO
        else:
            self.weights = self._find_weights(weights_path)
        print(f"[Detector] Loading YOLO model from: {self.weights}")
        self.yolo = YOLO(self.weights)
        self.names = getattr(self.yolo.model, "names", None)

        # 충돌 분류 모델 로드
        self.collision_model = None
        if collision_model_path and os.path.exists(collision_model_path):
            try:
                print(
                    f"[Detector] Loading collision model from: {collision_model_path}"
                )
                self.collision_model = load_model(collision_model_path)
            except Exception as e:
                print(f"[Detector] Failed to load collision model: {e}")
                self.collision_model = None
        elif os.path.exists(DEFAULT_COLLISION):
            try:
                print(
                    f"[Detector] Loading collision model from default: {DEFAULT_COLLISION}"
                )
                self.collision_model = load_model(DEFAULT_COLLISION)
            except Exception as e:
                print(f"[Detector] Failed to load default collision model: {e}")
                self.collision_model = None

        # LIME 결과
        self.last_mask_pos: Optional[np.ndarray] = None
        self.last_mask_neg: Optional[np.ndarray] = None
        self.last_lime_json_time: float = 0.0
        self.frame_count: int = 0

        # 스레드
        self.data_lock = Lock()
        self.cancel_event = Event()
        self.latest_job = {"frame": None, "boxes": None}
        self.worker_thread: Optional[Thread] = None

        # FPS, alert
        self.t0, self.cnt, self.fps = time.time(), 0, 0.0
        self.last_alert_time = 0

    def _find_weights(self, path):
        candidates = ["best.pt", "yolo11n.pt"]
        return next((c for c in candidates if os.path.exists(c)), candidates[-1])

    def get_config(self) -> Dict[str, Any]:
        with self.config_lock:
            return self.config.copy()

    def start_worker(self):
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.cancel_event.clear()
            self.worker_thread = Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()

    def stop_worker(self):
        self.cancel_event.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=3.0)

    # LIME 연산 (백그라운드에서 진행)
    def _worker_loop(self):
        frame_count = 0
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

            frame_count += 1
            N = 5
            if frame_count % N != 0:
                time.sleep(0.01)
                continue

            cfg = self.get_config()
            H, W = job_frame.shape[:2]

            sel = job_boxes[: cfg["topk"]]
            if not sel:
                time.sleep(0.02)
                continue

            mask_full_pos = np.zeros((H, W), np.float32)
            for x1, y1, x2, y2, cls, conf in sel:
                if self.cancel_event.is_set():
                    return
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
                    num_features=3,
                )

                m_roi_pos = cv2.resize(
                    m_small_pos,
                    (roi.shape[1], roi.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                full_p = np.zeros((H, W), np.float32)
                full_p[y1:y2, x1:x2] = m_roi_pos
                mask_full_pos = np.maximum(mask_full_pos, full_p)

            with self.data_lock:
                self.last_mask_pos = mask_full_pos
                self.last_mask_neg = np.zeros_like(mask_full_pos)

    # 위험도 평가
    def _evaluate_risk(self, max_conf: float) -> Dict[str, Any]:
        alert_event = None
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
            level, text = "safe", "안전"
            sound, tts = None, None

        now = time.time()
        is_risky = level != "safe"
        if is_risky and now - self.last_alert_time > 2.0:
            self.last_alert_time = now
            alert_event = {
                "level": level,
                "message": f"충돌 위험 감지: {max_conf*100:.1f}%",
                "sound": sound,
                "tts": tts,
            }
            print(
                f"[Detector DEBUG] Risk Detected: {level} ({max_conf*100:.1f}%). Alert event generated."
            )
        return {
            "max_conf": max_conf,
            "level": level,
            "text": text,
            "alert_event": alert_event,
        }

    # 객체 탐지 모델 -> 충돌 분류 모델 -> LIME
    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        if frame_bgr is None:
            return frame_bgr, self._evaluate_risk(0.0)

        cfg = self.get_config()
        results = self.yolo.predict(source=frame_bgr, imgsz=cfg["imgsz"], verbose=False)

        # bbox 그리기
        processed_frame, boxes = draw_boxes(
            frame_bgr.copy(), results, cfg["conf_thres"], self.names
        )

        # 가장 가까운 객체 선택
        min_distance = float("inf")
        closest_box = None
        for box in boxes:
            if len(box) < 6:
                continue
            x1, y1, x2, y2, cls, conf = box
            distance = estimate_distance(box, cls)
            if distance < min_distance:
                min_distance = distance
                closest_box = box

        # 충돌 분류 모델로 ROI 예측
        collision_prob = 0.0

        if closest_box is not None and self.collision_model is not None:
            x1, y1, x2, y2, cls, conf = closest_box
            x1, y1, x2, y2 = max(0, x1), max(0, y1), max(0, x2), max(0, y2)
            roi = frame_bgr[y1:y2, x1:x2]
            if roi.size != 0:
                try:
                    # 모델 입력
                    roi_resized = cv2.resize(roi, (128, 128))
                    roi_input = (roi_resized.astype(np.float32) / 255.0)[
                        np.newaxis, ...
                    ]

                    # 충돌 확률 예측
                    pred = self.collision_model.predict(roi_input, verbose=0)[0]
                    if hasattr(pred, "__len__") and len(pred) >= 2:
                        collision_prob = float(pred[1])
                    else:
                        collision_prob = float(pred[0])

                    # 거리 기반 위험도 보정
                    if distance > 0:
                        distance_factor = np.exp(
                            -distance / 10.0
                        )  # 거리 10m 기준으로 감쇠
                        collision_prob *= 0.5 + 0.5 * distance_factor
                        collision_prob = np.clip(collision_prob, 0.0, 1.0)

                except Exception as e:
                    print(f"[Detector WARN] collision model predict error: {e}")
                    collision_prob = 0.0

        # 위험도 산출
        max_conf = collision_prob if self.collision_model is not None else 0.0

        # LIME에 전달 - 위험 상태(>= min_conf_for_lime)일 때만
        if boxes and max_conf >= cfg["min_conf_for_lime"]:
            with self.data_lock:
                # topk 박스 전달 - 가장 높은 conf부터 topk
                self.latest_job["frame"] = frame_bgr.copy()
                self.latest_job["boxes"] = boxes
        else:
            # 안전 상태일 시 LIME 결과 삭제
            with self.data_lock:
                self.last_mask_pos = None
                self.last_mask_neg = None
                self.latest_job["frame"] = None
                self.latest_job["boxes"] = None

        # LIME 마스크 적용
        m_pos = None
        with self.data_lock:
            if (
                self.last_mask_pos is not None
                and self.last_mask_pos.shape[:2] == processed_frame.shape[:2]
            ):
                m_pos = self.last_mask_pos.copy()
        if m_pos is not None and max_conf >= cfg["min_conf_for_lime"]:
            processed_frame = blend_red_mask(
                processed_frame, m_pos, alpha=cfg["lime_alpha"]
            )

        # 위험도 평가 & UI 바
        risk_data = self._evaluate_risk(max_conf)
        draw_risk_indicator(processed_frame, max_conf, cfg["warning_threshold"])
        self._calculate_fps()

        # LIME 결과를 LLM으로 설명 생성
        now = time.time()
        JSON_COOLDOWN = 5.0  # 최소 5초마다 1번만 생성
        if (
            risk_data["alert_event"]
            and self.last_mask_pos is not None
            and now - self.last_lime_json_time >= JSON_COOLDOWN
        ):
            try:
                class_name = (
                    self.names[closest_box[4]]
                    if closest_box
                    and self.names
                    and 0 <= closest_box[4] < len(self.names)
                    else "unknown"
                )
                explanation_json = generate_lime_explanation(
                    self.last_mask_pos,
                    self.last_mask_neg,
                    class_name,
                    max_conf,
                )
                print(
                    "[LIME-EXPLANATION]",
                    json.dumps(explanation_json, ensure_ascii=False),
                )
                self.last_lime_json_time = now
            except Exception as e:
                print(f"[LIME-EXPLANATION ERROR] {e}")

        return processed_frame, risk_data

    def _calculate_fps(self):
        self.cnt += 1
        now = time.time()
        if now - self.t0 >= 0.5:
            self.fps = self.cnt / (now - self.t0)
            self.t0, self.cnt = now, 0
