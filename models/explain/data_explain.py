# models/explain/data_explain.py
# Webcam + YOLO(실시간) + LIME(비동기 ROI 저해상도, 즉시 반영)
# - 스냅샷/저장 없음: 생성 즉시 화면 합성
# - 최고 conf 박스(top-k)만 축소 ROI에서 LIME → 마스크 합성
# - 최신 프레임 도착 시 이전 LIME 작업 폐기(최신성 우선)

import os
import time
import argparse
import subprocess
from typing import Tuple, List, Optional
from threading import Thread, Lock, Event

import cv2
import numpy as np
from ultralytics import YOLO

# LIME (pip install lime scikit-image)
from lime import lime_image
from skimage.segmentation import slic

# ---------- 알림(선택) ----------
try:
    from plyer import notification

    def notify(title: str, message: str):
        try:
            notification.notify(title=title, message=message, timeout=2)
        except Exception:
            pass
except Exception:
    def notify(title: str, message: str):
        pass


def beep():
    """macOS 시스템 사운드(실패 시 무시)."""
    try:
        subprocess.Popen(["afplay", "/System/Library/Sounds/Glass.aiff"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

# ---------- 유틸 ----------


def open_cam(idx=0, width=1280, height=720):
    """AVFoundation 백엔드로 카메라 열기 + 저지연 설정."""
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    return cap


def put_fps(frame, fps):
    """좌상단 FPS 표시."""
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def draw_boxes(frame, results, conf_thres=0.35, names=None):
    """YOLO 박스를 그리며 (x1,y1,x2,y2,cls,conf) 리스트 반환."""
    if not results:
        return frame, []
    r0 = results[0]
    if getattr(r0, "boxes", None) is None:
        return frame, []
    boxes = []
    for b in r0.boxes:
        if b.conf is None or b.xyxy is None:
            continue
        conf = float(b.conf[0])
        if conf < conf_thres:
            continue
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
        cls = int(b.cls[0]) if b.cls is not None else -1
        label = names[cls] if names and 0 <= cls < len(names) else str(cls)
        label = f"{label} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(20, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        boxes.append((x1, y1, x2, y2, cls, conf))
    return frame, boxes


def blend_mask_color(frame_bgr: np.ndarray, mask01: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """0~1 마스크를 붉은색으로 부드럽게 합성."""
    h, w = frame_bgr.shape[:2]
    if mask01 is None or mask01.shape != (h, w):
        return frame_bgr
    m = cv2.GaussianBlur(mask01, (0, 0), 1.0)
    m3 = cv2.merge([m, m, m])
    red = np.zeros_like(frame_bgr)
    red[:, :, 2] = 255
    out = (frame_bgr.astype(np.float32) * (1.0 - alpha*m3)
           + red.astype(np.float32) * (alpha*m3)).astype(np.uint8)
    return out

# ---------- LIME 핵심 ----------


def make_predict_fn_for_roi(model: YOLO, class_id: int):
    """ROI에서 타깃 클래스 최대 conf를 [neg, pos]로 반환하는 의사 분류기."""
    def predict_proba(batch_rgb: np.ndarray) -> np.ndarray:
        probs = []
        for rgb in batch_rgb:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            # 작은 입력으로 지연 감소
            res = model.predict(source=bgr, verbose=False, imgsz=320)[0]
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


def lime_mask_on_roi(roi_bgr: np.ndarray,
                     model: YOLO,
                     class_id: int,
                     n_segments: int = 70,
                     num_samples: int = 90,
                     num_features: int = 6,
                     compactness: float = 10.0) -> np.ndarray:
    """ROI(BGR)에서 LIME 긍정 마스크(0~1) 생성. 실시간성을 위해 샘플/세그먼트 축소."""
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    def segmenter(img): return slic(img, n_segments=n_segments, compactness=compactness,
                                    sigma=1, start_label=0)
    explainer = lime_image.LimeImageExplainer()
    predict_fn = make_predict_fn_for_roi(model, class_id)

    explanation = explainer.explain_instance(
        roi_rgb,
        classifier_fn=predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=num_samples,
        segmentation_fn=segmenter
    )
    label = explanation.top_labels[0]
    _, mask = explanation.get_image_and_mask(
        label, positive_only=True, num_features=num_features, hide_rest=False
    )
    return (mask > 0).astype(np.float32)

# ---------- 메인(스트리밍 LIME) ----------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=512, help="YOLO 입력 크기(작게)")
    ap.add_argument("--conf", type=float, default=0.35, help="탐지 conf 임계값")
    ap.add_argument("--min_conf_for_lime", type=float,
                    default=0.50, help="LIME 최소 conf")
    ap.add_argument("--lime_alpha", type=float, default=0.6, help="오버레이 강도")
    ap.add_argument("--roi_shrink", type=int,
                    default=192, help="ROI 축소 크기(지연 핵심)")
    ap.add_argument("--show_fps", action="store_true")
    ap.add_argument("--topk", type=int, default=1,
                    help="LIME 적용 상위 박스 수(1~3 권장)")
    args = ap.parse_args()

    # 가중치 경로 후보 탐색
    candidates = [
        "data/runs/detect/train5/weights/best.pt",
        "data/runs/detect/train/weights/best.pt",
        "data/runs/train/weights/best.pt",
        "data/best.pt",
        "best.pt",
    ]
    weights = next((c for c in candidates if os.path.exists(c)), None)
    if not weights:
        raise FileNotFoundError(
            "best.pt 가중치를 찾지 못했습니다. data/runs/.../weights/best.pt 경로 확인")

    yolo = YOLO(weights)
    try:
        names = yolo.model.names if hasattr(yolo, "model") else None
    except Exception:
        names = None

    cap = open_cam(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"웹캠을 열 수 없습니다. index={args.cam}")

    print("[INFO] Webcam started. Press 'q' or ESC to quit. (Streaming LIME)")
    cv2.namedWindow("Webcam - LIME Streaming", cv2.WINDOW_NORMAL)

    # 최신 마스크 저장소(즉시 합성용)
    last_mask_full: Optional[np.ndarray] = None
    mask_lock = Lock()

    # 비동기 LIME 워커(항상 최신 작업만 처리)
    worker_thread: Optional[Thread] = None
    cancel_event = Event()
    latest_job = {"frame": None, "boxes": None}  # 최신 작업만 유지

    def worker_loop():
        nonlocal last_mask_full
        while not cancel_event.is_set():
            job_frame = None
            job_boxes = None
            # 최신 작업 스냅샷(가져오면 즉시 비움)
            if latest_job["frame"] is not None and latest_job["boxes"]:
                job_frame = latest_job["frame"]
                job_boxes = latest_job["boxes"]
                latest_job["frame"] = None
                latest_job["boxes"] = None
            else:
                time.sleep(0.005)
                continue

            H, W = job_frame.shape[:2]
            # top-k만 처리(최신성)
            sel = sorted(job_boxes, key=lambda b: b[5], reverse=True)[
                :max(1, args.topk)]

            # 박스별 LIME 마스크 → 화면 크기 마스크로 합성
            mask_full = np.zeros((H, W), np.float32)
            for (x1, y1, x2, y2, cls, conf) in sel:
                if cancel_event.is_set():
                    return
                if conf < args.min_conf_for_lime:
                    continue
                # ROI 추출 + 축소
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W-1, x2), min(H-1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                roi = job_frame[y1:y2, x1:x2].copy()
                roi_small = cv2.resize(
                    roi, (args.roi_shrink, args.roi_shrink), interpolation=cv2.INTER_AREA)

                # LIME → ROI 마스크(0~1)
                m_small = lime_mask_on_roi(roi_small, yolo, cls,
                                           n_segments=70, num_samples=90, num_features=6, compactness=10.0)
                m_roi = cv2.resize(
                    m_small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)
                full = np.zeros((H, W), np.float32)
                full[y1:y2, x1:x2] = m_roi
                mask_full = np.maximum(mask_full, full)

            # 최신 마스크로 즉시 교체(EMA 등 없음)
            with mask_lock:
                last_mask_full = mask_full

    # 워커 시작
    cancel_event.clear()
    worker_thread = Thread(target=worker_loop, daemon=True)
    worker_thread.start()

    t0, cnt, fps = time.time(), 0, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            # 1) YOLO 탐지(저해상도 입력)
            results = yolo.predict(
                source=frame, imgsz=args.imgsz, verbose=False)
            frame, boxes = draw_boxes(
                frame, results, conf_thres=args.conf, names=names)

            # 2) 최신 작업 큐 덮어쓰기(미완료 작업은 자연 폐기)
            if boxes:
                latest_job["frame"] = frame.copy()
                latest_job["boxes"] = boxes

            # 3) 최신 마스크 즉시 합성
            with mask_lock:
                m = None if last_mask_full is None else last_mask_full.copy()
            if m is not None:
                frame = blend_mask_color(frame, m, alpha=args.lime_alpha)

            # 경고/비프(예: 최고 conf ≥ 0.7)
            if boxes and boxes[0][5] >= 0.7:
                notify("Drone Collision Warning!",
                       f"충돌 위험 감지: {boxes[0][5]*100:.1f}%")
                beep()

            # FPS 표시(옵션)
            if args.show_fps:
                cnt += 1
                now = time.time()
                if now - t0 >= 0.5:
                    fps = cnt / (now - t0)
                    t0, cnt = now, 0
                put_fps(frame, fps)

            cv2.imshow("Webcam - LIME Streaming", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                break

    finally:
        cancel_event.set()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
