import cv2
import numpy as np
from lime import lime_image
from skimage.segmentation import slic
from typing import Tuple
from ultralytics import YOLO

CLASS_HEIGHTS = {0: 1.5, 1: 5.0, 2: 10.0, 3: 1.7, 4: 0.5}
FOCAL_LENGTH_PIXELS = 1400

# 모델 경로
DEFAULT_YOLO = "models/best.pt"  # 객체 탐지 모델
DEFAULT_COLLISION = "models/model_weights.h5"  # 충돌 분류 모델


# 유틸 함수들
def estimate_distance(box, class_id):
    # box: (x1,y1,x2,y2,cls,conf) 또는 이 형식에서 사용
    y1, y2 = box[1], box[3]
    h_pixels = max(y2 - y1, 1)
    H_actual = CLASS_HEIGHTS.get(class_id, 1.7)
    return (H_actual * FOCAL_LENGTH_PIXELS) / h_pixels


def draw_risk_indicator(frame, max_conf, warning_threshold):
    h, w = frame.shape[:2]
    bar_h = 20
    cv2.rectangle(frame, (0, 0), (w, bar_h), (50, 50, 50), -1)
    risk_w = int(w * max_conf)
    if max_conf < 0.5:
        r, g = int(255 * (max_conf * 2)), 255
    else:
        r, g = 255, int(255 * (1 - (max_conf - 0.5) * 2))
    color = (0, g, r)
    if risk_w > 0:
        cv2.rectangle(frame, (0, 0), (risk_w, bar_h), color, -1)
    tx = int(w * warning_threshold)
    if 0 <= tx < w:
        cv2.line(frame, (tx, 0), (tx, bar_h), (255, 255, 255), 2)
    text = f"RISK LEVEL: {max_conf*100:.1f}%"
    tc = (255, 255, 255) if max_conf < 0.6 else (10, 10, 10)
    cv2.putText(
        frame, text, (10, bar_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tc, 1, cv2.LINE_AA
    )


def draw_boxes(frame, results, conf_thres=0.35, names=None):
    if not results or getattr(results[0], "boxes", None) is None:
        return frame, []
    sorted_boxes = sorted(
        results[0].boxes,
        key=lambda b: float(b.conf[0]) if getattr(b, "conf", None) is not None else 0.0,
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


# 마스크 블렌드 - 빨간색(=위험)만
def _blend_single(
    bg: np.ndarray, fg_color_bgr: Tuple[int, int, int], mask: np.ndarray, alpha: float
) -> np.ndarray:
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
    h, w = frame_bgr.shape[:2]
    if pos_mask01 is None or pos_mask01.shape != (h, w):
        return frame_bgr
    COLOR_RED = (0, 0, 255)
    return _blend_single(frame_bgr, COLOR_RED, pos_mask01, alpha)


# LIME - YOLO 기반 예측 wrapper (ROI에 대해 신뢰도 반환)
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


# LIME 마스크 생성 (상위 3개 슈퍼픽셀만 반환)
def lime_mask_on_roi_weighted(
    roi_bgr: np.ndarray,
    model: YOLO,
    class_id: int,
    num_samples: int,
    n_segments=70,
    num_features=2,
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
        # 상위 3개 슈퍼픽셀만 선택 (절댓값 기준)
        sorted_exp = sorted(local_exp, key=lambda item: abs(item[1]), reverse=True)[:3]
        pos_mask = np.zeros((h, w), dtype=np.float32)
        neg_mask = np.zeros((h, w), dtype=np.float32)
        if not sorted_exp:
            return pos_mask, neg_mask
        for seg_id, weight in sorted_exp:
            mask_area = segments == seg_id
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