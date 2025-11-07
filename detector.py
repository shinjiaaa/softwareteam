import os, time, cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
from threading import Thread, Lock, Event
from ultralytics import YOLO
from lime import lime_image
from skimage.segmentation import slic

# 클래스별 평균 높이 (m)
CLASS_HEIGHTS = {0:1.5, 1:5.0, 2:10.0, 3:1.7, 4:0.5}
FOCAL_LENGTH_PIXELS = 1400

def estimate_distance(box, class_id):
    y1, y2 = box[1], box[3]
    h_pixels = max(y2 - y1, 1)
    H_actual = CLASS_HEIGHTS.get(class_id, 1.7)
    return (H_actual * FOCAL_LENGTH_PIXELS) / h_pixels

def draw_risk_indicator(frame, max_conf, warning_threshold):
    h, w = frame.shape[:2]
    bar_height = 20
    cv2.rectangle(frame, (0,0), (w, bar_height), (50,50,50), -1)
    risk_width = int(w * max_conf)
    if max_conf < 0.5: r,g = int(255*(max_conf*2)),255
    else: r,g = 255,int(255*(1-(max_conf-0.5)*2))
    color = (0,g,r)
    if risk_width>0: cv2.rectangle(frame,(0,0),(risk_width,bar_height),color,-1)
    thresh_x = int(w*warning_threshold)
    if 0<=thresh_x<w: cv2.line(frame,(thresh_x,0),(thresh_x,bar_height),(255,255,255),2)
    text = f"RISK LEVEL: {max_conf*100:.1f}%"
    text_color = (255,255,255) if max_conf<0.6 else (10,10,10)
    cv2.putText(frame, text, (10,bar_height-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)

def draw_boxes(frame, results, conf_thres=0.35, names=None):
    if not results or getattr(results[0], "boxes", None) is None: return frame,[]
    sorted_boxes = sorted(results[0].boxes, key=lambda b: float(b.conf[0]) if b.conf is not None else 0.0, reverse=True)
    boxes_info=[]
    for b in sorted_boxes:
        if b.conf is None or b.xyxy is None: continue
        conf = float(b.conf[0])
        if conf<conf_thres: break
        x1,y1,x2,y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
        cls = int(b.cls[0]) if b.cls is not None else -1
        color = (0,int(255*(1-conf)),int(255*conf))
        label = names[cls] if names and 0<=cls<len(names) else str(cls)
        label=f"{label} {conf:.2f}"
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        (w_text,h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6,2)
        text_y = max(y1,h+30)
        cv2.putText(frame,label,(x1,text_y-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2,cv2.LINE_AA)
        boxes_info.append((x1,y1,x2,y2,cls,conf))
    return frame, boxes_info

# 단일/dual mask blend
def _blend_single(bg, fg_color_bgr, mask, alpha):
    if mask is None or np.max(mask)==0: return bg
    m = cv2.GaussianBlur(mask,(0,0),2.5); m3=cv2.merge([m,m,m])
    fg=np.zeros_like(bg); fg[:]=fg_color_bgr
    out=(bg.astype(np.float32)*(1.0-alpha*m3)+fg.astype(np.float32)*(alpha*m3))
    return np.clip(out,0,255).astype(np.uint8)

def blend_dual_mask_sequential(frame_bgr,pos_mask01,neg_mask01,alpha=0.65):
    h,w=frame_bgr.shape[:2]
    if pos_mask01 is None or neg_mask01 is None or pos_mask01.shape!=(h,w): return frame_bgr
    out=_blend_single(frame_bgr,(0,255,0),neg_mask01,alpha*0.9)
    out=_blend_single(out,(0,0,255),pos_mask01,alpha)
    return out

# LIME wrapper for collision classifier
def make_predict_fn_for_collision(model):
    def predict_proba(batch_rgb):
        batch_input=np.array([cv2.resize(img,(128,128))/255.0 for img in batch_rgb])
        probs=model.predict(batch_input)
        return np.hstack([1-probs,probs])
    return predict_proba

def lime_mask_on_roi_weighted(roi_bgr, model,num_samples=100,n_segments=70,num_features=10,compactness=10.0):
    roi_rgb=cv2.cvtColor(roi_bgr,cv2.COLOR_BGR2RGB)
    h,w=roi_bgr.shape[:2]
    def segmenter(img): return slic(img,n_segments=n_segments,compactness=compactness,sigma=1,start_label=0)
    explainer=lime_image.LimeImageExplainer()
    predict_fn=make_predict_fn_for_collision(model)
    try:
        explanation=explainer.explain_instance(roi_rgb,classifier_fn=predict_fn,top_labels=1,
                                                hide_color=0,num_samples=num_samples,segmentation_fn=segmenter)
        label=explanation.top_labels[0]; segments=explanation.segments
        local_exp=explanation.local_exp[label]
        sorted_exp=sorted(local_exp,key=lambda item: abs(item[1]),reverse=True)[:num_features]
        pos_mask=np.zeros((h,w),dtype=np.float32); neg_mask=np.zeros((h,w),dtype=np.float32)
        if not sorted_exp: return pos_mask,neg_mask
        for seg_id,weight in sorted_exp:
            mask_area=(segments==seg_id)
            if weight>0: pos_mask[mask_area]=weight
            elif weight<0: neg_mask[mask_area]=abs(weight)
        max_weight=max(abs(w) for _,w in sorted_exp)
        if max_weight>0: pos_mask=np.clip(pos_mask/max_weight,0.0,1.0); neg_mask=np.clip(neg_mask/max_weight,0.0,1.0)
        return pos_mask,neg_mask
    except: return np.zeros((h,w),dtype=np.float32), np.zeros((h,w),dtype=np.float32)

class CollisionDetectorLIME:
    def __init__(self, weights_path=None, collision_model=None):
        self.config={"imgsz":320,"conf_thres":0.35,"warning_threshold":0.75,
                     "roi_shrink":192,"topk":1,"lime_samples":100,"lime_alpha":0.65,
                     "min_conf_for_lime":0.5}
        self.config_lock=Lock()
        self.weights=self._find_weights(weights_path)
        self.yolo=YOLO(self.weights); self.names=getattr(self.yolo.model,"names",None)
        self.collision_model=collision_model
        self.last_mask_pos=None; self.last_mask_neg=None
        self.data_lock=Lock(); self.cancel_event=Event()
        self.latest_job={"frame":None,"boxes":None}; self.worker_thread=None
        self.t0,self.cnt,self.fps=time.time(),0,0.0
        self.last_alert_time=0

    def _find_weights(self,path):
        if path and os.path.exists(path): return path
        candidates=["best.pt","yolo11n.pt"]
        return next((c for c in candidates if os.path.exists(c)),candidates[-1])

    def get_config(self): 
        with self.config_lock: return self.config.copy()

    def start_worker(self):
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.cancel_event.clear()
            self.worker_thread=Thread(target=self._worker_loop,daemon=True)
            self.worker_thread.start()

    def stop_worker(self):
        self.cancel_event.set()
        if self.worker_thread: self.worker_thread.join(timeout=3.0)

    def _worker_loop(self):
        while not self.cancel_event.is_set():
            job_frame,job_boxes=None,None
            with self.data_lock:
                if self.latest_job["frame"] is not None and self.latest_job["boxes"]:
                    job_frame=self.latest_job["frame"]; job_boxes=self.latest_job["boxes"]
                    self.latest_job["frame"]=None; self.latest_job["boxes"]=None
            if job_frame is None: time.sleep(0.01); continue
            cfg=self.get_config()
            H,W=job_frame.shape[:2]; sel=job_boxes[:cfg["topk"]]
            mask_full_pos=np.zeros((H,W),np.float32); mask_full_neg=np.zeros((H,W),np.float32)
            for (x1,y1,x2,y2,cls,conf) in sel:
                if self.cancel_event.is_set(): return
                if conf<cfg["min_conf_for_lime"]: continue
                x1,y1,x2,y2=max(0,x1),max(0,y1),min(W-1,x2),min(H-1,y2)
                if x2<=x1 or y2<=y1: continue
                roi=job_frame[y1:y2,x1:x2]
                try: roi_small=cv2.resize(roi,(cfg["roi_shrink"],cfg["roi_shrink"]),interpolation=cv2.INTER_AREA)
                except cv2.error: continue
                if self.collision_model:
                    m_small_pos,m_small_neg=lime_mask_on_roi_weighted(roi_small,self.collision_model,num_samples=cfg["lime_samples"])
                    m_roi_pos=cv2.resize(m_small_pos,(roi.shape[1],roi.shape[0]),interpolation=cv2.INTER_LINEAR)
                    m_roi_neg=cv2.resize(m_small_neg,(roi.shape[1],roi.shape[0]),interpolation=cv2.INTER_LINEAR)
                    full_p=np.zeros((H,W),np.float32); full_n=np.zeros((H,W),np.float32)
                    full_p[y1:y2,x1:x2]=m_roi_pos; full_n[y1:y2,x1:x2]=m_roi_neg
                    mask_full_pos=np.maximum(mask_full_pos,full_p)
                    mask_full_neg=np.maximum(mask_full_neg,full_n)
            with self.data_lock:
                self.last_mask_pos=mask_full_pos; self.last_mask_neg=mask_full_neg

    def _evaluate_risk(self,max_conf):
        alert_event=None
        if max_conf>=0.80: level,text="danger","위험"; sound,tts="alert_high_repeat","충돌 위험"
        elif max_conf>=0.60: level,text="warning","경고"; sound,tts="alert_mid_2","경고"
        elif max_conf>=0.50: level,text="caution","주의"; sound,tts="alert_low_1","주의"
        else: level,text="safe","안전"; sound,tts=None,None
        now=time.time()
        is_risky=level!="safe"
        if is_risky and now-self.last_alert_time>2.0:
            self.last_alert_time=now
            alert_event={"level":level,"message":f"충돌 위험 감지: {max_conf*100:.1f}%","sound":sound,"tts":tts}
            print(f"[Detector DEBUG] Risk Detected: {level} ({max_conf*100:.1f}%). Alert event generated.")
        return {"max_conf":max_conf,"level":level,"text":text,"alert_event":alert_event}

    def process_frame(self,frame_bgr):
        if frame_bgr is None: return frame_bgr,self._evaluate_risk(0.0)
        cfg=self.get_config()
        results=self.yolo.predict(source=frame_bgr,imgsz=cfg["imgsz"],verbose=False)
        processed_frame=frame_bgr.copy()
        processed_frame,boxes=draw_boxes(processed_frame,results,conf_thres=cfg["conf_thres"],names=self.names)

        # 가장 가까운 객체 선택
        min_distance=float("inf"); closest_box=None
        collision_conf=0.0
        for box in boxes:
            x1,y1,x2,y2,cls,conf=box
            distance=estimate_distance(box,cls)
            if distance<min_distance: min_distance=distance; closest_box=box

        # 충돌 분류 모델 적용
        if closest_box is not None and self.collision_model:
            x1,y1,x2,y2,cls,conf=closest_box
            roi=frame_bgr[y1:y2,x1:x2]
            roi_resized=cv2.resize(roi,(128,128))
            roi_input=np.expand_dims(roi_resized.astype(np.float32)/255.0,axis=0)
            collision_conf=float(self.collision_model.predict(roi_input)[0][0])
            max_conf=max(conf,collision_conf)
        else:
            max_conf=boxes[0][5] if boxes else 0.0

        # 최신 작업 저장
        if boxes:
            with self.data_lock:
                self.latest_job["frame"]=frame_bgr.copy()
                self.latest_job["boxes"]=boxes

        # LIME 마스크 적용
        m_pos,m_neg=None,None
        with self.data_lock:
            if self.last_mask_pos is not None and self.last_mask_neg is not None:
                if self.last_mask_pos.shape[:2]==processed_frame.shape[:2]:
                    m_pos=self.last_mask_pos.copy(); m_neg=self.last_mask_neg.copy()
        if m_pos is not None and m_neg is not None:
            processed_frame=blend_dual_mask_sequential(processed_frame,m_pos,m_neg,alpha=cfg["lime_alpha"])
        risk_data=self._evaluate_risk(max_conf)
        draw_risk_indicator(processed_frame,max_conf,cfg["warning_threshold"])
        self._calculate_fps()
        return processed_frame,risk_data

    def _calculate_fps(self):
        self.cnt+=1; now=time.time()
        if now-self.t0>=0.5: self.fps=self.cnt/(now-self.t0); self.t0,self.cnt=now,0
