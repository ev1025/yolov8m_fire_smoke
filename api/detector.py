# -*- coding: utf-8 -*-
"""YOLOv8m 화재/연기 추론 래퍼. 이미지·영상 공통."""
import base64, os, tempfile
import cv2
import numpy as np
import torch
from ultralytics import YOLO


class FireSmokeDetector:
    def __init__(self, weights, imgsz=640):
        self.weights = str(weights)
        self.model = YOLO(self.weights)
        self.names = self.model.names  # 클래스명은 모델에서 직접 읽음 (예: {0:'fire',1:'smoke'})
        self.device = 0 if torch.cuda.is_available() else "cpu"  # GPU 우선(실시간 속도)
        self.imgsz = imgsz

    # ---- 내부 추론 ----
    def _infer(self, img_bgr, conf):
        res = self.model.predict(img_bgr, conf=conf, device=self.device,
                                 imgsz=self.imgsz, half=(self.device != "cpu"),
                                 verbose=False)[0]
        dets = []
        for b in res.boxes:
            cid = int(b.cls[0])
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            dets.append({
                "class": self.names.get(cid, str(cid)),
                "class_id": cid,
                "confidence": round(float(b.conf[0]), 4),
                "bbox_xyxy": [round(v, 1) for v in (x1, y1, x2, y2)],
            })
        return res, dets

    @staticmethod
    def _alarm(dets):
        if not dets:
            return {"alarm": False, "status": "NORMAL", "classes": [], "max_confidence": 0.0}
        classes = sorted({d["class"] for d in dets})
        return {
            "alarm": True,
            "status": "ALARM",
            "classes": classes,
            "counts": {c: sum(1 for d in dets if d["class"] == c) for c in classes},
            "max_confidence": max(d["confidence"] for d in dets),
        }

    @staticmethod
    def _encode(img_bgr):
        ok, buf = cv2.imencode(".jpg", img_bgr)
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else None

    # ---- 이미지 ----
    def detect_image(self, img_bytes, conf=0.25, annotate=True):
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("이미지 디코드 실패")
        res, dets = self._infer(img, conf)
        out = {
            "detections": dets,
            "alarm": self._alarm(dets),
            "image_size": {"width": img.shape[1], "height": img.shape[0]},
        }
        if annotate:
            out["annotated_image"] = self._encode(res.plot())
        return out

    # ---- 영상 (프레임 샘플링) ----
    def detect_video(self, video_bytes, conf=0.25, frame_stride=15, max_frames=120, annotate_frames=False):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            tf.write(video_bytes)
            path = tf.name
        try:
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frames, idx, processed = [], 0, 0
            while processed < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % frame_stride == 0:
                    res, dets = self._infer(frame, conf)
                    fo = {
                        "frame": idx,
                        "time_sec": round(idx / fps, 2),
                        "detections": dets,
                        "alarm": bool(dets),
                    }
                    if annotate_frames and dets:
                        fo["annotated_image"] = self._encode(res.plot())
                    frames.append(fo)
                    processed += 1
                idx += 1
            cap.release()
        finally:
            os.unlink(path)

        alarm_frames = [f for f in frames if f["alarm"]]
        summary = {
            "frames_scanned": len(frames),
            "alarm": len(alarm_frames) > 0,
            "alarm_frame_count": len(alarm_frames),
            "first_alarm_time_sec": alarm_frames[0]["time_sec"] if alarm_frames else None,
        }
        return {"summary": summary, "fps": round(fps, 2), "frame_stride": frame_stride, "frames": frames}
