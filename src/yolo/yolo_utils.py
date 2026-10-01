# src/yolo/yolo_utils.py
import os
from ultralytics import YOLO
import numpy as np

# Load a YOLOv8 model (default = yolov8n for best speed on CPU)


def load_yolo(model_name="yolov8n.pt"):
    """
    Loads YOLOv8 model. If model_name is 'yolov8n', it downloads automatically.
    """
    model = YOLO(model_name)
    return model

# Perform inference and return list of detections


def yolo_detect(model, frame, conf=0.35):
    """
    Runs YOLO detection on a frame.
    Returns a list of dict:
    { "name": str, "conf": float, "box": (l,t,r,b) }
    """
    # YOLO accepts RGB or BGR; we pass it directly.
    results = model(frame, imgsz=640, conf=conf, verbose=False)[0]

    detections = []
    for box, cls, score in zip(results.boxes.xyxy.cpu().numpy(),
                               results.boxes.cls.cpu().numpy().astype(int),
                               results.boxes.conf.cpu().numpy()):

        left, top, right, bottom = box.astype(int)
        name = model.names.get(cls, str(cls))

        detections.append({
            "name": name,
            "conf": float(score),
            "box": (left, top, right, bottom)
        })

    return detections