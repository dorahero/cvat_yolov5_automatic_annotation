import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf
import yaml
import torch
import json
import cv2
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.datasets import letterbox
from model_loader import ModelLoader
import os 

os.environ["CUDA_VISIBLE_DEVICES"]="1"
model_path = "/opt/nuclio/model.pb"
model_handler = ModelLoader(model_path)
functionconfig = yaml.safe_load(open("/opt/nuclio/function.yaml"))
labels_spec = functionconfig['metadata']['annotations']['spec']
labels = {item['_id']: item['name'] for item in json.loads(labels_spec)}
image = Image.open('zinger_2016_gray_20201027_01_n-00089.jpg')
w, h = image.size
max_size = max(w, h)

pred = model_handler.infer(image)
pred[..., :4] *= 640
pred = torch.tensor(pred)
pred = non_max_suppression(pred, 0.5, 0.45, agnostic=False)

results = []
for i, det in enumerate(pred):
    if len(det):
        det[:, :4] = scale_coords((640, 640, 3), det[:, :4], (max_size, max_size, 3)).round()
    for *xyxy, conf, cls in reversed(det):
        xtl = float(xyxy[0])
        ytl = float(xyxy[1])
        xbr = float(xyxy[2])
        ybr = float(xyxy[3])
        obj_score = str(conf)
        obj_class = int(cls)
        obj_label = labels.get(obj_class, "unknown")
        results.append({
            "confidence": str(obj_score),
            "label": obj_label,
            "points": [xtl, ytl, xbr, ybr],
            "type": "rectangle",
        })
print(results)