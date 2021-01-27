import json
import base64
import io
from PIL import Image
import yaml
from model_loader import ModelLoader
from utils.general import non_max_suppression, scale_coords
import torch



def init_context(context):
    context.logger.info("Init context...  0%")
    model_path = "/opt/nuclio/model.pb"
    model_handler = ModelLoader(model_path)
    setattr(context.user_data, 'model_handler', model_handler)
    functionconfig = yaml.safe_load(open("/opt/nuclio/function.yaml"))
    labels_spec = functionconfig['metadata']['annotations']['spec']
    labels = {item['_id']: item['name'] for item in json.loads(labels_spec)}
    setattr(context.user_data, "labels", labels)
    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run faster_rcnn_inception_v2_coco model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"].encode('utf-8')))
    threshold = float(data.get("threshold", 0.5))
    image = Image.open(buf)
    w, h = image.size
    max_size = max(w, h)

    pred = context.user_data.model_handler.infer(image)
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
            obj_label = context.user_data.labels.get(obj_class, "unknown")
            results.append({
                "confidence": str(obj_score),
                "label": obj_label,
                "points": [xtl, ytl, xbr, ybr],
                "type": "rectangle",
            })

    return context.Response(body=json.dumps(results), headers={},
        content_type='application/json', status_code=200)