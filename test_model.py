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

with open('function.yaml') as f:
    names_json = yaml.load(f, Loader=yaml.FullLoader)['metadata']['annotations']['spec']  # class names (assume COCO)
names = [item['name'] for item in json.loads(names_json)]
colors = [[0, 0, 255]]
# tf.disable_v2_behavior()
model_path = "model.pb"
img0 = cv2.imread("zinger_2016_gray_20201027_01_n-00089.jpg")
def wrap_frozen_graph(graph_def, inputs, outputs):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph
    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))
graph = tf.Graph()
# with detection_graph.as_default():
#     od_graph_def = tf.GraphDef()
#     with tf.gfile.GFile(model_path, 'rb') as fid:
#         serialized_graph = fid.read()
#         od_graph_def.ParseFromString(serialized_graph)
#         tf.import_graph_def(od_graph_def, name='')

#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     print(detection_graph.get_tensor_by_name("x:0"))
#     print(detection_graph.get_tensor_by_name("Identity:0"))
graph_def = graph.as_graph_def()
graph_def.ParseFromString(open(model_path, 'rb').read())
frozen_func = wrap_frozen_graph(graph_def=graph_def, inputs="x:0", outputs="Identity:0")
w, h, _ = img0.shape
back_size = max(w, h)
back = np.zeros((back_size, back_size, 3))
back[:w, :h] = img0
im0 = back.copy()
# cv2.imwrite('test.jpg', back)
imgsz = 640
img = torch.zeros((1, 3, imgsz, imgsz), device=torch.device('cuda:0'))
img = letterbox(back, new_shape=imgsz)[0]

# Convert
img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
img = np.ascontiguousarray(img)
img = torch.from_numpy(img).to(torch.device('cuda:0'))
img = img.float()  # uint8 to fp16/32
img /= 255.0  # 0 - 255 to 0.0 - 1.0
if img.ndimension() == 3:
    img = img.unsqueeze(0)
pred = frozen_func(x=tf.constant(img.permute(0, 2, 3, 1).cpu().numpy())).numpy()
pred[..., :4] *= imgsz
pred = torch.tensor(pred)
pred = non_max_suppression(pred, 0.5, 0.45, agnostic=False)
# print(pred)
for i, det in enumerate(pred): 
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        # # Print results
        # for c in det[:, -1].unique():
        #     n = (det[:, -1] == c).sum()  # detections per class

        # Write results
        for *xyxy, conf, cls in reversed(det):
            print(xyxy)
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, img0, label=label, color=colors[0], line_thickness=3)
            # print(img.shape)
            # print(im0.shape)
# im0_o = cv2.resize(img0, (640, 480)) 
# cv2.imshow('1', im0_o)
# cv2.waitKey()
# cv2.waitKey(0)
# cv2.destroyAllWindows()
