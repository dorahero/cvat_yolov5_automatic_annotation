import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf
from utils.datasets import letterbox
import torch

tf.disable_v2_behavior()

class ModelLoader:
    def __init__(self, model_path):
        self.session = None

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.session = tf.Session(graph=detection_graph, config=config)

            self.image_tensor = detection_graph.get_tensor_by_name('x:0')
            self.boxes = detection_graph.get_tensor_by_name('Identity:0')

    def __del__(self):
        if self.session:
            self.session.close()
            del self.session

    def infer(self, image):
        width, height = image.size
        image_np = np.array(image.getdata())[:, :3].reshape(
            (image.height, image.width, -1)).astype(np.uint8)
        # image_np = np.expand_dims(image_np, axis=0)
        back_size = max(width, height)
        back = np.zeros((back_size, back_size, 3))     
        back[:height, :width] = image_np 
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
        image_np = img.permute(0, 2, 3, 1).cpu().numpy()
        # if width > 1920 or height > 1080:
        #     image = image.resize((width // 2, height // 2), Image.ANTIALIAS)
        # image_np = np.array(image.getdata())[:, :3].reshape(
        #     (image.height, image.width, -1)).astype(np.uint8)
        # image_np = np.expand_dims(image_np, axis=0)

        return self.session.run(
            self.boxes,
            feed_dict={self.image_tensor: image_np})