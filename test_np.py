import numpy as np
from PIL import Image

image = Image.open('zinger_2016_gray_20201027_01_n-00089.jpg')
width, height = image.size
image_np = np.array(image.getdata())[:, :3].reshape(
    (image.height, image.width, -1)).astype(np.uint8)
# image_np = np.expand_dims(image_np, axis=0)
print(image_np.shape)