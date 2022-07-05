import cv2
import os
import time
import numpy as np

from mrcnn.config import Config
from mrcnn import model as modellib

class BalloonConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "cirtus"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + cirtus

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.5

config = BalloonConfig()
model = modellib.MaskRCNN(mode="inference", config=config,
                              model_dir=r'.\logs\\')

model.load_weights(r'.\logs\mask_rcnn_res50_cirtus2228_0030.h5', by_name=True)

img_dir = os.listdir(r'.\images\test')
t = []

for img in img_dir:
    
    image = cv2.imread(r'.\images\test\\'+img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    t1 = time.time()
    model.detect([image])
    t2 = time.time()
    t.append(t2-t1)
    print(t2-t1)
t.pop(0)
print('avgtime:', np.mean(t))