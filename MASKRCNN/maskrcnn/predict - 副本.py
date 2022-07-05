
import cv2

image = cv2.imread(r'.\images\test\0455.jpg')
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


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
    NUM_CLASSES = 1 + 4  # Background + cirtus

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.1

config = BalloonConfig()
model = modellib.MaskRCNN(mode="inference", config=config,
                              model_dir=r'.\logs\\')

model.load_weights(r'.\logs\epoch013_loss0.180_val_loss0.525.h5', by_name=True)

result = model.detect([image])


class_names = ['BG', 'BigFlower', 'MiddleFlower', 'SmallFlower', 'Bud']
from mrcnn.visualize import display_instances
display_instances(image, result[0]['rois'], result[0]['masks'], result[0]['class_ids'], class_names,
                      scores=result[0]['scores'], title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None)


