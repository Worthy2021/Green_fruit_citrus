
import cv2

image = cv2.imread(r'../MASKRCNN/maskrcnn/images/test/4.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


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

result = model.detect([image])


class_names = ['BG', 'cirtus']
from mrcnn.visualize import display_instances
display_instances(image, result[0]['rois'], result[0]['masks'], result[0]['class_ids'], class_names,
                      scores=result[0]['scores'], title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None)


