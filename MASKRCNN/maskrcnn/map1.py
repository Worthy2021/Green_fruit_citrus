import cv2
import os

from mrcnn import utils
from mrcnn.utils import compute_ap
from mrcnn.config import Config
from mrcnn import model as modellib
import skimage.draw
import numpy as np
import json
from mrcnn.visualize import display_instances
import matplotlib.pyplot as plt

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

class BalloonDataset(utils.Dataset):

    def load_cirtus(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("cirtus", 1, "cirtus")

        # Train or validation dataset?
        assert subset in ["train", "val","test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "cirtus",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "cirtus":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cirtus":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def text_save(filename, data):#filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存txt文件成功")

if __name__ == '__main__':
    config = BalloonConfig()
    dataset = BalloonDataset()
    dataset.load_cirtus(r'.\images', "test")
    dataset.prepare()
    
    model = modellib.MaskRCNN(mode="inference", config=config,
                              model_dir=r'.\logs\\')
    
    mAps = []
    precisions = []
    recalls = []
    overlaps = []
    count = 0
    
    model.load_weights(r'.\logs\mask_rcnn_res50_cirtus2228_0030.h5', by_name=True)
    class_names = ['BG', 'cirtus']
    
    for image_id in dataset.image_ids:
        
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        
        # 将所有ground truth载入并保存
        if count == 0:
            save_box, save_class, save_mask = gt_bbox, gt_class_id, gt_mask
        else:
            save_box = np.concatenate((save_box, gt_bbox), axis=0)
            save_class = np.concatenate((save_class, gt_class_id), axis=0)
            save_mask = np.concatenate((save_mask, gt_mask), axis=2)
        
        result = model.detect([image])
        r = result[0]
 
        # 将所有检测结果保存
        if count == 0:
            save_roi, save_id, save_score, save_m = r["rois"], r["class_ids"], r["scores"], r['masks']
        else:
            save_roi = np.concatenate((save_roi, r["rois"]), axis=0)
            save_id = np.concatenate((save_id, r["class_ids"]), axis=0)
            save_score = np.concatenate((save_score, r["scores"]), axis=0)
            save_m = np.concatenate((save_m, r['masks']), axis=2)
     
        count += 1
        
        
    AP, precisions, recalls, overlaps = \
            utils.compute_ap(save_box, save_class, save_mask,
                             save_roi, save_id, save_score, save_m)
     
    print("AP: ", AP)
       
        
    print("mAps:", np.mean(AP))
    
    plt.plot(recalls, precisions, 'b', label='PR')
    plt.title('precision-recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()
    
    #text_save('precisioncbnet.txt', precisions)
    #text_save('recallcbnet.txt', recalls)
    
    
    
    
    

    
   


