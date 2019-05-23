from change_tracker.tracker import ChangeTracker
from change_tracker.frcnn_object_detector import FRCNN_Object_detector
from utils import label_map_util
from utils import visualization_utils as vis_util
import os
import time
from PIL import Image
import numpy as np
import scipy
from matplotlib import pyplot as plt


IMAGE_SIZE = (18, 12)


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def download_images(session_folder, log=False):
    if log:
        print ("download images")
    image_paths = sorted([f for f in os.listdir(session_folder) if f.endswith(".jpg")])
    total_images = len(image_paths)
    prev_features = None

    images = []
    for image_fname in image_paths:
        t0 = time.time()
        image_path = os.path.join(session_folder, image_fname)

        image = Image.open(image_path)
        image_np = load_image_into_numpy_array(image)
        images.append(image_np)
        if log:
            print (str(len(images)/len(image_paths))[:5], end="\r")
    print ()
    return images

class SessionChangeTracker:
    def __init__(self, detection_model_path, similarity_model_path):
        self.detector = FRCNN_Object_detector(detection_model_path)
        self.changeTracker = ChangeTracker()

    def process_session(self, session_folder, visualize=False, save_results=False, results_folder="/tmp/"):
        images = download_images(session_folder, log=True)
        # print (len(images))
        detections = self.detector.run_inference_for_images(images, log=True)

        # print (len(detections))
        self.changeTracker.reset()

        for frame in detections:
            self.changeTracker.update(frame["detection_boxes"], frame["image_np"])

        if visualize:
            category_index = {
                1: {'id': 1, 'name': 'add'},
                2: {'id': 2, 'name': 'remove'},
                3: {'id': 3, 'name': 'move'}, 
                4: {'id': 4, 'name': 'same'}}

            for i in range(len(detections)):
                det = detections[i]
                boxes = np.array(self.changeTracker.detectionHistory["boxes"][i])
                classes = np.array(self.changeTracker.detectionHistory["classes"][i])
                ids = np.array(self.changeTracker.detectionHistory["ids"][i])
                
                image = det["image_np"]
                image_np_out = np.copy(image)
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np_out,
                    np.array(boxes),
                    np.array(classes),
                    np.array(ids)/100.,
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=5,
                    min_score_thresh=-0.0000000000001)
                
                if save_results:
                    scipy.misc.imsave(os.path.join(results_folder, str(i) + ".jpg"), image_np_out)        
                
                plt.figure(figsize=IMAGE_SIZE)
                plt.title(str(i))
                plt.imshow(image_np_out)
        # print ("]")
