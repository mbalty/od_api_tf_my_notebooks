from frcnn_object_detector import FRCNN_Object_detector
from tracker import ChangeTracker
from utils import label_map_util
from utils import visualization_utils as vis_util
import os
import time
from pil import Image


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


class SessionChangeTracker:
    def __init__(self, detection_model_path, similarity_model_path):
        self.detector = FRCNN_Object_detector(detection_model_path)
        self.changeTracker = ChangeTracker()

    def process_session(self, session_folder):
        pass

    def download_images(self, session_folder):
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
            print (str(len(images)/len(image_paths))[:5], end="\r")
