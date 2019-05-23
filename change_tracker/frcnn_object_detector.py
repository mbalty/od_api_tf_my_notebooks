import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops
from object_detection.builders import model_builder
from google.protobuf import text_format
import tensorflow as tf
from object_detection.protos import model_pb2
import time
from IPython.display import clear_output
from keras.models import load_model
import keras.backend as K
from math import floor, ceil

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')


def batch_gen(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i: min(i + n, len(l))]


def split_output_dicts(outputs, images, score_tresh=.9):
    output_splitted = []
    i=0
    for big_dict in outputs:
        for j in range(len(big_dict["num_detections"])):
            output_dict = {}
            for k in big_dict.keys():
                output_dict[k] = big_dict[k][j]
            filter_scores_idx = np.where(output_dict['detection_scores'] > score_tresh)[0]
            for k in output_dict.keys():
                try:
                    output_dict[k] = output_dict[k][filter_scores_idx]
                except:
                    continue
            image_np = images[i]
            output_dict["image_np"] = image_np
            output_splitted.append(output_dict)
            i+=1
    return output_splitted


class FRCNN_Object_detector:
    def __init__(self, graph_path, memory_fraction=0.9):
        print ("detection model")

        self.graphPath = graph_path
        self.tfConfig = tf.ConfigProto()
        self.tfConfig.gpu_options.allow_growth = True
        self.od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            self.od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(self.od_graph_def, name='')
        print ("done")
    
    def run_inference_for_batch(self, images):
        with tf.Session(config=self.tfConfig) as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
            features = tf.get_default_graph().get_tensor_by_name(
                'FirstStageFeatureExtractor/resnet_v1_50/resnet_v1_50/block3/unit_6/bottleneck_v1/Relu:0')

            tensor_dict['features'] = tf.expand_dims(features, 0)
            
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            
            # Run inference
            output_dict = sess.run(tensor_dict, 
                                feed_dict={image_tensor: images})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = output_dict['num_detections'].astype(np.uint8)
            output_dict['detection_classes'] = output_dict[
            'detection_classes'].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes']
            output_dict['detection_scores'] = output_dict['detection_scores']
            output_dict['features'] = output_dict['features'][0]

            return output_dict

    def run_inference_for_single_image(self, image):
        return self.run_inference_for_batch([image])[0]

    
    def run_inference_for_images(self, images, batch_size=16, log=False):
        if log:
            print ("detect all")
        outputs = []
        progress = 0
        total_images = len(images)
        image_batch = batch_gen(images, batch_size)      
        for img_b in image_batch:
            output_dict = self.run_inference_for_batch(img_b)
            outputs.append(output_dict)
            
            progress += 1

            if log:
                print (str(min(progress*batch_size, len(images))/len(images))[:5], end="\r")
        print ("")    
        return split_output_dicts(outputs, images)

    
    