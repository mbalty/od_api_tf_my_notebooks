# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_pascal_tf_record.py \
        --data_dir=/home/user/VOCdevkit \
        --year=VOC2012 \
        --output_path=/home/user/pascal.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import cv2
from collections import deque
from lxml import etree
import PIL.Image
import tensorflow as tf
import random
import numpy as np 
from PIL import Image

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                                    'merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('output_path_train', '', 'Path to output TFRecord')
flags.DEFINE_string('output_path_val', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                                                          'difficult instances')
FLAGS = flags.FLAGS

def intersection_over_second_box(new_box, old_box):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        new_box (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        old_box (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """

    new_box = [float(x) for x in new_box]
    old_box = [float(x) for x in old_box]

    (x0_1, y0_1, x1_1, y1_1) = new_box
    (x0_2, y0_2, x1_2, y1_2) = old_box

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_new = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_old = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)

    return size_intersection / size_old

def iou(new_box, old_box):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        new_box (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        old_box (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """

    new_box = [float(x) for x in new_box]
    old_box = [float(x) for x in old_box]

    (x0_1, y0_1, x1_1, y1_1) = new_box
    (x0_2, y0_2, x1_2, y1_2) = old_box

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_new = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_old = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_new + size_old - size_intersection

    return size_intersection / size_union

norm_image = lambda image: cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

def diff(first, second):
    return ((norm_image(first)-norm_image(second)) * 255).astype("uint8")

def load_image_into_numpy_array(image_path):
    image = Image.open(image_path)
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


class GenericObjSessionData_TfCreator:
    def __init__(self, session_dir):
        self.prevAnnotationsToRemove = list()
        self.removeAnnotationNext = list()
        self.session_dir = session_dir
        self.annotations_dir =  os.path.join(session_dir, "annotations")
        self.image_annotation_dict = dict()
        self.width = 0
        self.height = 0
        self.tf_examples = list()
        self.overlap_tresh = 0.6
        self.iou_tresh = 0.6
        self.ann_expiration = 50
        self.max_ann = 0
        self.n_boxes = 0
        self.dif_folder = os.path.join(session_dir, "stack")
        if not os.path.exists(self.dif_folder):
            os.makedirs(self.dif_folder)

        self.error_labels_file = open("error_labels_file.txt", "w+")
        print (session_dir)
        if os.path.exists(self.annotations_dir):
            for ann_file in sorted(os.listdir(self.annotations_dir)):
                path = os.path.join(self.annotations_dir, ann_file)
                with tf.gfile.GFile(path, 'r') as fid:
                    xml_str = fid.read()
                xml = etree.fromstring(xml_str)
                data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
                self.image_annotation_dict[data['filename']] = data
                if self.width == 0:
                    self.width = int(data['size']['width'])
                    self.height = int(data['size']['height'])
        else:
            raise Exception("this session does not have annotations")

    def box(self, obj):
        return (float(obj['bndbox']['xmin']) / self.width,
                float(obj['bndbox']['ymin']) / self.height,
                float(obj['bndbox']['xmax']) / self.width,
                float(obj['bndbox']['ymax']) / self.height)


                
    def generate_tf_examples(self):
        imgs_names = sorted([f for f in os.listdir(self.session_dir) if f.endswith(".jpg")])
        dif_names = [f for f in os.listdir(self.dif_folder) if f.endswith(".jpg")]
        prev = None

        if len(imgs_names) - len(dif_names) > 1:
            print ("generating dif")
            for img_name in imgs_names:
                img_path = os.path.join(self.session_dir, img_name)
                im = load_image_into_numpy_array(img_path)
                dif_path = os.path.join(self.dif_folder, img_name)
                if prev is not None and not os.path.exists(dif_path):
                    dif_img = np.concatenate([prev, im], axis=-1)
                    np.save.save(os.path.join(self.dif_folder, img_name), dif_img)
                prev = im
        print ("creating tf records")

        for img_name in imgs_names[1:]:
            self.tf_examples.append(self.dict_to_tf_examples(img_name))
        return self.tf_examples, self.n_boxes


    def dict_to_tf_examples(self,
                            img_name,
                            ignore_difficult_instances=False):

        img_path = os.path.join(self.dif_folder, img_name)
        encoded_jpg = np.load(img_path)
        encoded_inputs = encoded_jpg.tostring()
        key = hashlib.sha256(encoded_inputs).hexdigest()

        if img_name in self.image_annotation_dict:
            data = self.image_annotation_dict[img_name]
        else:
            data = dict()

        width = self.width
        height = self.height

        xmin = []
        ymin = []
        xmax = []
        ymax = []
        classes = []
        classes_text = []
        truncated = []
        poses = []
        difficult_obj = []

        new_ann = list()
        new_ann += self.prevAnnotationsToRemove
        self.prevAnnotationsToRemove = list()


        if "object" in data:

            for obj in data["object"]:
                label = obj["name"]
                if "add" in label:
                    obj["cls"] = 1
                    new_ann.append(obj)
                elif ("move" in label and not "move_from" in label):
                    obj["cls"] = 3
                    new_ann.append(obj)
                elif "remove" in label:
                    obj["cls"] = 2
                    self.prevAnnotationsToRemove.append(obj)
                elif"move_from" in label:
                    obj["cls"] = 4
                    self.prevAnnotationsToRemove.append(obj)
                elif label != "same" and label != "first":
                    self.error_labels_file.write(str((self.session_dir, img_name, label)))

        for obj in new_ann:
            difficult = bool(int(obj['difficult']))
            if ignore_difficult_instances and difficult:
                continue
            difficult_obj.append(int(difficult))
            x1, y1, x2, y2 = self.box(obj)
            xmin.append(x1)
            ymin.append(y1)
            xmax.append(x2)
            ymax.append(y2)
            classes_text.append(obj["name"].encode('utf8'))
            classes.append(obj["cls"])
            truncated.append(int(obj['truncated']))
            poses.append(obj['pose'].encode('utf8'))

        self.n_boxes += len(xmin)

        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(
                img_name.encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(
                img_name.encode('utf8')),
            'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded_inputs),
            'image/channels': dataset_util.int64_feature(6),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
            'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
            'image/object/truncated': dataset_util.int64_list_feature(truncated),
            'image/object/view': dataset_util.bytes_list_feature(poses),
        }))
        return example


def main(_):
    data_dir = FLAGS.data_dir
    sessions=[]

    for s in sorted(os.listdir(data_dir)):
        try:
            a = int(s)
            sessions.append(s)
        except:
            continue

    writer_train = tf.python_io.TFRecordWriter(FLAGS.output_path_train)
    writer_val = tf.python_io.TFRecordWriter(FLAGS.output_path_val)

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    random.shuffle(sessions)

    val_sessions = {'162737', '123071', '123229', '164085', '160827', '163832', '161552', '118828', '161441', '118859', '123178', '160685', '122874', '122949', '161209', '161612', '123051', '163888', '160037', '161185', '161069', '161095', '123082', '160665', '160977'}

    n_boxes_total = 0
    n_train = 0
    n_val = 0
    for session in sorted(sessions):
        session_dir = os.path.join(data_dir, session)

        tf_creator = GenericObjSessionData_TfCreator(session_dir=session_dir)

        session_examples, n_annotations = tf_creator.generate_tf_examples()

        if session in val_sessions:
            for tf_example in session_examples:
                writer_val.write(tf_example.SerializeToString())
                n_val += 1
        else:
            for tf_example in session_examples:
                writer_train.write(tf_example.SerializeToString())
                n_train += 1


    print (val_sessions)

    print ("n_train", n_train, "\nn_val", n_val)

    writer_train.close()
    writer_val.close()

if __name__ == '__main__':
    tf.app.run()
