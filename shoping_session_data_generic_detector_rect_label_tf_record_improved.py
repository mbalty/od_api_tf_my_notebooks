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

from collections import deque
from lxml import etree
import PIL.Image
import tensorflow as tf
import random
import numpy as np 

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



class GenericObjSessionData_TfCreator:
    def __init__(self, session_dir):
        self.pastAnnotations = list()
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
        for img_name in imgs_names:
            self.tf_examples.append(self.dict_to_tf_examples(img_name))
        print(self.max_ann)
        return self.tf_examples, self.n_boxes


    def dict_to_tf_examples(self,
                            img_name,
                            ignore_difficult_instances=False):

        img_path = os.path.join(self.session_dir, img_name)
        with tf.gfile.GFile(img_path, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = PIL.Image.open(encoded_jpg_io)
        if image.format != 'JPEG':
            raise ValueError('Image format not JPEG')
        key = hashlib.sha256(encoded_jpg).hexdigest()

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

        if "object" in data:
            toRemove = set()
            filtered_ann = list()
            new_ann = list()

            for i in range(len(self.pastAnnotations)):
                label = self.pastAnnotations[i]["name"]
                if label.startswith("remove") or label.startswith("move_from"):
                    toRemove.add(i)

            for obj in data["object"]:
                label = obj["name"]
                if "add" in label or ("move" in label and not "move_from" in label):
                    for i in range(len(self.pastAnnotations)):
                        if intersection_over_second_box(self.box(obj), self.box(self.pastAnnotations[i])) > self.overlap_tresh:
                            toRemove.add(i)
                    new_ann.append(obj)
                elif "remove" in label or "move_from" in label:
                    iou_arr = np.array([iou(self.box(obj), self.box(past_obj)) for past_obj in self.pastAnnotations])
                    if iou_arr.shape[0]:
                        max_iou_i = np.argmax(iou_arr)
                        if iou_arr[max_iou_i] > self.iou_tresh:
                            toRemove.add(max_iou_i)
                    new_ann.append(obj)
                elif label != "same" and label != "first":
                    self.error_labels_file.write(str((self.session_dir, img_name, label)))

            for i in range(len(self.pastAnnotations)):
                if i not in toRemove:
                    filtered_ann.append(self.pastAnnotations[i])

            filtered_ann += new_ann

            if len(filtered_ann) > self.ann_expiration:
                filtered_ann = filtered_ann[-self.ann_expiration:]

            self.pastAnnotations = filtered_ann
        self.max_ann = max(self.max_ann, len(self.pastAnnotations))
        for obj in self.pastAnnotations:
            difficult = bool(int(obj['difficult']))
            if ignore_difficult_instances and difficult:
                continue

            difficult_obj.append(int(difficult))

            x1, y1, x2, y2 = self.box(obj)
            xmin.append(x1)
            ymin.append(y1)
            xmax.append(x2)
            ymax.append(y2)
            classes_text.append("object".encode('utf8'))
            classes.append(1)
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
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
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

    val_sessions = sessions[:25]
    train_sessions = sessions[25:]

    n_boxes_total = 0
    n_train = 0
    for session in train_sessions:
        session_dir = os.path.join(data_dir, session)

        tf_creator = GenericObjSessionData_TfCreator(session_dir=session_dir)

        session_examples, n_annotations = tf_creator.generate_tf_examples()

        n_train += len(session_examples)
        n_boxes_total += n_annotations


        for tf_example in session_examples:
            writer_train.write(tf_example.SerializeToString())

    n_val = 0
    for session in val_sessions:
        session_dir = os.path.join(data_dir, session)

        tf_creator = GenericObjSessionData_TfCreator(session_dir=session_dir)

        session_examples, n_annotations = tf_creator.generate_tf_examples()

        n_val += len(session_examples)
        n_boxes_total += n_annotations

        for tf_example in session_examples:
            writer_val.write(tf_example.SerializeToString())

    print (val_sessions)

    print ("n_train", n_train, "\nn_val", n_val)

    writer_train.close()
    writer_val.close()

if __name__ == '__main__':
    tf.app.run()
