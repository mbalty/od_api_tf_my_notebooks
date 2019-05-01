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


class GenericObjSessionData_TfCreator:
    def __init__(self, session_dir,  past_window=10):
        self.past = deque()
        self.pastWindow = past_window
        self.session_dir = session_dir
        self.annotations_dir =  os.path.join(session_dir, "annotations")
        self.image_annotation_dict = dict()
        self.width = 0
        self.height = 0
        self.tf_examples = list()

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


    def generate_tf_examples(self):
        imgs_names = sorted([f for f in os.listdir(self.session_dir) if f.endswith(".jpg")])
        for img_name in imgs_names:
            self.tf_examples.append(self.dict_to_tf_examples(img_name))

        return self.tf_examples


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

        self.past.appendleft(data)

        if len(self.past) > self.pastWindow:
            self.past.pop()
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

        for k in range(len(self.past)):
            pastData = self.past[k]
            if 'object' in pastData:
                for obj in pastData['object']:
                    label = obj['name']
                    if label == "add" or ("move" in label[:4] and "from" not in label) or (
                            k == 0 and "remo" in label) or (k == 0 and "from" in label):
                        difficult = bool(int(obj['difficult']))
                        if ignore_difficult_instances and difficult:
                            continue

                        difficult_obj.append(int(difficult))

                        xmin.append(float(obj['bndbox']['xmin']) / width)
                        ymin.append(float(obj['bndbox']['ymin']) / height)
                        xmax.append(float(obj['bndbox']['xmax']) / width)
                        ymax.append(float(obj['bndbox']['ymax']) / height)
                        classes_text.append("object".encode('utf8'))
                        classes.append(1)
                        truncated.append(int(obj['truncated']))
                        poses.append(obj['pose'].encode('utf8'))

                        if ("move" in label[:4] and "from" not in label):
                            break
                    else:
                        break


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

    sessions = sorted(os.listdir(data_dir))

    writer_train = tf.python_io.TFRecordWriter(FLAGS.output_path_train)
    writer_val = tf.python_io.TFRecordWriter(FLAGS.output_path_val)

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    idx = 0
    val_cut = 0.001

    n_train = 0
    n_val = 0
    for session in sessions:
        try:
            int(session)
        except:
            continue

        if idx % 100 == 0:
            logging.info('On session # %d of %d', idx, len(sessions))
        idx += 1
        session_dir = os.path.join(data_dir, session)
        annotations_dir = os.path.join(data_dir, session, "annotations")

        tf_creator = GenericObjSessionData_TfCreator(session_dir=session_dir)


        for tf_example in tf_creator.generate_tf_examples():
            if random.random() > val_cut:
                writer_train.write(tf_example.SerializeToString())
                n_train += 1
            else:
                writer_val.write(tf_example.SerializeToString())
                n_val += 1


    print ("n_train", n_train, "\nn_val", n_val)
    writer_train.close()
    writer_val.close()


if __name__ == '__main__':
    tf.app.run()
