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

"""Detection model trainer.

This file provides a generic training method that can be used to train a
DetectionModel.
"""
import warnings
warnings.filterwarnings("ignore")
import os
import errno
import tensorflow as tf
import helper.train_helper as th
import pickle
import glob
import helper.base_fl
from object_detection.core import batcher
from object_detection.core import preprocessor
from object_detection.core import standard_fields as fields
from object_detection.utils import ops as util_ops
from object_detection.utils import config_util
from object_detection.builders import model_builder
import numpy as np
import math
import helper.io_utils as iou
from multiprocessing.dummy import Pool as ThreadPool
from helper.feature.vis_pred_kitchen import SeqLoader
import argparse


tf.logging.set_verbosity(tf.logging.INFO)
# MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28' #faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28, ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03
MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28' #faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28, ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03
# MODEL_NAME = 'ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03' #faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28, ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03
ds='coco'
task= 'hand'
mode='fine'#coco_0,coco_fine,oid_0
fname='_'.join([ds,mode])
num_clones=1


flags = tf.app.flags
flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')
flags.DEFINE_integer('task', 0, 'task id')
flags.DEFINE_integer('gpu_id', 0, 'gpu id')

flags.DEFINE_integer('num_clones', num_clones, 'Number of clones to deploy per worker.')
flags.DEFINE_integer('batch_size', 10, 'Number of clones to deploy per worker.')
flags.DEFINE_boolean('clone_on_cpu', False,
                     'Force clones to be deployed on CPU.  Note that even if '
                     'set to False (allowing ops to run on gpu), some ops may '
                     'still be run on the CPU if they have no GPU kernel.')
flags.DEFINE_integer('worker_replicas', 1, 'Number of worker+trainer '
                     'replicas.')
flags.DEFINE_integer('ps_tasks', 1,
                     'Number of parameter server tasks. If None, does not use '
                     'a parameter server.')
flags.DEFINE_string('train_dir', '{0}_train/{1}/{2}/'.format(task, MODEL_NAME, fname),
                    'Directory to save the checkpoints and training summaries.')

flags.DEFINE_string('pipeline_config_path', iou.getBaseDir()+'/hand/hand_train/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/send/pipeline_analogai5.config',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file. If provided, other configs are ignored')

flags.DEFINE_string('train_config_path', '',
                    'Path to a train_pb2.TrainConfig config file.')
flags.DEFINE_string('input_config_path', '',
                    'Path to an input_reader_pb2.InputReader config file.')
flags.DEFINE_string('model_config_path', '',
                    'Path to a model_pb2.DetectionModel config file.')

FLAGS = flags.FLAGS


slim = tf.contrib.slim
GPU_dic={}

# height = 1080;     width = 1920
# height = 1080;     width = 1920
height = 720;  width = 1280
# height = 480;  width = 640

#train_sbj_lst=['P02','P03','P04','P05','P06','P07','P08','P12','P13','P14','P15','P17','P19','P20','P21','P22','P24','P25','P26','P27','P28','P29','P31']
#val_sbj_lst=['P01','P10','P16','P23','P30']
# P_lst=['P01','P02','P03','P04','P05','P06','P07','P08','P10','P12','P13','P14','P15','P16','P17','P19','P20','P21','P22','P23','P24','P25','P26','P27','P28','P29','P30','P31']


P_lst= ['P19']#['P19','P20','P26']
#01=110000,22=111516,P08=78K
#This computer: ['P02','P14']
sub_path=''
sub_lbl_lst={}
s_lbl=[-1,-1]
if 'analogai' in iou.getMName():
    bpath = '/mnt/ds/kitchen/segments_crop_rcnn/'
    sub_path = 'imgs_6'
else:
    bpath='/media/cskn/ds/kitchen/analogai_merge/segments_crop_rcnn/'
    sub_path='imgs_6'

# cls_lst=['124_reverse']
cls_lst=[]
sub_lbl=[]
seq_lst=[]
P_lst=[]

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", help="which gpu to use",
                    type=int)
args = parser.parse_args()
gpu_id=args.gpu_id

dic_paths=th.get_paths()
bpath_hand=dic_paths['bpath_hand_detect']

bpath_hand_detections_base=dic_paths['bpath_hand_detect']
bpath_hand_detections_base=dic_paths['bpath_hand_detect']

if len(cls_lst)>0:
    pickle_path_pre=bpath_hand_detections_base+'SeqLoader{}_pre.p'.format('_'.join(cls_lst))
    pickle_path_4=bpath_hand_detections_base+'SeqLoader{}_4.p'.format('_'.join(cls_lst))
else:
    pickle_path_pre = bpath_hand_detections_base + 'SeqLoader{}_pre.p'.format('_all')
    pickle_path_4 = bpath_hand_detections_base + 'SeqLoader{}_4.p'.format('_all')

print('Looking for pickle %s'%pickle_path_pre)
print('Looking for pickle %s'%pickle_path_4)


if os.path.exists(pickle_path_pre):
    print('Loading saved detections %s'%pickle_path_pre)
    with open(pickle_path_pre, 'rb') as thefile:
       sl_pre= pickle.load(thefile)
else:
    bpath_hand = dic_paths['bpath_hand_detect']
    sl_pre = SeqLoader(bpath_hand + '/hand/atrous/detections', im_height=72, im_width=128, im_depth=1088
    ,sel_lbl = cls_lst,dim_multplier = 1.75, dim_square = 'h')

    with open(pickle_path_pre,'wb') as thefile:
        pickle.dump(sl_pre,thefile)

if os.path.exists(pickle_path_4):
    print('Loading saved detections %s'%pickle_path_4)
    with open(pickle_path_4, 'rb') as thefile:
       sl_4= pickle.load(thefile)
else:
    bpath_hand = dic_paths['bpath_hand_detect']

    sl_4 = SeqLoader(bpath_hand + '/hand/atrous/detections', im_height=144, im_width=256, im_depth=192,
                     sel_lbl=cls_lst,dim_multplier=1.75,dim_square='h')
    with open(pickle_path_4,'wb') as thefile:
        pickle.dump(sl_4,thefile)


# sl =SeqLoader(bpath+'/features_19497_10_all/hand/atrous/detections',im_height=72, im_width=128,im_depth=1088,P_lst=P_lst,cls_lst=cls_lst,sub_lbl=sub_lbl,seq_prms=seq_lst,dim_multplier=1.75,dim_square='h')
# print('Detections are loaded: %i'%sl.total_cnt)
complete_image_list = sorted(glob.glob(bpath+sub_path+'/**/**/**/*.jpg'))
print 'Complete images: %i'%len(complete_image_list)
image_list = []
for im in complete_image_list:
    new_im=im.replace('/imgs_6/', '/features_19497_10_coco/files_dump/')
    # new_im=new_im.replace('/media/cskn/ds/kitchen/analogai_merge/segments_crop_rcnn/features_19497_10_coco/','/mnt/4tb/ds/kitchen/analogai_merge/features_19497_10_coco_new/')
    new_im=new_im.replace('.jpg','_pre.p')
    if os.path.exists(new_im):
        continue
    image_list.append(im)
# image_list = iou.read_labeled_image_large_list(bpath + '/' + sub_path, P_lst,cls_lst_prms=cls_lst,sub_lbl_prms=sub_lbl,seq_prms=seq_lst,sub=6,sindex=0)
# item_count = float(len(image_list))
# iter_count = int(math.ceil(item_count / FLAGS.batch_size))
# print('Images are loaded: %i'%len(image_list))

image_list=sorted(image_list)
image_list=np.asarray(image_list)
image_list=image_list[range(gpu_id,len(image_list),12)]
# image_list=image_list[6000:]
image_list=image_list.tolist()
print('Processing images: %i, gpu_id: %i'%(len(image_list),gpu_id))
#/media/t-hucosk/ds/GAZE/imgs/P17-R04-ContinentalBreakfast/P17-R04-ContinentalBreakfast-398216-410378-F009528-F009879
# image_list = glob.glob('/media/t-hucosk/ds/GAZE/imgs/P17-R04-ContinentalBreakfast/P17-R04-ContinentalBreakfast-398216-410378-F009528-F009879/*.jpg')
item_count = float(len(image_list))
iter_count = int(math.ceil(item_count / FLAGS.batch_size))
print('Iteration count: %i'%iter_count)



def multi_thread_save_detection(vals):
    # print(vals[0])
    # if 'frame_0000000043.jpg' in vals[0]:
    #     print 'Cacth....'
    base_name = os.path.basename(vals[0]).replace('.jpg', '')
    new_dir = os.path.dirname(vals[0].replace('/imgs_6/', '/features_19497_10_coco/files_dump3/'))

    # print vals[0]
    # print vals[1].shape
    # print vals[2].shape
    feats_pre_aux = sl_pre.get_all_max_mean_feats(vals[1], vals[0], im_width=128, im_height=72)
    feats_4 = sl_4.get_all_max_mean_feats(vals[2], vals[0], im_width=256, im_height=144)

    if not os.path.exists(new_dir):
        try:
            os.makedirs(new_dir)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass

    pickle_path_file_pre = os.path.join(new_dir, base_name + '_pre.p')
    pickle_path_file_4 = os.path.join(new_dir, base_name + '_4.p')
    with open(pickle_path_file_pre, 'wb') as thefile:
        pickle.dump(feats_pre_aux, thefile)
    thefile.close()

    with open(pickle_path_file_4, 'wb') as thefile:
        pickle.dump(feats_4, thefile)
    thefile.close()



def save_features(path_lst,feats_pre_aux,feats_4):
    # feats_pre_aux = results[1]
    prms=zip(path_lst,feats_pre_aux,feats_4)
    # for i in range(len(prms)):
    #     print i
    #     multi_thread_save_detection(prms[i])
    pool = ThreadPool(len(prms))
    pool.map(multi_thread_save_detection, prms)
    pool.close()
    pool.join()

def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    file_contents = tf.read_file(input_queue)
    image = tf.image.decode_jpeg(file_contents, channels=3)
    image = tf.cast(image, tf.float32)
    image.set_shape([height, width, 3])

    return image

def wrapped_complex_calulation(X_paths,bidx):
        image = tf.map_fn(read_images_from_disk, elems=X_paths, dtype=(tf.float32),
                                    parallel_iterations=FLAGS.batch_size)
        return image,bidx,X_paths


def test_gen():
    for idx in range(10000000):
        bidx = idx % iter_count
        miss = 0
        if (bidx + 1) * FLAGS.batch_size > len(image_list):
            miss = (bidx + 1) * FLAGS.batch_size - len(image_list)

        res=image_list[bidx * FLAGS.batch_size - miss:(bidx + 1) * FLAGS.batch_size - miss]
        yield res,bidx

def run_inference_for_single_image():
    if FLAGS.pipeline_config_path:
        configs = config_util.get_configs_from_pipeline_file(
            FLAGS.pipeline_config_path)
    else:
        configs = config_util.get_configs_from_multiple_files(
            model_config_path=FLAGS.model_config_path,
            train_config_path=FLAGS.train_config_path,
            train_input_config_path=FLAGS.input_config_path)
        if FLAGS.task == 0:
            for name, config in [('model.config', FLAGS.model_config_path),
                                 ('train.config', FLAGS.train_config_path),
                                 ('input.config', FLAGS.input_config_path)]:
                tf.gfile.Copy(config, os.path.join(FLAGS.train_dir, name),
                              overwrite=True)

    model_config = configs['model']


    with tf.Graph().as_default() as graph:
        ds = tf.data.Dataset.from_generator(
            test_gen, (tf.string,tf.int32),
            output_shapes=(
                tf.TensorShape([FLAGS.batch_size]),tf.TensorShape([])))

        ds = ds.map(wrapped_complex_calulation,
                    num_parallel_calls=20).prefetch(5)

        image_batch,bidx_lst,img_path_lst = ds.make_one_shot_iterator().get_next()

        # config = tf.ConfigProto()
        config = tf.ConfigProto(
            device_count={'GPU': 1},
            # allow_soft_placement=True,log_device_placement = False
        )
        # config.gpu_options.per_process_gpu_memory_fraction = 0.8
        print('Creating session')
        with tf.Session(config=config) as sess:
            print('Session created....')

            detection_model = model_builder.build(model_config=model_config, is_training=False)
            print('Model build done...')

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            print('Coord start done...')

            preprocessed_images = []
            true_image_shapes = []

            image_lst = image_batch
            resized_image, true_image_shape = detection_model.preprocess(image_lst)
            preprocessed_images.append(resized_image)
            true_image_shapes.append(true_image_shape)

            rpn_features_to_crop, rett = detection_model._feature_extractor.extract_proposal_features(
                resized_image, scope=detection_model.first_stage_feature_extractor_scope)

            # prediction_dict = detection_model.predict(resized_image, true_image_shape)
            # post_prediction_dict=detection_model.postprocess(prediction_dict,true_image_shape)

            available_var_map = tf.global_variables()
            init_saver = tf.train.Saver(available_var_map)
            # check_point_path = 'hand_train/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/send/model.ckpt-19497'
            check_point_path = iou.getBaseDir() + '/pre_models/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/model.ckpt'

            def initializer_fn(sess):
                init_saver.restore(sess,check_point_path)

            init_fn = initializer_fn
            init_fn(sess)
            print('Coord start done...')

            flst=[]
            import time

            # print('Running for %s' % '_'.join(P_lst))
            # return
            detection_sum=0

            # iter_count=10
            for i in range(iter_count):
                stime=time.time()

                # results = sess.run([img_path_lst,prediction_dict])
                results = sess.run([img_path_lst, rpn_features_to_crop, rett['Conv2d_4a_3x3']])
                run_time = time.time()

                img_path_ret = results[0]
                feats_pre_aux = results[1]
                feats_4 = results[2]
                # print feats_4.shape
                # print feats_pre_aux.shape
                # tsne= TSNE(n_components=2).fit_transform(feats['rpn_features_to_crop'][0].reshape((-1,1088)))
                # pca = PCA(n_components=2)
                # pca.fit(feats['rpn_features_to_crop'][0].reshape((-1,1088)))
                # pca.transform(feats['rpn_features_to_crop'][0].reshape((-1,1088)))



                # print('Total Detection... %i'%detection_sum)

                save_features(img_path_ret,feats_pre_aux,feats_4)
                endtime = time.time()
                print('Completed...%i / %i, tm: %f | %f ' % (i, int(iter_count),run_time-stime,endtime-run_time))

            print('Total %i' % len(flst))
            print('Completed for %s' % '_'.join(P_lst))
            coord.request_stop()
            coord.join(threads)
            print('Creating Session...')

run_inference_for_single_image()