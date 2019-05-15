from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
from enum import Enum
from keras.models import load_model
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential
from keras.layers import Concatenate, Dense, Input, Conv2D, Lambda, MaxPool2D, Flatten
import keras.backend as K
import tensorflow as tf
import cv2

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union

def sift_sim(img_a, img_b):
  '''
  Use SIFT features to measure image similarity
  @args:
    {str} path_a: the path to an image file
    {str} path_b: the path to an image file
  @returns:
    TODO
  '''
  # initialize the sift feature detector
  orb = cv2.ORB_create()

  # get the images

  # find the keypoints and descriptors with SIFT
  kp_a, desc_a = orb.detectAndCompute(img_a, None)
  kp_b, desc_b = orb.detectAndCompute(img_b, None)

  # initialize the bruteforce matcher
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

  # match.distance is a float between {0:100} - lower means more similar
  matches = bf.match(desc_a, desc_b)
  similar_regions = [i for i in matches if i.distance < 70]
  if len(matches) == 0:
    return 0
  return 1. - len(similar_regions) / len(matches)


class DetectionHistory():
    def __init__(self, identifier):
        self.id = identifier
        self.boxes = []
        self.image_patches = []
        self.statuses = []
        self.scores = []

    def add(self, box, patch, status, score):
        self.boxes.append(box)
        self.image_patches.append(patch)
        self.statuses.append(status)
        self.scores.append(score)

class ChangeStatus(Enum):
    ADD = 1
    SAME = 4
    REMOVE = 2
    MOVE = 3

class SimilarityModel():
    def __init__(self, keras_model_path = None):
        input_before = Input(shape=(128, 128, 3))
        input_after = Input(shape=(128, 128, 3))
        mobile = MobileNet(input_shape=(128,128,3), weights=None)
        mobile = Sequential(layers=mobile.layers[2:-5])
        before_feature = mobile(input_before)
        after_feature = mobile(input_after)
        distance = Lambda(euclidean_distance,
                        output_shape=eucl_dist_output_shape)([before_feature, after_feature])

        self._model = Model([input_before, input_after], distance)
        self._model.load_weights(keras_model_path)

    def __call__(self, first, second): 
        first = cv2.resize(first, (128, 128)).astype(np.float32)
        second = cv2.resize(second, (128, 128)).astype(np.float32)
        try:
            sift_similarity = sift_sim(first[0], second[0])
        except:
            sift_similarity = 0.5
        first = np.expand_dims(first, axis=0)
        second = np.expand_dims(second, axis=0)
        # first -= np.mean(first, axis = 0)
        # second -= np.mean(second, axis = 0)
        return self._model.predict([first, second])[0]


def crop(image, box):
    h, w, _ = image.shape
    return image[int(box[0]*h):int(box[2]*h), 
                 int(box[1]*w):int(box[3]*w)]

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

class ChangeTracker():
    is_initalized = False
    def __init__(self, maxDisappeared=3, iou_thresh =.4, similarity_tresh=0.4, past_similarity=3):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.removedObjects = OrderedDict()
        self.pastSimilarity = past_similarity
        self.imageSimilarity = SimilarityModel(
            keras_model_path =
            "/home/wc-gpu/MasterThesis/models/research/object_detection/od_api_tf_my_notebooks/checkpoint_similar/keep/2019-05-15weights-epoch15-val_acc0.90-val_loss0.02_all_add_l2.hdf5")  \
                if ChangeTracker.is_initalized is False else self.imageSimilarity
        self.maxDisappeared = maxDisappeared
        self.iouThresh = iou_thresh
        self.similarityThresh = similarity_tresh
        self.detectionHistory = dict(boxes=[], classes=[], ids=[])
        self.images = []

        ChangeTracker.is_initalized = True

    def reset(self):
        ChangeTracker.__init__(self, self.maxDisappeared, self.iouThresh, self.similarityThresh)

    def register(self, box, image):
        # when registering an object we use the next available object
        # ID to store the centroid
        
        # if len(self.images) and self.imageSimilarity(crop(image, box), crop(self.images[-1], box)) < self.similarityThresh:
        #     return
        history = DetectionHistory(self.nextObjectID)
        history.add(box, crop(image, box), ChangeStatus.ADD, 0)
        self.objects[self.nextObjectID] = history
        
        self.disappeared[self.nextObjectID] = 0
        self.addDetection(box, ChangeStatus.ADD, self.nextObjectID)
        self.nextObjectID += 1

    def addDetection(self, box, status, objectID):
        self.detectionHistory["boxes"][-1].append(box)
        self.detectionHistory["classes"][-1].append(status.value)
        self.detectionHistory["ids"][-1].append(objectID)

    def track(self, objectID, box, image, status=ChangeStatus.SAME, score=0): 
        history = self.objects[objectID]
        history.add(box, crop(image, box), status, score)

        self.disappeared[self.nextObjectID] = 0
        self.addDetection(box, status, objectID)

    def remove(self, objectID):
        history = self.objects[objectID]
        history.statuses.append(ChangeStatus.REMOVE)
        self.disappeared[objectID] += 1
        if history.statuses[-2] != ChangeStatus.REMOVE:
            self.addDetection(history.boxes[-1], ChangeStatus.REMOVE, objectID)

        if self.disappeared[objectID] > self.maxDisappeared:
            self.deregister(objectID)

    def deregister(self, objectID):
        self.removedObjects[objectID] = self.objects[objectID]
        del self.objects[objectID]
        del self.disappeared[objectID]

    def img_patch_similarity(self, box, image, id):
        cropped = crop(image, box)
        patches = self.objects[id].image_patches
        if self.pastSimilarity < len(patches):
            patches = patches[-self.pastSimilarity:]
        return mean([self.imageSimilarity(cropped, patch) for patch in patches])

    def update(self, boxes, image=None):
        self.detectionHistory["boxes"].append([])
        self.detectionHistory["classes"].append([])
        self.detectionHistory["ids"].append([])

        if len(boxes) == 0:
            objIds = list(self.disappeared.keys())
            for objectID in objIds:
                self.remove(objectID)

            self.images.append(image)
            return self.objects

        if len(self.objects) == 0:
            for i in range(0, len(boxes)):
                self.register(boxes[i], image)

        else:
            objectIDs = np.array(list(self.objects.keys()))
            objectBoxes = np.array(list([a.boxes[-1] for a in self.objects.values()]))

            D = np.empty((len(objectBoxes), len(boxes)), dtype=np.float32)

            for i in range(len(objectBoxes)):
                for j in range(len(boxes)):
                    D[i, j] = 1 - iou(objectBoxes[i], boxes[j])
            rows = D.min(axis=1).argsort()

            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols or D[row][col] > self.iouThresh:
                    continue

                objectID = objectIDs[row]
                self.track(objectID, boxes[col], image, score=D[row][col])

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if len(unusedRows) and len(unusedCols):
                patchRows = np.array(sorted(list(unusedRows)), dtype=np.int16)
                patchCols = np.array(sorted(list(unusedCols)), dtype=np.int16)
                objectIDs = objectIDs[patchRows]
                objectBoxes = objectBoxes[patchRows]
                boxes = boxes[patchCols]
                D = D[patchRows, :]
                D = D[:, patchCols]

                for i in range(len(objectBoxes)):
                    for j in range(len(boxes)):
                        D[i, j] = self.img_patch_similarity(boxes[j], image, objectIDs[i])


                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]
                usedRows = set()
                usedCols = set()

                for (row, col) in zip(rows, cols):
                    if row in usedRows or col in usedCols or D[row][col] > self.similarityThresh:
                        continue
                    objectID = objectIDs[row]
                    self.track(objectID, boxes[col], image, ChangeStatus.MOVE, D[row][col])

                    usedRows.add(row)
                    usedCols.add(col)

                unusedRows = set(range(0, D.shape[0])).difference(usedRows)
                unusedCols = set(range(0, D.shape[1])).difference(usedCols)


            for row in unusedRows:
                self.remove(objectIDs[row])

            for col in unusedCols:
                self.register(boxes[col], image)
        self.images.append(image)
        return self.objects

