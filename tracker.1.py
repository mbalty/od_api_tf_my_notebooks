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


class DetectionHistory():
    def __init__(self, identifier):
        self.id = identifier
        self.boxes = []
        self.image_patches = []
        self.statuses = []


class ChangeStatus(Enum):
    ADD = 1
    SAME = 4
    REMOVE = 2
    MOVE = 3


class SimilarityModel():
    def __init__(self, keras_model_path = None):
        input_before = Input(shape=(128, 128, 3))
        input_after = Input(shape=(128, 128, 3))
        mobile = MobileNet(input_shape=(128,128,3), weights="imagenet")
        mobile = Sequential(layers=mobile.layers[2:-5])
        before_feature = mobile(input_before)
        after_feature = mobile(input_after)
        distance = Lambda(euclidean_distance,
                        output_shape=eucl_dist_output_shape)([before_feature, after_feature])

        self._model = Model([input_before, input_after], distance)
        self._model.load_weights(keras_model_path)

    def __call__(self, first, second): 
        first = np.expand_dims(cv2.resize(first, (128, 128)), axis=0)
        second = np.expand_dims(cv2.resize(second, (128, 128)), axis=0)
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
    def __init__(self, maxDisappeared=3, iou_thresh = 0.6, similarity_tresh=0.5):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.removedObjects = OrderedDict()
        self.pastSimilarity = 1
        self.imageSimilarity = SimilarityModel(
            keras_model_path =
            "/home/wc-gpu/MasterThesis/models/research/object_detection/od_api_tf_my_notebooks/checkpoint_similar/keep/2019-05-01weights-epoch08-val_acc0.91-val_loss0.10_fixed_val.hdf5") 

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared
        self.iouThresh = iou_thresh
        self.distThresh = similarity_tresh
        self.detectionHistory = dict(boxes=[], classes=[])

    def register(self, box, image):
        # when registering an object we use the next available object
        # ID to store the centroid
        
        history = DetectionHistory(self.nextObjectID)
        history.boxes.append(box)
        history.image_patches.append(crop(image, box))
        history.statuses.append(ChangeStatus.ADD)

        self.objects[self.nextObjectID] = history
        
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
        self.addDetection(box, ChangeStatus.ADD)

    def addDetection(self, box, status):
        self.detectionHistory["boxes"][-1].append(box)
        self.detectionHistory["classes"][-1].append(int(status))

    def track(self, id, box, image, status=ChangeStatus.SAME):
        history = self.objects[id]
        history.boxes.append(box)
        history.image_patches.append(crop(image, box))
        history.statuses.append(status)
        self.disappeared[self.nextObjectID] = 0
        self.addDetection(box, status)


    def remove(self, objectID):
        history = self.objects[objectID]
        history.statuses.append(ChangeStatus.REMOVE)
        self.disappeared[objectID] += 1
        self.addDetection(history.boxes[-1], ChangeStatus.REMOVE)

        if self.disappeared[objectID] > self.maxDisappeared:
            self.deregister(objectID)

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        self.removedObjects[objectID] = self.objects[objectID]
        del self.objects[objectID]
        del self.disappeared[objectID]

    def img_patch_similarity(self, box, image, id):
        cropped = crop(image, box)
        return mean([self.imageSimilarity(cropped, patch) for patch in self.objects[id].image_patches[-self.pastSimilarity:]])

    def update(self, boxes, image=None):
        self.detectionHistory["boxes"].append([])
        self.detectionHistory["classes"].append([])

        # check to see if the list of input bounding box rectangles
        # is empty
        if len(boxes) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in self.disappeared.keys():
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # return early as there are no centroids or tracking info
            # to update
            return self.objects

        # initialize an array of input centroids for the current frame
        boxes = boxes # np.array(boxes, dtype="int")

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(boxes)):
                self.register(boxes[i], image)

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = np.array(list(self.objects.keys()))
            objectBoxes = np.array(list([a.boxes[-1] for a in self.objects.values()]))

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            # D = dist.cdist(np.array(objectBoxes), boxes)


            D = np.empty((len(objectBoxes), len(boxes)), dtype=np.float32)

            for i in range(len(objectBoxes)):
                for j in range(len(boxes)):
                    D[i, j] = 1 - iou(objectBoxes[i], boxes[j])
            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols or D[row][col] > self.iouThresh:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                
                objectID = objectIDs[row]
                self.track(objectID, boxes[col], image)

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            # loop over the unused row indexes

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
                    if row in usedRows or col in usedCols or D[row][col] > self.distThresh:
                        continue
                    objectID = objectIDs[row]
                    self.track(objectID, boxes[col], image, ChangeStatus.MOVE)

                    usedRows.add(row)
                    usedCols.add(col)

                unusedRows = set(range(0, D.shape[0])).difference(usedRows)
                unusedCols = set(range(0, D.shape[1])).difference(usedCols)


            for row in unusedRows:
                # grab the object ID for the corresponding row
                # index and increment the disappeared counter
                self.remove(objectIDs[row])

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            for col in unusedCols:
                self.register(boxes[col], image)

        # return the set of trackable objects
        return self.objects

