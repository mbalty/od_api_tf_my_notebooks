# import the necessary packages
from collections import OrderedDict
import numpy as np
import cv2
import json
import math


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

def dist(c1, c2):
    return math.sqrt((math.pow(c1[0]-c2[0], 2.) + math.pow(c1[1]-c2[1], 2)))

def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

class IoUTracker():
    def __init__(self, maxDisappeared=5, maxDetectionPatched=10):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.detectionPatched = OrderedDict()

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared
        self.maxDetectionPatched = maxDetectionPatched
        self.cvTracker = cv2.TrackerKCF_create
        self.patchingTrackers = dict()
        self.initPatchingTracker = dict()
        self.prevImage = None
        self.history = list()
        self.historyFile = open("./history.json", "w")
        self.historyFile.close()
        self.historyFile = open("./history.json", "w+")
        self.historyFile.write("[\n")
        self.imw = None
        self.imh = None
        self.distTrackTresh = .15

    def register(self, centroid, data=None):
        # when registering an object we use the next available object
        # ID to store the centroid
        print("add", self.nextObjectID)
        self.objects[self.nextObjectID] = [(centroid, data)]
        self.disappeared[self.nextObjectID] = 0
        self.detectionPatched[self.nextObjectID] = 0
        self.patchingTrackers[self.nextObjectID] = [self.cvTracker(), False]
        self.nextObjectID += 1

    def deregister(self, objectID):
        print("REMOVE", objectID)
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        obj = dict(
            id=int(objectID),
            data=[
                dict(
                    cls = int(a[1]["cls"]),
                    score = float(a[1]["score"]),
                    box = dict(
                        y1=float(a[1]["box"][0]),
                        x1=float(a[1]["box"][1]),
                        y2=float(a[1]["box"][2]),
                        x2=float(a[1]["box"][3])
                    )
                ) for a in self.objects[objectID]])
        self.historyFile.write(json.dumps(obj))
        self.historyFile.write(",\n")
        self.history.append(obj)

        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.detectionPatched[objectID]
        del self.patchingTrackers[objectID]

    def cv_tracker_get_box_and_set(self, objectID, image):
        try:
            self.detectionPatched[objectID] += 1
            c, data = self.objects[objectID][-1]
            (startY, startX, endY, endX) = data["box"]
            init_bb = (startY, startX, endY-startY, endX-startX)
            tracker, trackerInit = self.patchingTrackers[objectID]
            if not trackerInit:
                tracker.init(self.prevImage, init_bb)
                self.patchingTrackers[objectID] = [tracker, True]

            (success, new_box) = tracker.update(image)
            if success:
                (startY, startX, endY, endX) = new_box
                endX += startX
                endY += startY

                new_data = data.copy()
                new_data["box"] = [startY, startX, endY, endX]

                iou_ = iou(data["box"], new_data["box"])
                dist_ = dist(((data["box"][0] + data["box"][2]) / 2. / self.imh,
                              (data["box"][1] + data["box"][3]) / 2. / self.imw),
                             ((new_data["box"][0] + new_data["box"][2]) / 2. / self.imh,
                              (new_data["box"][1] + new_data["box"][3]) / 2. / self.imw))
                if iou_ > 0 or dist_ < self.distTrackTresh:
                    self.objects[objectID].append(((startY, startX, endY, endX), new_data))
                    self.disappeared[objectID] = 0
                else:
                    success = False

            if not success:
                self.disappeared[objectID] += 1

            return success
        except Exception as ex:
            print (ex)
            return False

    def update(self, rects, data=None, image=None):
        # check to see if the list of input bounding box rectangles
        # is empty
        if image is not None:
            self.imh = image.shape[0]
            self.imw = image.shape[1]
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in self.disappeared.keys():
                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                self.cv_tracker_get_box_and_set(objectID, image)
                if self.disappeared[objectID] > self.maxDisappeared or \
                                self.detectionPatched[objectID] > self.maxDetectionPatched:
                    self.deregister(objectID)

            # return early as there are no centroids or tracking info
            # to update
            self.prevImage = image
            return self.objects

        # initialize an array of input centroids for the current frame
        inputBoxes = rects # np.array(rects, dtype="int")

        # # loop over the bounding box rectangles
        # for (i, (startY, startX, endY, endX)) in enumerate(rects):
        #     # use the bounding box coordinates to derive the centroid
        #     cX = int((startX + endX) / 2.0)
        #     cY = int((startY + endY) / 2.0)
        #     inputCentroids[i] = (cX, cY)

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputBoxes)):
                if data is None:
                    self.register(inputBoxes[i])
                else:
                    self.register(inputBoxes[i], data[i])



        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectBoxes = list([a[-1][0] for a in self.objects.values()])
            objectCentroids = np.array(
                [[(a[0] + a[2]) / 2. / self.imh, (a[1] + a[3]) / 2. / self.imw] for a in objectBoxes])
            inputCentroids = np.array(
                [[(a[0] + a[2]) / 2. / self.imh, (a[1] + a[3]) / 2. / self.imw] for a in inputBoxes])

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid

            D = np.empty((len(objectBoxes), len(inputBoxes)), dtype=np.double)


            for i in range(len(objectBoxes)):
                for j in range(len(inputBoxes)):
                    D[i, j] = 1 - iou(objectBoxes[i], inputBoxes[j])
                    if D[i, j] == 1:
                        ds = dist(objectCentroids[i], inputCentroids[j])
                        if ds > self.distTrackTresh:
                            ds = 1
                        D[i, j] += ds
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
                if row in usedRows or col in usedCols or D[row, col] == 2:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID].append((inputBoxes[col], data[col]))
                self.disappeared[objectID] = 0
                self.detectionPatched[objectID] = 0
                tracker, trackerInit = self.patchingTrackers[objectID]
                if trackerInit:
                    tracker.clear()
                    self.patchingTrackers[objectID] = [tracker, False]

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

            print(D.shape)
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    self.cv_tracker_get_box_and_set(objectID, image)
                    if self.disappeared[objectID] > self.maxDisappeared or \
                                    self.detectionPatched[objectID] > self.maxDetectionPatched:
                        self.deregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputBoxes[col], data[col])

        # return the set of trackable objects
        self.prevImage = image
        return self.objects
