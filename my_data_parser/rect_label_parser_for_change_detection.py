import xml.etree.ElementTree as ET
import os
from image_label import *
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import random


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


def get_move_idx(lb):
    try:
        index = int(lb[-2:])
    except:
        try:
            index = int(lb[-1:])
        except:
            index = -1

    return index


class DataGenerator:
    def __init__(self, data_path, output_path):
        self.data_path = data_path
        self.output_path = output_path
        self.labels = list()
        self.non_labels = list()
        self.lines = list()
        self.objIndex = 0
        self.countAnnotatedImages = 0
        self.countAnnotations = dict(move=0, move_from = 0, add = 0, remove = 0, garbage=[])

    @staticmethod
    def drawRectangles(image_lbl):
        image = Image.open(image_lbl.path)

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300


        for rect_label in image_lbl.rectLabels:
            label = rect_label.label
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = rect_label.xmin, rect_label.ymin, rect_label.xmax, rect_label.ymax
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            if label == "add":
                color = (255, 0, 0)
            elif label == "remove":
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=color)
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=color)
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

    @staticmethod
    def convertRectLabelFile(fpath: str) :
        fpath = os.path.abspath(fpath)
        tree = ET.parse(fpath)
        root = tree.getroot()

        img_path = os.path.abspath(os.path.join(os.path.dirname(fpath), "..", root.find("filename").text))
        img_label = ImageLabel(img_path, root.find("filename").text)

        for child in root.findall("object"):
            name = child.find("name").text
            bndbox = child.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

            img_label.addRectLabel(name, xmin, ymin, xmax, ymax)

        return img_label

    def checkLabel(self, label):
        new_rect_labels = []
        for rect in label.rectLabels:
            l = rect.label
            if l != "add" and l != "remove" and l != "first" and l != "same" and not l.startswith("move"):
                self.non_labels.append((l, label.path))
            else:
                new_rect_labels.append(rect)

        label.rectLabels = new_rect_labels



    def cropBoxes(self, before, after, session):
        image_before = Image.open(os.path.join(before.path))
        image_after = Image.open(os.path.join(after.path))
        for lb in before.rectLabels:
            rb = (lb.xmin, lb.ymin, lb.xmax, lb.ymax)
            crop = True
            ra = None

            if lb.label == "add":
                for la in after.rectLabels:
                    ra = (la.xmin, la.ymin, la.xmax, la.ymax)
                    if (la.label == "add" or (la.label.startswith("move") and not la.label.startswith("move_from"))) and \
                        intersection_over_second_box(ra, rb) > 0.3:
                        crop = False
                    if not crop:
                        break

                ra = rb
            elif lb.label.startswith("move_from"):
                idxb = get_move_idx(lb.label)
                if idxb < 0:
                    crop = False
                else:
                    crop = False
                    for la in after.rectLabels:
                        if la.label.startswith("move") and not la.label.startswith("move_from") and idxb == get_move_idx(la.label):
                            ra = (la.xmin, la.ymin, la.xmax, la.ymax)
                            crop = True
                            break
            else:
                crop = False

            if crop:
                try:
                    cropped_before = image_before.crop(rb)
                    cropped_after = image_after.crop(ra)

                    cropped_before.save(os.path.join(self.output_path, str(self.objIndex) + "-before-" + lb.label + ".jpg"))
                    cropped_after.save(os.path.join(self.output_path, str(self.objIndex) + "-after-" + lb.label + ".jpg"))
                    self.objIndex += 1
                except Exception as ex:
                    print(ex)


    def generate_one_session(self, session):
        session_dir = os.path.join(self.data_path, session)
        annotations_dir = os.path.join(session_dir, "annotations")

        labels = list()
        if os.path.exists(annotations_dir):
            for image_fn in sorted(os.listdir(session_dir)):
                if image_fn.endswith(".jpg"):
                    annotation_file = os.path.splitext(image_fn)[0] + ".xml"
                    annotation_file = os.path.join(annotations_dir, annotation_file)
                    image_fpath = os.path.join(session_dir, image_fn)
                    if os.path.exists(annotation_file):
                        label = DataGenerator.convertRectLabelFile(annotation_file)
                    else:
                        label = ImageLabel(image_fpath, image_fn)

                    labels.append(label)

            for l in labels:
                if len(l.rectLabels) > 0:
                    self.countAnnotatedImages += 1
                    for r in l.rectLabels:
                        if "add" in r.label:
                            self.countAnnotations["add"] += 1
                        elif r.label == "remove":
                            self.countAnnotations["remove"] += 1
                        elif r.label.startswith("move_from"):
                            self.countAnnotations["move_from"] += 1
                        elif r.label.startswith("move"):
                            self.countAnnotations["move"] += 1
                        else:
                            self.countAnnotations["garbage"].append(r.label)



                            # for i in range(1, len(labels)):
            #     before = labels[i - 1]
            #     after = labels[i]
            #     self.cropBoxes(before, after, session)
            #
            #     # self.generate_one_output_row(before, after)
            #     # self.generate_one_output_row(after, before, reverted=True)



    def generate_one_output_row(self, before, after, reverted=False):
        if len(after.rectLabels) and len(before.rectLabels):
            file_line = before.path + " " + after.path

            add_label = "remove" if reverted else "add"
            remove_label = "add" if reverted else "remove"
            move_label = "move_from" if reverted else "move"
            move_from_label = "move" if reverted else "move_from"

            for rect in after.rectLabels:
                if rect.label == add_label:
                    x_min = rect.xmin
                    y_min = rect.ymin
                    x_max = rect.xmax
                    y_max = rect.ymax

                    file_line += " %d,%d,%d,%d,%d" % (x_min, y_min, x_max, y_max, 0)

            # for rect in before.rectLabels:
            #     if rect.label == remove_label:
            #         x_min = rect.xmin
            #         y_min = rect.ymin
            #         x_max = rect.xmax
            #         y_max = rect.ymax
            #
            #         file_line += " %d,%d,%d,%d,%d" % (x_min, y_min, x_max, y_max, 1)

            self.lines.append(file_line)
            # self.output_file.write(file_line)
            # self.output_file.write("\n")

    def generate(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        for session_dir in sorted(os.listdir(self.data_path)):
            try:
                a = int(session_dir)
            except:
                continue
            self.generate_one_session(session_dir)

        # random.shuffle(self.lines)
        # self.output_file = open(self.output_path, "w")
        # self.output_file.write("\n".join(self.lines))
        # self.output_file.close()



data_folder = "/home/wc-gpu/storage4tb/session_data_thesis/sessions160000_165000"
output_ann_file = '/home/wc-gpu/storage4tb/session_data_thesis/similar_boxes/'
# data_folder = "/Users/mbaltac/Home/MasterThesis/MasterThesis/labeled_images_shopping_sessions/labeled"
# output_ann_file = '/Users/mbaltac/Home/MasterThesis/MasterThesis/labeled_images_shopping_sessions/similarity'


dg = DataGenerator(data_folder, output_ann_file)
dg.generate()
print (dg.countAnnotations)
print (len(dg.countAnnotations["garbage"
                               ""]))
print (dg.countAnnotatedImages)
import pickle
with open("count.annotations", "wb") as f:
    pickle.dump(dg.countAnnotations, f)


