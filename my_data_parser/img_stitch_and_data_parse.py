import xml.etree.ElementTree as ET
import os
from image_label import *
from PIL import Image, ImageFont, ImageDraw
import numpy as np


def drawRectangles(image_lbl):
    image = Image.open(image_lbl.path)
    font = ImageFont.truetype(font='/Library/Fonts/Arial.ttf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300


    for rect_label in image_lbl.rectLabels:
        label = rect_label.label
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = rect_label.ymin, rect_label.xmin, rect_label.ymax, rect_label.xmax
        # top = max(0, np.floor(top + 0.5).astype('int32'))
        # left = max(0, np.floor(left + 0.5).astype('int32'))
        # bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        # right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

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
    image.save(image_lbl.path)

def stackImages(first_path, second_path):
    imgs = [Image.open(i) for i in [first_path, second_path]]
    img_merge = np.vstack( (np.asarray( i.resize(imgs[0].size,Image.ANTIALIAS) ) for i in imgs ) )
    img_merge = Image.fromarray(img_merge)
    return img_merge


def convertRectLabelFile(fpath: str) :
    fpath = os.path.abspath(fpath)
    tree = ET.parse(fpath)
    root = tree.getroot()

    img_path = os.path.abspath(os.path.join(os.path.dirname(fpath), "..", root.find("filename").text))
    img_label = ImageLabel(img_path)
    size = root.find("size")
    img_label.width = int(size.find("width").text)
    img_label.height = int(size.find("height").text)
    for child in root.findall("object"):
        name = child.find("name").text

        bndbox = child.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        img_label.addRectLabel(name, xmin, ymin, xmax, ymax)

    return img_label, root.find("filename").text


folder = "/home/wc-gpu/MasterThesis/session_data"
out_img_folder = "/home/wc-gpu/MasterThesis/session_data_stacked"


labels = dict()

print("make_dict")
c = 0
for session_dir in sorted(os.listdir(folder)):
    print (c/len(os.listdir(folder)), end="\r")
    c += 1
    session_dir = os.path.join(folder, session_dir)
    annotations_dir = os.path.join(session_dir, "annotations")
    if os.path.exists(annotations_dir):
        for annotation_file in sorted(os.listdir(annotations_dir)):
            annotation_file = os.path.join(annotations_dir, annotation_file)
            label, key = convertRectLabelFile(annotation_file)
            labels[key] = label

labels_list = []
print("stitch 1")
c = 0

for session_dir in sorted(os.listdir(folder)):
    print(c / len(os.listdir(folder)), end="\r")
    c += 1
    session_dir_glob = os.path.join(folder, session_dir)
    first_img = None
    annotations_dir = os.path.join(session_dir_glob, "annotations")
    if os.path.exists(annotations_dir) and os.path.isdir(session_dir_glob):
        for second_img in sorted(os.listdir(session_dir_glob)):
            if second_img.endswith(".jpg"):
                if first_img:
                    # merged = stackImages(os.path.join(session_dir_glob, first_img), os.path.join(session_dir_glob, second_img))
                    first_label = labels.get(first_img)
                    second_label = labels.get(second_img)
                    merged_img_path =  os.path.join(out_img_folder, session_dir,
                                                    str(first_img.split(".jpg")[0]) + "_" + str(second_img.split(".jpg")[0]) + ".jpg")
                    if not os.path.exists(os.path.dirname(merged_img_path)):
                        os.makedirs(os.path.dirname(merged_img_path))
                    # merged.save(merged_img_path)

                    label = ImageLabel(merged_img_path)
                    if first_label:
                        for r in first_label.rectLabels:
                            if "rem" in r.label:
                                new_r = RectLabel("remove", r.xmin, r.ymin, r.xmax, r.ymax)
                                label.rectLabels.append(new_r)


                    if second_label:
                        for r in second_label.rectLabels:
                            if "add" in r.label:
                                new_r = RectLabel("add", r.xmin, r.ymin, r.xmax, r.ymax)
                                # new_r.ymin += second_label.height
                                # new_r.ymax += second_label.height

                                label.rectLabels.append(new_r)



                    labels_list.append(label)

                    # drawRectangles(label)

                first_img = second_img


print("stitch 2")
c = 0
for session_dir in sorted(os.listdir(folder))[::-1]:
    print(c / len(os.listdir(folder)), end="\r")
    c += 1
    session_dir_glob = os.path.join(folder, session_dir)
    first_img = None
    annotations_dir = os.path.join(session_dir_glob, "annotations")
    if os.path.exists(annotations_dir) and os.path.isdir(session_dir_glob):
        for second_img in sorted(os.listdir(session_dir_glob))[::-1]:
            if second_img.endswith(".jpg"):
                if first_img:
                    # merged = stackImages(os.path.join(session_dir_glob, first_img), os.path.join(session_dir_glob, second_img))
                    first_label = labels.get(first_img)
                    second_label = labels.get(second_img)
                    merged_img_path =  os.path.join(out_img_folder, session_dir,
                                                    str(first_img.split(".jpg")[0]) + "_" + str(second_img.split(".jpg")[0]) + ".jpg")
                    if not os.path.exists(os.path.dirname(merged_img_path)):
                        os.makedirs(os.path.dirname(merged_img_path))
                    # merged.save(merged_img_path)

                    label = ImageLabel(merged_img_path)
                    if first_label:
                        for r in first_label.rectLabels:
                            if "add" in r.label:
                                new_r = RectLabel("remove", r.xmin, r.ymin, r.xmax, r.ymax)
                                label.rectLabels.append(new_r)

                    if second_label:
                        for r in second_label.rectLabels:
                            if "rem" in r.label:
                                new_r = RectLabel("add", r.xmin, r.ymin, r.xmax, r.ymax)
                                # new_r.ymin += second_label.height
                                # new_r.ymax += second_label.height
                                label.rectLabels.append(new_r)

                    labels_list.append(label)
                    # drawRectangles(label)
                first_img = second_img

f = open('train_sessions_stitched.txt', 'w')
count  = 0
session=""
past_k = 0

classes = ["add", "remove"]
print("make_labels")

for label in labels_list:
    print(c / len(labels_list), end="\r")
    c += 1
    file_line = label.path

    for rect in label.rectLabels:
        x_min = rect.xmin
        y_min = rect.ymin
        x_max = rect.xmax
        y_max = rect.ymax

        file_line += " %d,%d,%d,%d,%d" % (
            x_min, y_min, x_max, y_max, classes.index(rect.label))



    f.write(file_line)
    f.write('\n')
