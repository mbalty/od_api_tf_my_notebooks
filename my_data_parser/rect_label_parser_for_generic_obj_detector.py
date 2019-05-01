import xml.etree.ElementTree as ET
import os
from image_label import *
from PIL import Image, ImageFont, ImageDraw
import numpy as np

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

def convertRectLabelFile(fpath: str) :
    fpath = os.path.abspath(fpath)
    tree = ET.parse(fpath)
    root = tree.getroot()

    img_path = os.path.abspath(os.path.join(os.path.dirname(fpath), "..", root.find("filename").text))
    img_label = ImageLabel(img_path)
    for child in root.findall("object"):
        name = child.find("name").text
        bndbox = child.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        img_label.addRectLabel(name, xmin, ymin, xmax, ymax)

    return img_label


folder = "/home/wc-gpu/MasterThesis/session_data"

labels = []

for session_dir in sorted(os.listdir(folder)):
    session_dir = os.path.join(folder, session_dir)
    annotations_dir = os.path.join(session_dir, "annotations")
    if os.path.exists(annotations_dir):
        for annotation_file in sorted(os.listdir(annotations_dir)):
            annotation_file = os.path.join(annotations_dir, annotation_file)
            label = convertRectLabelFile(annotation_file)
            labels.append(label)


f = open('train_sessions.txt', 'w')
count  = 0
session=""
past_k = 0

for i in range(len(labels)):
    # print(str(count) + "\r")

    file_line = labels[i].path
    hasRectangles = False
    if session != file_line.split("/")[5]:
        session = file_line.split("/")[5]
        past_k = 0

    past_k += 1
    past_k = min(past_k, 10)
    for k in range(past_k):
        j = i-k
        if j >= 0:
            label = labels[j]
            for rect in label.rectLabels:
                if rect.label == "add" or ("move" in rect.label[:4] and "from" not in rect.label) or (k == 0 and "remo" in rect.label) or (k == 0 and "from" in rect.label):
                    count += 1
                    hasRectangles = True
                    x_min = rect.xmin
                    y_min = rect.ymin
                    x_max = rect.xmax
                    y_max = rect.ymax

                    file_line += " %d,%d,%d,%d,%d" % (
                        x_min, y_min, x_max, y_max, 0)

                    if ("move" in rect.label[:4] and "from" not in rect.label):
                        print(rect.label, k)
                        past_k = k
                        break
                elif ("remo" in rect.label) or ("from" in rect.label):
                    print(rect.label, k)
                    past_k = k
                    break
                else:
                    break


    if hasRectangles:
        f.write(file_line)
        f.write('\n')
print (count)
f.close()
