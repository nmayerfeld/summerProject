import numpy as np
import os
import PIL
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.models import load_model
import sys
import cv2
import tensorflow_datasets as tfds
import os as os
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt
import random

def width_overlap(bbox_x1, bbox_x2, ss_x1, ss_x2):
    if bbox_x1 == ss_x1 and bbox_x2 == ss_x2:
        return ss_x2 - ss_x1
    elif bbox_x1 > ss_x2 or ss_x1 > bbox_x2:
        return 0
    elif bbox_x1 >= ss_x1 and bbox_x1 < ss_x2 and bbox_x2 >= ss_x2:
        return ss_x2 - bbox_x1 + 1
    elif bbox_x1 < ss_x1 and bbox_x2 > ss_x2:
        return ss_x2 - ss_x1
    elif ss_x1 < bbox_x1 and ss_x2 > bbox_x2:
        return bbox_x2 - bbox_x1
    else:
        return bbox_x2 - ss_x1 + 1

def height_overlap(bbox_y1, bbox_y2, ss_y1, ss_y2):
    if bbox_y1 == ss_y1 and bbox_y2 == ss_y2: #The lines are identical and have complete overlap.
        return ss_y2 - ss_y1
    elif bbox_y1 > ss_y2 or ss_y1 > bbox_y2: #There is no overlap between them at al.
        return 0
    elif bbox_y1 >= ss_y1 and bbox_y1 < ss_y2 and bbox_y2 >= ss_y2: #Partial overlap, where the selective search box starts lower.
        return ss_y2 - bbox_y1 + 1
    elif bbox_y1 < ss_y1 and bbox_y2 > ss_y2: #Bounding box completely encompasses the selective search box. 
        return ss_y2 - ss_y1
    elif ss_y1 < bbox_y1 and ss_y2 > bbox_y2: #Selective search box completely encompasses the bounding box. 
        return bbox_y2 - bbox_y1
    else: #Partial overlap, where the bounding box starts lower. 
        return bbox_y2 - ss_y1 + 1

def areDisjoint(set1, set2, m, n): 
    # Take every element of set1[] and search it in set2 
    for i in range(0, m): 
        for j in range(0, n): 
            if (set1[i] == set2[j]): 
                return False
  
    # If no element of set1 is present in set2 
    return True

try:
   os.mkdir("sortedPics")
   os.mkdir("sortedPics/forTesting")
except OSError as error:
   pass

try:
   os.mkdir("sortedPics/forTraining")
except OSError as error:
   pass

keys = ['cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe']
numbers = [15,16,17,18,19,20,21,22,23]
dict = {}

for item in keys:
   try:
      os.mkdir("sortedPics/forTraining/"+item)
   except OSError as error:
      continue

cv2.setUseOptimized(True)
cv2.setNumThreads(4)


coco_data = tfds.load('coco')
for split in coco_data.keys():
    data_set = coco_data[split]
    for num,example in enumerate(data_set):
        scores = []
        coordinates = []
        print("Picture number: " + str(num))
        save = False
        image = example['image']
        im = image.numpy()
        copy = im.copy()
        copy = Image.fromarray(copy)
        labels = example['objects']['label']
        bboxes = example['objects']['bbox']
        id = example['image/id'].numpy()
        labelNums = labels.numpy()
        if areDisjoint(labelNums, numbers, len(labelNums), len(numbers)):
            continue
        else:
            if random.randint(0, 100) < 4:
                filename = "sortedPics/forTesting/"+ str(id) + ".jpg"
                copy.save(filename)
            else:
                shape = image.shape
                #draw = ImageDraw.Draw(copy)
                for count,label in enumerate(labelNums):
                    if label in numbers:
                        save = True
                        # bbox = bboxes[count].numpy()
                        # top, left, bottom, right = bbox[0] * shape[0], bbox[1] * shape[1], bbox[2] * shape[0], bbox[3] * shape[1] #this is a tensorflow box of an animal
                        # if (right - left + 1) * (bottom - top + 1) < 3600:
                        #     continue
                        # else:
                        #     draw.rectangle([left, top, right, bottom], outline = "blue")
                if save:
                    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
                    ss.setBaseImage(im)
                    ss.switchToSelectiveSearchFast()
                    rects = ss.process()
                    print('Total Number of Region Proposals: {}'.format(len(rects)))
                    for idx, rect in enumerate(rects):
                        left2, top2, width, height = rect
                        right2 = width + left2 - 1
                        bottom2 = top2 + height - 1
                        for count,label in enumerate(labelNums):
                            if label in numbers:
                                bbox = bboxes[count].numpy()
                                top, left, bottom, right = bbox[0] * shape[0], bbox[1] * shape[1], bbox[2] * shape[0], bbox[3] * shape[1] #this is a tensorflow box of an animal
                                if (right - left + 1) * (bottom - top + 1) < 6400:
                                    break
                                skip = False
                                for index, second_label in enumerate(labelNums):
                                    if second_label in numbers:
                                        bbox = bboxes[index].numpy()
                                        top3, left3, bottom3, right3 = bbox[0] * shape[0], bbox[1] * shape[1], bbox[2] * shape[0], bbox[3] * shape[1] #this is a tensorflow box of an animal
                                        if top == top3 and left == left3 and bottom == bottom3 and right == right3:
                                            continue
                                        else:
                                            if top3 > top and top3 < bottom and bottom3 > top and bottom3 < bottom and left3 > left and left3 < right and right3 > left and right3 < right and label != second_label:
                                                skip = True
                                if skip:
                                    continue
                                else:
                                    #draw.rectangle([left, top, right, bottom], outline = "blue")     
                                    inter_width = width_overlap(left, right, left2, right2)
                                    inter_height = height_overlap(top, bottom, top2, bottom2)
                                    # compute the area of intersection rectangle
                                    interArea = inter_width * inter_height
                                    # compute the area of both the prediction and ground-truth
                                    # rectangles
                                    boxAArea = (right - left + 1) * (bottom - top + 1)
                                    boxBArea = (right2 - left2 + 1) * (bottom2 - top2 + 1)
                                    # compute the intersection over union by taking the intersection
                                    # area and dividing it by the sum of prediction + ground-truth
                                    # areas - the interesection area
                                    iou = interArea / float(boxAArea + boxBArea - interArea) 
                                    iou = round(iou * 100)
                                    if iou > 60:
                                        #draw.text((left2 + 3, top2 + 3), str(iou), color = "green")
                                        #draw.rectangle([left2, top2, right2, bottom2], outline = "red")
                                        #draw.text((left2 + 10, top2 + 10), "top:" + str(top2) + ", bottom:" + str(bottom2) + ", left:" + str(left2) + ", right:" + str(right2), color = "orange")
                                        print(iou)
                                        scores.append(float(iou))
                                        coordinates.append([float(top2), float(left2), float(bottom2), float(right2)])
                                        dict[top2, left2, bottom2, right2] = (iou, label)
                    max_output_size = 3 #how many more boxes do we want per animal? Probably just like 2
                    iou_threshold = .6
                    print(coordinates)
                    print(scores)
                    scores = tf.convert_to_tensor(scores)
                    coordinates = tf.convert_to_tensor(coordinates)
                    print(coordinates)
                    if(len(scores) > 0):
                        selected_indices = tf.image.non_max_suppression(coordinates, scores, max_output_size, iou_threshold, score_threshold = .8) #how strongly do we want our new boxes to overlap with the original tensorflow animal box
                        selected_boxes = tf.gather(coordinates, selected_indices)
                        for index, box in enumerate(selected_boxes):
                            box = box.numpy()
                            top, left, bottom, right = box[0], box[1], box[2], box[3]
                            newcopy = copy.copy()
                            img = newcopy.crop((left, top, right, bottom))
                            filename="sortedPics/forTraining/"+keys[dict[top, left, bottom, right][1] - 15]+ "/" + str(id) + "Crop" + str(index) +".jpg"
                            img.save(filename)