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

keys = ['cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe']
numbers = [15,16,17,18,19,20,21,22,23]

cv2.setUseOptimized(True)
cv2.setNumThreads(4)

try:
   os.mkdir("newcroptesting")
except OSError as error:
   pass

coco_data = tfds.load('coco', split='train', shuffle_files=True)
for num,example in enumerate(coco_data):
   if num > 5:
    break #just want to test it on small amount first
   image = example['image']
   labels = example['objects']['label']
   bboxes = example['objects']['bbox']
   labelNums=labels.numpy()   
   shape=image.shape
   numpyImage=Image.fromarray(image.numpy())
   im = image.numpy()
   ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
   ss.setBaseImage(im)
   ss.switchToSelectiveSearchQuality()
   rects = ss.process()
   for count,label in enumerate(labelNums):
    if label in numbers:
        print("YAY")
        bbox = bboxes[count].numpy()
        print('Total Number of Region Proposals: {}'.format(len(rects)))
        numShowRects = 4000
        scores = []
        coordinates = []
        for i, rect in enumerate(rects):
            if (i < numShowRects):
                x, y, w, h = rect
                top, left, bottom, right = bbox[0], bbox[1], bbox[2], bbox[3] #this is a tensorflow box of an animal
                top2, left2, bottom2, right2 = float(y), float(x), float(y+h), float(x+w) #this is a box from opencv
                coordinates.append([float(y), float(x), float(y+h), float(x+w)])
                xA = max(left, left2)
                yA = max(top, top2)
                xB = min(right,right2)
                yB = min(bottom, bottom2)
                # compute the area of intersection rectangle
                interArea = (xB - xA + 1) * (yB - yA + 1)
                # compute the area of both the prediction and ground-truth
                # rectangles
                boxAArea = (right - left + 1) * (bottom - top + 1)
                boxBArea = (right2 - left2 + 1) * (bottom2 - top2 + 1)
                # compute the intersection over union by taking the intersection
                # area and dividing it by the sum of prediction + ground-truth
                # areas - the interesection area
                iou = interArea / float(boxAArea + boxBArea - interArea)
                scores.append(float(iou))
            else:
                break
        print('working on thing now')
        scores = tf.convert_to_tensor(scores)
        coordinates = tf.convert_to_tensor(coordinates)
        max_output_size = 2 #how many more boxes do we want per animal? Probably just like 2
        iou_threshold = .9 #this is NOT the iou from above, it's what iou we want to be allowed between the 2 (or more) new boxes we make. What should it be?
        max_output_size = tf.convert_to_tensor(max_output_size)
        iou_threshold = tf.convert_to_tensor(iou_threshold)
        selected_indices = tf.image.non_max_suppression(coordinates, scores, max_output_size, iou_threshold, score_threshold = .4) #how strongly do we want our new boxes to overlap with the original tensorflow animal box
        selected_boxes = tf.gather(coordinates, selected_indices)
        print("Stage 1:")
        print(selected_boxes)
        copy = im.copy()
        copy = Image.fromarray(copy)
        for index, box in enumerate(selected_boxes):
            print("This should print")
            print(box)
            box = box.numpy()
            top, left, bottom, right = box[0], box[1], box[2], box[3]
            draw = ImageDraw.Draw(copy)
            draw.rectangle([left, top, right, bottom], outline = "red")
            copy.save("newcroptesting/" +str(count) + str(index) +".jpg")




   