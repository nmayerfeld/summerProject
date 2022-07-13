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
import random
import sys
import cv2
import tensorflow_datasets as tfds
import os as os
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt
import random

keys = ['cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe']
numbers = [15,16,17,18,19,20,21,22,23]
coordinates = []

cv2.setUseOptimized(True)
cv2.setNumThreads(4)

coco_data = tfds.load('coco', split='train', shuffle_files=True)
for num,example in enumerate(coco_data):
   if num > 20:
    break #just want to test it on small amount first
   image = example['image']
   labels = example['objects']['label']
   bboxes = example['objects']['bbox']
   labelNums=labels.numpy()   
   shape=image.shape
   numpyImage=Image.fromarray(image.numpy())
   im = image.numpy()
   ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
   ss.setBaseImage(numpyImage)
   ss.switchToSelectiveSearchQuality()
   rects = ss.process()
   for count,label in enumerate(labelNums):
    if label in numbers:
        bbox = bboxes[count].numpy()
        scores = []
        print('Total Number of Region Proposals: {}'.format(len(rects)))
        numShowRects = 5000
        for i, rect in enumerate(rects):
            if (i < numShowRects):
                x, y, w, h = rect
                top, left, bottom, right = bbox[0].numpy(), bbox[1].numpy(), bbox[2].numpy(), bbox[3].numpy() #this is a tensorflow box of an animal
                top2, left2, bottom2, right2 = float(y), float(x), float(y+h), float(x+w) #this is a box from opencv
                score = #the iou between the two boxes
                scores.append(score)
            else:
                break
        scores = tf.convert_to_tensor(scores)
        max_output_size = 2 #how many more boxes do we want per animal? Probably just like 2
        iou_threshold = 0.3 #this is NOT the iou from above, it's what iou we want to be allowed between the 2 (or more) new boxes we make. What should it be?
        max_output_size = tf.convert_to_tensor(max_output_size)
        iou_threshold = tf.convert_to_tensor(iou_threshold)
        selected_indices = tf.image.non_max_suppression(coordinates, scores, max_output_size, iou_threshold, score_threshold = .7) #how strongly do we want our new boxes to overlap with the original tensorflow animal box
        selected_boxes = tf.gather(coordinates, selected_indices)
        #take code from other things to cut an image from the tensorflow box (unless we're running this code as an add-on, in which case that's already been done)
        #and cut images from the extra boxes too this time and save them all to the class training data




   