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
from time import sleep
cv2.setUseOptimized(True);
cv2.setNumThreads(4);

model = load_model('/home/ec2-user/visionaries/ProjectRepo/summerProject/model4.h5')
class_names = ['background', 'teddy bear', 'bear', 'cat', 'cow', 'dog', 'elephant', 'giraffe', 'horse', 'sheep', 'zebra']

img_height = 160
img_width = 160

coordinates = []
dict = {}

link = "/home/ec2-user/visionaries/WholeImageTests/dog/image36155crop0.jpg"

im = cv2.imread(link)
imOut = im.copy()

# create Selective Search Segmentation Object using default parameters
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# set input image on which we will run segmentation
ss.setBaseImage(im)

ss.switchToSelectiveSearchQuality()

rects = ss.process()
print('Total Number of Region Proposals: {}'.format(len(rects)))

# number of region proposals to show
numShowRects = 1700

# itereate over all the region proposals
for i, rect in enumerate(rects):
    x, y, w, h = rect
    coordinates.append([float(y), float(x), float(y+h), float(x+w)])

coordinates = tf.convert_to_tensor(coordinates)
scores = []
pics = []
for box in coordinates:
    copy = im.copy()
    copy = Image.fromarray(copy)
    box = box.numpy()
    top, left, bottom, right = box[0], box[1], box[2], box[3]
    croppedImage = copy.crop((left, top, right, bottom))
    croppedImage = croppedImage.resize((img_height, img_width))
    img_array = tf.keras.utils.img_to_array(croppedImage)
    pics.append(img_array)


np_pics = np.array(pics)
print("making predictions")
predictions = model.predict(np_pics)
print("Predictions completed.")
sleep(3)

for index, prediction in enumerate(predictions):
    score = tf.nn.softmax(prediction)
    #print("score: \n" + str(score))
    #print("Score type: " + str(type(score)))
    if (class_names[np.argmax(score)] == "background"):
        scores.append(0)
        #print("it's a background")
    else:
        scores.append(np.max(score))
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    top, left, bottom, right =  coordinates[index][0].numpy(), coordinates[index][1].numpy(), coordinates[index][2].numpy(), coordinates[index][3].numpy(),
    dict[(top, left, bottom, right)] = class_names[np.argmax(score)]

scores = tf.convert_to_tensor(scores)

max_output_size = 7
iou_threshold = 0.01

max_output_size = tf.convert_to_tensor(max_output_size)
iou_threshold = tf.convert_to_tensor(iou_threshold)

selected_indices = tf.image.non_max_suppression(
    coordinates, scores, max_output_size, iou_threshold, score_threshold = .7)

selected_boxes = tf.gather(coordinates, selected_indices)
im = Image.fromarray(im)

for box in selected_boxes:
    top, left, bottom, right = box[0].numpy(), box[1].numpy(), box[2].numpy(), box[3].numpy()
    draw = ImageDraw.Draw(im)
    draw.rectangle([left, top, right, bottom], outline = "red")
    draw.text((left + 3, top + 3), dict[(top, left, bottom, right)], color = "red")
        
try:
   os.mkdir("boxed_images")
except OSError as error:
   pass

im.save("boxed_images/" + os.path.basename(link))