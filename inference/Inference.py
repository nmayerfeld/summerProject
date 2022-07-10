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
cv2.setUseOptimized(True);
cv2.setNumThreads(4);

model = load_model('model.h5')
class_names = ['background', 'bear', 'cat', 'cow', 'dog', 'elephant', 'giraffe', 'horse', 'sheep', 'zebra']

img_height = 180
img_width = 180

coordinates = []
dict = {}

link = "/home/ec2-user/visionaries/WholeImageTests/zebra/image51409crop0.jpg"

im = cv2.imread(link)
imOut = im.copy()

#newHeight = 200
#newWidth = int(im.shape[1]*200/im.shape[0])

#im = cv2.resize(im, (newWidth, newHeight)) I TOOK THIS OUT, MAYBE put it back BUT WHY?

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

for box in coordinates:
    copy = im.copy()
    copy = Image.fromarray(copy)
    box = box.numpy()
    top, left, bottom, right = box[0], box[1], box[2], box[3]
    croppedImage = copy.crop((left, top, right, bottom))
    croppedImage = croppedImage.resize((img_height, img_width))
    #croppedImage.save('SavedPatches/' + str(box) + '.jpg')
    img_array = tf.keras.utils.img_to_array(croppedImage)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print("score: \n" + str(score))
    print("Score type: " + str(type(score)))
    if (class_names[np.argmax(score)] == "background"):
        scores.append(0)
        print("it's a background")
    else:
        scores.append(np.max(score))
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    dict[(top, left, bottom, right)] = class_names[np.argmax(score)]
#print("scores1: \n" + str(scores))
scores = tf.convert_to_tensor(scores)
#print("scores2: \n" + str(scores))

max_output_size = 7
iou_threshold = 0.15

#scores = tf.convert_to_tensor(scores)
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
        
#text = top_name
#draw.text((5, 5), text, fill='red', align ="left") 
#number = 100 * np.max(score)
#draw.text((5, 5), number,  align ="right") 
try:
   os.mkdir("boxed_images")
except OSError as error:
   pass

#im.save("boxed_images/thirteenth_test.jpg")
im.save("boxed_images/" + os.path.basename(link))
#cv2.imwrite("opencv.jpg", imOut) #you can take this stuff out I was just testing
#print(dict)