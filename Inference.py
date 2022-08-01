import numpy as np
import os
import PIL
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import pathlib
import tensorflow_addons as tfa

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

model = load_model('RNm7PostTune')
class_names = ['background', 'bear', 'cat', 'cow', 'dog', 'elephant', 'giraffe', 'horse', 'sheep', 'zebra']

img_height = 160
img_width = 160

directory='sortedPics/forTesting'
for filename in os.listdir(directory):
    link = os.path.join(directory, filename)
    coordinates = []
    dict = {}
    im = cv2.imread(link)
    imOut = im.copy()

    #coordinates = []
    #dict = {}

    #link = "/home/ubuntu/WholeImageTests/zebra/image41040crop0.jpg"

    #im = cv2.imread(link)
    #imOut = im.copy()

    # create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    # set input image on which we will run segmentation
    ss.setBaseImage(im)

    ss.switchToSelectiveSearchFast()

    rects = ss.process()
    print('Total Number of Region Proposals: {}'.format(len(rects)))

    # number of region proposals to show
    numShowRects = 15000

    # itereate over all the region proposals
    for i, rect in enumerate(rects):
        x, y, w, h = rect
        if w > 60 and h > 60:
            coordinates.append([float(y), float(x), float(y+h), float(x+w)])


    coordinates = tf.convert_to_tensor(coordinates)
    scores = []
    pics = []
    print("Number of coordinate boxes: " + str(len(coordinates)))
    if len(coordinates) == 0:
        continue
    else:
        for box in coordinates:
            copy = im.copy()
            RGB_image = cv2.cvtColor(copy, cv2.COLOR_BGR2RGB)
            copy = Image.fromarray(RGB_image)
            box = box.numpy()
            top, left, bottom, right = box[0], box[1], box[2], box[3]
            croppedImage = copy.crop((left, top, right, bottom))
            croppedImage = croppedImage.resize((img_height, img_width))
            img_array = tf.keras.utils.img_to_array(croppedImage)
            pics.append(img_array)
        print("Number of patches: " + str(len(pics)))
        np_pics = np.array(pics)
        print("making predictions")
        predictions = model.predict(np_pics)
        print("Predictions completed.")



        for index, prediction in enumerate(predictions):
            score = prediction
            top, left, bottom, right =  coordinates[index][0].numpy(), coordinates[index][1].numpy(), coordinates[index][2].numpy(), coordinates[index][3].numpy(),
            dict[(top, left, bottom, right)] = class_names[np.argmax(score)]

            if (class_names[np.argmax(score)] == "background"):
                scores.append(0.0)       
            else:
                scores.append(np.max(score))
            print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
            )
            


        scores = tf.convert_to_tensor((scores))

        max_output_size = 30
        iou_threshold = 0.999

        max_output_size = tf.convert_to_tensor(max_output_size)
        iou_threshold = tf.convert_to_tensor(iou_threshold)
        
        print("scores: " + str(scores))
        selected_indices = tf.image.non_max_suppression(
            coordinates, scores, max_output_size, iou_threshold, score_threshold = .995)

        selected_boxes = tf.gather(coordinates, selected_indices)

        counter = 0
        total = 0
        for box in selected_boxes:
            top, left, bottom, right = box[0].numpy(), box[1].numpy(), box[2].numpy(), box[3].numpy() 
            total += (right-left)*(bottom - top)
            counter += 1
        if counter != 0:
            average = total/counter


        sizes = []
        coordinates2 = []
        for box in selected_boxes:
            top, left, bottom, right = box[0].numpy(), box[1].numpy(), box[2].numpy(), box[3].numpy()
            size = (right - left) * (bottom-top)
            denominator = abs(size - (average + (average / 6)))
            if denominator == 0:
                denominator = .01
            sizes.append(np.float32((1/(denominator)) + 1))
            coordinates2.append([float(top), float(left), float(bottom), float(right)])
        print("Sizes: " + str(sizes))
        sizes = tf.convert_to_tensor(sizes)
        print("Tensor Sizes: " + str(sizes))
        coordinates2 = tf.convert_to_tensor(coordinates2)

        if len(coordinates2) != 0:
            print("number of boxes: " + str(len(coordinates2)))

            max_output_size = 7
            iou_threshold = 0.01

            max_output_size = tf.convert_to_tensor(max_output_size)
            iou_threshold = tf.convert_to_tensor(iou_threshold)

            selected_indices = tf.image.non_max_suppression(
                coordinates2, sizes, max_output_size, iou_threshold, score_threshold = -1000.01)

            selected_boxes = tf.gather(coordinates2, selected_indices)


            imageRGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(imageRGB)

            print("Number of selected boxes: " + str(len(selected_boxes)))
            for box in selected_boxes:
                top, left, bottom, right = box[0].numpy(), box[1].numpy(), box[2].numpy(), box[3].numpy()
                draw = ImageDraw.Draw(im)
                draw.rectangle([left, top, right-1, bottom-1], outline = "red")
                draw.text((left + 3, top + 3), dict[(top, left, bottom, right)], color = "red")
                    
            try:
                os.mkdir("PostPatchInferencedImages")
            except OSError as error:
                pass
            print("link:" + str(link))
            im.save("/home/ec2-user/visionaries/PostPatchInferencedImages/" + os.path.basename(link))