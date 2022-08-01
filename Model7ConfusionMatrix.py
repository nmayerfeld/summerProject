import numpy as np
import os
import PIL
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow_addons
from keras.models import load_model
import random
import sys
import cv2
from time import sleep
val_ds = tf.keras.utils.image_dataset_from_directory(
  'sortedPics/forTraining',
  validation_split=0.2,
  image_size=(160,160),
  subset="validation",
  seed=123,
  batch_size=64)
train_ds = tf.keras.utils.image_dataset_from_directory(
  'sortedPics/forTraining',
  validation_split=0.2,
  image_size=(160,160),
  subset="training",
  seed=123,
  batch_size=64)
model = load_model('RNm7PostTune')
class_names = ['background',  'bear', 'cat', 'cow', 'dog', 'elephant', 'giraffe', 'horse', 'sheep', 'zebra']
#confusion matrix
y_pred = []  # store predicted labels
y_true = []  # store true labels

# iterate over the dataset
for image_batch, label_batch in val_ds:   # use dataset.unbatch() with repeat
   # append true labels
   y_true.append(label_batch)
   # compute predictions
   preds = model.predict(image_batch)
   # append predicted labels
   y_pred.append(np.argmax(preds, axis = - 1))

# convert the true and predicted labels into tensors
correct_labels = tf.concat([item for item in y_true], axis = 0)
predicted_labels = tf.concat([item for item in y_pred], axis = 0)

print(tf.math.confusion_matrix(correct_labels, predicted_labels))
file1 = open("RNm7ValSetPostTune.txt","w")
file1.write(str(tf.math.confusion_matrix(correct_labels, predicted_labels).numpy()))
file1.close()
