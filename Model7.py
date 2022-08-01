import matplotlib.pyplot as plt
import numpy as np
import os
import keras
from tensorflow.keras import layers
from PIL import Image
import tensorflow as tf
import tensorflow_addons as tfa
train_ds = tf.keras.utils.image_dataset_from_directory(
  'sortedPics/forTraining',
  label_mode='categorical',
  validation_split=0.2,
  image_size=(160,160),
  subset="training",
  seed=123,
  batch_size=64)
val_ds = tf.keras.utils.image_dataset_from_directory(
  'sortedPics/forTraining',
  label_mode='categorical',
  validation_split=0.2,
  image_size=(160,160),
  subset="validation",
  seed=123,
  batch_size=64)
class_names = train_ds.class_names
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
size = (160, 160)
#MODEL 2-STANDARD, lr=0.001 epochs=50/10, opt=adam, 
resnet_model7 = tf.keras.Sequential([tf.keras.layers.RandomFlip('vertical'),
  tf.keras.layers.RandomRotation(0.2),])

pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                   input_shape=(160,160,3),
                   pooling='avg',classes=10,
                   weights='imagenet')
for layer in pretrained_model.layers:
        layer.trainable=False

resnet_model7.add(pretrained_model)
base_learning_rate=0.0001
resnet_model7.add(tf.keras.layers.Dense(10, activation='softmax'))
resnet_model7.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
    loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits=False),
    metrics=['accuracy'],
)

initial_epochs = 8
history=resnet_model7.fit(train_ds, epochs=initial_epochs, validation_data=val_ds)
resnet_model7.save("RNm7PreTune")
resnet_model7.save("RNm7PreTune.h5")

#create graph
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(initial_epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')

plt.savefig("RNm7PreFineTune.jpg")

#fine-tuning
pretrained_model.trainable = True
# Fine-tune from this layer onwards
fine_tune_at = 115

# Freeze all the layers before the `fine_tune_at` layer
for layer in pretrained_model.layers[:fine_tune_at]:
  layer.trainable = False
#set up callback for early stopping
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

resnet_model7.compile(loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits=False),
              optimizer = tf.keras.optimizers.Adam(learning_rate=base_learning_rate/10),
              metrics=['accuracy'])
fine_tune_epochs = 10
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = resnet_model7.fit(train_ds,
                         epochs=total_epochs, callbacks=[callback],
                         initial_epoch=history.epoch[-1],
                         validation_data=val_ds)
resnet_model7.save("RNm7PostTune")
resnet_model7.save("RNm7PostTune.h5")

#create graph
acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig("RNm7PostFineTune.jpg")