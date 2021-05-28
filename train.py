import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow_model_optimization as tfmot

import icenet

img_width = 80
img_height = 80
batch_size = 32

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False, samplewise_center=False,
    featurewise_std_normalization=False, samplewise_std_normalization=False,
    zca_whitening=False, zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.0,
    height_shift_range=0.0, brightness_range=[0.5,1.2], shear_range=0.0, zoom_range=[1.0,1.0],
    channel_shift_range=15, fill_mode='nearest',
    horizontal_flip=False, vertical_flip=False, rescale=1.0/256.0)

train_gen = train_datagen.flow_from_directory(
        'data/train',
        class_mode='sparse',
        target_size=(80,80),
        batch_size=32)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/256.0)

val_gen = val_datagen.flow_from_directory(
        'data/val',
        class_mode='sparse',
        target_size=(80,80),
        batch_size=32)

class_names = train_gen.class_indices
print(class_names)

model = icenet.make_net((img_width,img_height,3),len(class_names))
model.summary()


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs=14
history = model.fit(
  train_gen, steps_per_epoch=235,
  validation_data=val_gen, validation_steps=45,
  epochs=epochs
)

model.save('ice_model')
