# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 14:30:25 2021

@author: שיר גבריאל
"""

import tensorflow as tf
from tensorflow.keras import layers


img_height = 250
img_width = 250
num_classes = 4

def model():
    model = tf.keras.Sequential([
      layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes)
    ])
    

    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()
    model.save("C:/Users/שיר גבריאל/Desktop/savedmodel")
    return model



"""תיאור של כל המודל הערות, ליירס, מקספולינג וכו"""