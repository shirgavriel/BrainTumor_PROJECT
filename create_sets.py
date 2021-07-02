# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 14:22:47 2021

@author: שיר גבריאל
"""

'Creating Testing Validation and Testing Sets'


import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


batch = 12
img_height = 250
img_width = 250
 
def create_train(data_dir):
    train = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset = 'training',
    seed = 42,
    image_size  =(img_height,img_width),
    batch_size = batch)
    return train
  
def create_val(data_dir):    
    val = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset = 'validation',
    seed = 42,
    image_size = (img_height,img_width),
    batch_size = batch)
    return val

def create_test(data_dir): 
    test = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    seed = 42,
    image_size = (img_height,img_width),
    batch_size = batch)


#changing the photos in order to make a better colection
data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(img_height, 
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

def improve_images():
#changing the photos in order to make a better colection
    data_augmentation = keras.Sequential(
      [
        layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                     input_shape=(img_height, 
                                                                  img_width,
                                                                  3)),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
      ]
    )
    return data_augmentation
        
    
    
