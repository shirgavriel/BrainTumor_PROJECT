# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 18:21:21 2021

@author: שיר גבריאל
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
#import cv2
#import os
#import glob
#import csv


#from sklearn.model_selection import KFold, StratifiedKFold
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.preprocessing.image import image_dataset_from_directory


def train_model(model, train, val):
    epochs=10
    history = model.fit(
      train,
      validation_data=val,
      epochs=epochs
    )
    return history


def test_model(train, test, model, classes):
    #Retrieve a batch of images from the test set
    image_batch, label_batch = test.as_numpy_iterator().next()
    prediction = model.predict_on_batch(image_batch).flatten()
    
    # Apply a sigmoid since our model returns logits
    predictions = tf.nn.sigmoid(prediction).numpy()
    
    n = 0
    predict = []
    while n<=(predictions.shape[0]-4):
        pred = np.argmax(predictions[n:n+4]) #Returns the index of the largest element in the selected subarray
        n+=4
        predict.append(pred)
    predict = np.array(predict)
    
    #print('Predictions:\n',predictions)#.numpy())
    print('Labels:\n', label_batch)
    print('Predictions:\n',predict)
    '''
        print(predictions.shape)
        print(label_batch.shape)
        print(predict.shape)
    '''
    results = model.evaluate(test)
    print("test loss, test acc:", results)
    
    
    labels_entire = []
    pred_entire = []
    for image_batch,label_batch in test.as_numpy_iterator():
        prediction = model.predict_on_batch(image_batch).flatten()
    
        # Apply a sigmoid since our model returns logits
        predictions = tf.nn.sigmoid(prediction).numpy()
    
        n = 0
        predict = []
        while n<=(predictions.shape[0]-4):
            pred = np.argmax(predictions[n:n+4]) #Returns the index of the largest element in the selected subarray
            n+=4
            pred_entire.append(pred)
        for el in label_batch:
            labels_entire.append(el)
    pred_entire = np.array(pred_entire)
    labels_entire = np.array(labels_entire)
    print(pred_entire)
    print(labels_entire)
   # classes = train.class_names
    print(classification_report(labels_entire, pred_entire, target_names=classes))

    
