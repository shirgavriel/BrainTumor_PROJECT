# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 14:20:27 2021

@author: שיר גבריאל
"""

#main code
import tensorflow as tf
import PIL
import pathlib
import create_sets
import train_the_model
import my_model

import matplotlib.pyplot as plt




#getting the path
dataset = "C:/Users/שיר גבריאל/Desktop/brain images/Training"
data_dir = pathlib.Path(dataset)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

'a tumor image ' #print for user imeges to the screen
tumors = list(data_dir.glob('glioma_tumor/*'))
print(tumors[1])
img1 = PIL.Image.open(str(tumors[0]))
'a non-tumor image'
not_tumors = list(data_dir.glob('no_tumor/*'))
img2 = PIL.Image.open(str(not_tumors[0]))

'Creating Testing Validation and Testing Sets'
train = create_sets.create_train(data_dir)
val = create_sets.create_val(data_dir)
test = create_sets.create_val(data_dir)

classes = train.class_names

AUTOTUNE = tf.data.experimental.AUTOTUNE
train = train.prefetch(buffer_size=AUTOTUNE)
val = val.prefetch(buffer_size=AUTOTUNE)
test = test.prefetch(buffer_size=AUTOTUNE)
model = my_model.model()
data_augmantation = create_sets.improve_images()




history = train_the_model.train_model(model, train, val)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = 10
epochs_range = range(epochs)

test_result = train_the_model.test_model(train, test, model, classes)
#Retrieve a batch of images from the test set
image_batch, label_batch = test.as_numpy_iterator().next()
prediction = model.predict_on_batch(image_batch).flatten()

# Apply a sigmoid since the model returns logits
predictions = tf.nn.sigmoid(prediction).numpy()


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
plt.title('Training and Validation Loss')
plt.show()




