# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 17:50:09 2022

@author: User
"""

train_path = 'C:/Users/User\Desktop/Latest-GP/Actual-Running/UPDATEDcat-img-code/train'
test_path = 'C:/Users/User/Desktop/Latest-GP/Actual-Running/UPDATEDcat-img-code/train'



from tensorflow.keras.preprocessing.image import ImageDataGenerator


batch_size=32


train_datagen = ImageDataGenerator(rescale=1./255 , shear_range=0.2 , zoom_range=0.2 , horizontal_flip=True,vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
train_set = train_datagen.flow_from_directory(train_path,target_size=(100,100),class_mode='categorical',batch_size=batch_size)
test_set = test_datagen.flow_from_directory(test_path,target_size=(100,100),class_mode='categorical',batch_size=batch_size)

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.applications.vgg16 import preprocess_input,VGG16
import tensorflow as tf
from tensorflow.keras.models import  Model
from tensorflow.keras.preprocessing import image

vgg = VGG16(include_top=False,weights="imagenet",input_shape=(100,100,3))

for layer in vgg.layers:
  layer.trainable=False
  
x = Flatten()(vgg.output)
x = Dense(6, activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=x)

from tensorflow import keras

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime 

checkpointer = ModelCheckpoint(filepath='./audio_classification.hdf5', 
                               verbose=1, save_best_only=True)
epochs = 20
history = model.fit(train_set, batch_size=8, 
                    epochs=epochs, 
                    validation_data=test_set,
                    steps_per_epoch=len(train_set),
                    callbacks=[checkpointer],
                    validation_steps=(len(test_set)))




plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()



# Loss
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()





score = model.evaluate(test_set)
print('accuracy:', score[1])






  