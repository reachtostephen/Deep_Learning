#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 22:08:24 2020

@author: stephenraj
"""

#Importing Dataset
from keras.datasets import mnist

#Data Representation
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape)
print(len(train_labels))
print(train_labels)
print(test_images.shape)
print(len(test_labels))
print(test_labels)


#Importing Classes
from keras import models
from keras import layers

#Building the architecture
network = models.Sequential()
network.add(layers.Dense(512, activation = 'relu', input_shape = (28*28,)))
network.add(layers.Dense(10,activation = 'softmax'))
network.compile(optimizer = 'rmsprop', loss='categorical_crossentropy', metrics = ['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000,28 * 28))
test_images = test_images.astype('float32') / 255

from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs = 20, batch_size = 128)
test_loss, test_accuracy = network.evaluate(test_images,test_labels)
print('Test Accuracy :', test_accuracy)


