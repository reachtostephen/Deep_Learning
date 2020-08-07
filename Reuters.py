#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 11:10:41 2020

@author: stephenraj
"""

from keras.datasets import reuters
(train_data,train_labels), (test_data,test_labels) = reuters.load_data(num_words=10000)
print(len(train_data))
print(len(test_data))
print(train_data[10])
#Decoding back to words
word_index = reuters.get_word_index()
reverse_word_index = dict((value, key) for (key, value) in word_index.items())
decoded_newswire = ' '.join([reverse_word_index.get(i-3,'?')for i in train_data[0]])
print(reverse_word_index)
print(decoded_newswire)

print(train_labels[10])

#Vectorise data
import numpy as np

def vectorize_sequences(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

print(x_train)

def to_one_hot(labels, dimension= 46):
    results = np.zeros((len(labels),dimension))
    for i, label in enumerate(labels):
        results[i,label] = 1.
    return results

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)
print(one_hot_train_labels)

from keras.utils.np_utils import to_categorical

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation = 'relu', input_shape = (10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer = 'rmsprop',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = 20,
                    batch_size = 512,
                    validation_data = (x_val,y_val))

import matplotlib.pyplot as plt
history_dict = history.history
loss = history_dict['loss']
print(loss)
val_loss = history_dict['val_loss']
print(val_loss)
acc = history_dict['accuracy']

epochs = range(1, len(loss)+1)
print(epochs)
plt.plot(epochs,loss,'bo',label = 'Training loss')
plt.plot(epochs,val_loss,'b',label = 'valiidation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.clf()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
plt.plot(epochs,acc,'bo',label = 'Training Accuracy')
plt.plot(epochs,val_acc,'b',label = 'valiidation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model = models.Sequential()
model.add(layers.Dense(64,activation = 'relu',input_shape = (10000,)))
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer = 'rmsprop',
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

model.fit(partial_x_train, partial_y_train, epochs =9, batch_size = 512, validation_data=(x_val,y_val))
results = model.evaluate(x_test, one_hot_test_labels)
print(results)

import copy
test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
hits_array = np.array(test_labels) == np.array(test_labels_copy)
print(float(np.sum(hits_array))/ len(test_labels))

predictions = model.predict(x_test)

predictions[0].shape

print(np.sum(predictions[0]))
print(np.argmax(predictions[0]))

y_train = np.array(train_labels)
y_test = np.array(test_labels)
model.compile(optimizer = 'rmsprop',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model = models.Sequential()
model.add(layers.Dense(64,activation = 'relu',input_shape = (10000,)))
model.add(layers.Dense(4, activation = 'relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer = 'rmsprop',
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])
model.fit(partial_x_train, partial_y_train, epochs =20, batch_size = 128, validation_data=(x_val,y_val))






