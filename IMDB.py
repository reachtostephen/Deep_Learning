#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 15:19:48 2020

@author: stephenraj
"""

from keras.datasets import imdb
(train_data, train_labels), (test_data,test_labels) = imdb.load_data(num_words=10000)

print(train_data[0])
print(train_labels[0])
max([max(sequence) for sequence in train_data])

#Decoding back tp english words
word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for key,value in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i-3,'?')for i in train_data[0]])
print(decoded_review)

#Encoding the integer sequences into binary matrices - One Hot Encoder
import numpy as np
def vectorize_sequences(sequences, dimensions = 10000):
    results = np.zeros((len(sequences),dimensions))
    for i, sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
print(x_train)

y_train = np.array(train_labels).astype('float32')
y_test = np.array(test_labels).astype('float32')
print(y_train)

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation = "relu", input_shape = (10000,)))
model.add(layers.Dense(16, activation = "relu"))
model.add(layers.Dense(1, activation = "sigmoid"))

model.compile(optimizer = "rmsprop", loss = 'binary_crossentropy', metrics = ['accuracy'])

#To involve the parameters of the options
from keras import optimizers
from keras import losses
from keras import metrics

model.compile(optimizer = optimizers.RMSprop(lr = 0.001), loss = losses.binary_crossentropy, metrics = [metrics.binary_accuracy])

#Validation Set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train, partial_y_train, epochs = 20, batch_size = 512, validation_data = (x_val,y_val))
history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc = history_dict['binary_accuracy']

#Plotting Training and Validation Loss
epochs = range(1, len(acc)+1)
plt.plot(epochs,loss_values, 'bo', label = 'Training Loss')
plt.plot(epochs,val_loss_values,'b',label = 'Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show

#Plotting Training and Validation accuracy
plt.clf()
acc_values = history_dict['binary_accuracy']
val_acc_values = history_dict['val_binary_accuracy']
plt.plot(epochs,acc_values, 'bo', label = 'Training Accuracy')
plt.plot(epochs,val_acc_values,'b',label = 'Validation Accuracy')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show

#Reducing Model fit to 4 epochs due to overfitting
model = models.Sequential()
model.add(layers.Dense(16, activation = "relu", input_shape = (10000,)))
model.add(layers.Dense(16, activation = "relu"))
model.add(layers.Dense(1, activation = "sigmoid"))
model.compile(optimizer = "rmsprop", loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(x_train,y_train,epochs = 4,batch_size = 512)
results = model.evaluate(x_test,y_test)
print(results)

model.predict(x_test)


