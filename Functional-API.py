#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 14:03:48 2020

@author: stephenraj
"""

from keras.models import Sequential, Model
from keras import layers
from keras import Input
from keras import Input,layers
from keras.models import Model

seq_model = Sequential()
seq_model.add(layers.Dense(32,activation = 'relu', input_shape = (64,)))
seq_model.add(layers.Dense(32, activation = 'relu'))
seq_model.add(layers.Dense(10,activation = 'softmax'))

input_tensor = Input(shape = (64,))
x = layers.Dense(32, activation = 'relu')(input_tensor)
x = layers.Dense(32, activation = 'relu')(x)
output_tensor = layers.Dense(10,activation = 'softmax')(x)

model = Model(input_tensor,output_tensor)
print(model.summary())

unrelated_input= Input(shape = (32,))
bad_model = model = Model(unrelated_input, output_tensor)
#Since Output tensor and Input tensor aren't of same shape it causes runtime error

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy')

import numpy as np
x_train = np.random.random((1000,64))
y_train = np.random.random((1000,10))

model.fit(x_train, y_train, epochs =19, batch_size = 128)
score = model.evaluate(x_train,y_train)
print(score)

'''
from keras.models import Model
from keras import layers
from keras import Input

text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500

text_input = Input(shape =(None,), dtype = 'int32', name = 'text')

embedded_text = layers.Embedding(64, text_vocabulary_size)(text_input)

encoded_text = layers.LSTM(32)(embedded_text)

question_input = Input(shape = (None,),
                       dtype = 'int32',
                       name = 'question')

embedded_question = layers.Embedding(
    32,question_vocabulary_size)(question_input)

encoded_question = layers.LSTM(16)(embedded_question)
concatenated = layers.concatenate([encoded_text, encoded_question], axis = -1)

answer = layers.Dense(answer_vocabulary_size,
                      activation = 'softmax')(concatenated)

model = Model([text_input, question_input], answer)

model.compile(optimizer = 'rmsprop',
              loss = 'categorical_crossentropy',
              metrics = ['acc'])

import numpy as np
num_samples = 1000
max_length = 100

text = np.random.randint(1, text_vocabulary_size,
                         size = (num_samples,max_length))
question = np.random.randint(1,question_vocabulary_size,
                             size = (num_samples,max_length))
answers = np.random.randint(0,1,size = (num_samples,answer_vocabulary_size))

#model.fit([text,question],answers, epochs =10,batch_size =128)

model.fit({'text':text,'question':question},answers,epochs =19, batch_size =128)
'''

from keras import layers
from keras import Input
from keras.models import Model

'''
vocabulary_size =50000
num_income_groups =10

posts_input = Input(shape = (None,),dtype = 'int32', name = 'posts')
embedded_posts = layers.Embedding(256,vocabulary_size)(posts_input)
x = layers.Conv1D(128,5, activation = 'relu')(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256,5,activation = 'relu')(x)
x = layers.Conv1D(256,5,activation = 'relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256,5,activation = 'relu')(x)
x = layers.Conv1D(256,5,activation = 'relu')(x)
x = layers.GlobalMaxPool1D()(x)
x = layers.Dense(128,activation = 'relu')(x)

age_prediction = layers.Dense(1, name ='age')(x)
income_prediction = layers.Dense(num_income_groups,
                                 activation = 'softmax',
                                 name = 'income')(x)

gender_prediction = layers.Dense(1,activation ='sigmoid', name='gender')(x)
model = Model(posts_input,
              [age_prediction, income_prediction, gender_prediction])


model.compile(optimizer = 'rmsprop',
              loss = ['mse','categorical_crossentropy','binary_crossentropy'])

model.compile(optimizer='rmsprop',
              loss = {'age':'mse',
                      'income':'categorical_crossentropy',
                      'gender':'binary_crossentropy'})

model.compile(optimizer = 'rmsprop',
              loss = ['mse','categorical_crossentropy','binary_crossentropy'],
              loss_weights = [0.25,1.,10.])

model.compile(optimizer='rmsprop',
              loss = {'age':'mse',
                      'income':'categorical_crossentropy',
                      'gender':'binary_crossentropy'},
              loss_weights = {'age':0.25,
                              'income' :1.,
                              'gender' :10.})

model.fit(posts,[age_targets, income_targets, gender_targets],
          epochs =10, batch_size = 64)

'''

from keras import layers

branch_a = layers.Conv2D(128,1,
                         activation = 'relu',strides = 2)(x)
branch_b = layers.Conv2D(128,1,activation = 'relu')(x)
branch_b = layers.Conv2D(128,3,
                         activation = 'relu',strides = 2)(branch_b)
branch_c = layers.AveragePooling2D(3,strides =2)(x)

branch_d = layers.Conv2D(128,1,
                         activation = 'relu',strides = 2)(x)
branch_d = layers.Conv2D(128,3,
                         activation = 'relu',strides = 2)(branch_d)

branch_d = layers.Conv2D(128,3,
                         activation = 'relu',strides = 2)(branch_d)

output = layers.concatenate(
    [branch_a,branch_b,branch_c,branch_d],axis =-1)

from keras import layers





