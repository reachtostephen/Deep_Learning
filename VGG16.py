#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:26:41 2020

@author: stephenraj
"""

from keras.applications import VGG16

conv_base = VGG16(weights = 'imagenet',
                  include_top = False,
                  input_shape = (150,150,3))

print(conv_base.summary())

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

base_dir = '/Users/stephenraj/Deep_Learning/Datasets/dogs-vs-cats-1'
train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'validation')
test_dir = os.path.join(base_dir,'test')

datagen = ImageDataGenerator(rescale = 1./255)
batch_size = 20

def extract_features(directory, sample_count):
    features = np.zeros(shape = (sample_count,4,4,512))
    labels = np.zeros(shape= (sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size = (150,150),
        batch_size = batch_size,
        class_mode = 'binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i+1) * batch_size] = features_batch
        labels[i * batch_size : (i+1) * batch_size ] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features,labels

train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features,test_labels = extract_features(test_dir, 1000) 

train_features = np.reshape(train_features, (2000, 4*4*512))
validation_features = np.reshape(validation_features, (1000,4*4*512))
test_features = np.reshape(test_features,(1000,4*4*512))

from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(layers.Dense(256,activation = 'relu', input_dim = 4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer = optimizers.RMSprop(lr = 2e-5),
              loss = 'binary_crossentropy',
              metrics = ['acc'])

history = model.fit(train_features, train_labels,
                    epochs =30,
                    batch_size = 30,
                    validation_data = (validation_features, validation_labels))

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo',label = 'Training acc')
plt.plot(epochs,val_acc,'b',label = 'Validation acc')
plt.title('Training and Validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss,'bo',label ='Training loss')
plt.plot(epochs, val_loss,'b',label = 'Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()

from keras.models import load_model
model = load_model('/Users/stephenraj/Deep_Learning/cats_and_dogs_small_2.h5')
print(model.summary())

img_path = '/Users/stephenraj/Deep_Learning/Datasets/dogs-vs-cats-1/test/cats/cat.1700.jpg'

from keras.preprocessing import image
import numpy as np

img =image.load_img(img_path, target_size = (150,150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis = 0)
img_tensor /=255.
print(img_tensor.shape)

import matplotlib.pyplot as plt
plt.imshow(img_tensor[0])
plt.show()

from keras import models

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs = model.input, outputs = layer_outputs)

activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]
print(first_layer_activation.shape)

import matplotlib.pyplot as plt
plt.matshow(first_layer_activation[0,:,:,4],cmap = 'viridis')

plt.matshow(first_layer_activation[0,:,:,7],cmap = 'viridis')

layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16
for layer_name,layer_activation in zip(layer_names,activations):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                            :,:,
                                            col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *=64
            channel_image += 128
            channel_image = np.clip(channel_image, 0 , 255).astype('uint8')
            display_grid[col * size : (col+1) * size,
                         row * size : (row+1) * size]= channel_image

scale = 1. / size
plt.figure(figsize = (scale * display_grid.shape[1],
                      scale * display_grid.shape[0]))

plt.title(layer_name)
plt.grid(False)
plt.imshow(display_grid, aspect = 'auto', cmap = 'viridis')
    
from keras.applications import VGG16
from keras import backend as K

model = VGG16(weights = 'imagenet',
              include_top = False)

layer_name = 'block3_conv1'
filter_index = 0

layer_output = model.get_layer(layer_name).output
loss = K.mean(layer_output[:,:,:,filter_index])

grads = K.gradients(loss, model.input)[0]

grads /= (K.sqrt(K.mean(K.square(grads)))+ 1e-5)

iterate = K.function([model.input],[loss,grads])

import numpy as np
loss_value, grads_value = iterate([np.zeros((1,150,150,3))])

input_img_data = np.random.random((1,150,150,3)) * 20 +128

step =1 
for i in range(40):
    loss_value, grads_value = iterate([input_img_data])
    input_img_data += grads_value * step


def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    
    x += 0.5
    x = np.clip(x,0,1)
    
    x *=255
    x = np.clip(x,0,255).astype('uint8')
    return x

def generate_pattern(layer_name, filter_index, size = 150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:,:,:,filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads)))+ 1e-5)
    iterate = K.function([model.input],[loss,grads])
    input_img_data = np.random.random((1,size,size,3)) * 20 +128
    step =1 
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        
    img = input_img_data[0]
    return deprocess_image(img) 
    
plt.imshow(generate_pattern('block3_conv1',0))

size = 64
margin = 5

results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

for i in range(8):
    for j in range(8):
        filter_img = generate_pattern(layer_name, i + (j * 8), size = size)
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        results[horizontal_start : horizontal_end,
               vertical_start:vertical_end,:] = filter_img
        
plt.figure(figsize=(20,20))
plt.imshow(results)

from keras.applications.vgg16 import VGG16
model = VGG16(weights = 'imagenet')

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input,decode_predictions
import numpy as np

img_path ='/Users/stephenraj/Deep_Learning/download.jpeg'
img = image.load_img(img_path,target_size = (224,224))
x = image.img_to_array(img)
x = np.expand_dims(x,axis = 0)
x = preprocess_input(x)
preds = model.predict(x)
print('Predicted : ',decode_predictions(preds, top=3)[0])

print(np.argmax(preds[0]))

african_elephant_output = model.output[:,386]
last_conv_layer = model.get_layer('block5_conv3')
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis = (0,1,2))
iterate = K.function([model.input],
                     [pooled_grads,last_conv_layer.output[0]])

pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(512):
    conv_layer_output_value[:,:,i] *= pooled_grads_value[i]
    
heatmap = np.mean(conv_layer_output_value,axis =-1)

heatmap = np.maximum(heatmap,0)
heatmap/= np.max(heatmap)
plt.matshow(heatmap)

import cv2
img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1],img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 +img
cv2.imwrite('/Users/stephenraj/Deep_Learning/Elephant.jpeg',superimposed_img)