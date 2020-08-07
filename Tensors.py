#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 22:43:52 2020

@author: stephenraj
"""

#Scalar - 0 Dim Tensor
import numpy as np
x = np.array(12)
print(x)
print(x.ndim)

#Vector - 1 Dim Tensor
x = np.array([12, 3, 6, 14])
print(x)
print(x.ndim)

#Matrices - 2 Dim Tensor
x = np.array([[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]])
print(x.ndim)

#Cube of numbers - 3 Dim Tensor
x = np.array([[[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]],
              [[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]],
              [[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]]])
print(x.ndim)

#Mnist Dataset
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.ndim)

print(train_images.shape)
print(train_images.dtype)

#Displaying the 4th digit
digit = train_images[4]

import matplotlib.pyplot as plt
plt.imshow(digit,cmap = plt.cm.binary)
plt.show()

#Tensor Slicing
my_slice = train_images[10:100]
print(my_slice.shape)

my_slice = train_images[10:100,:,:]
print(my_slice.shape)

my_slice = train_images[10:100,0:28,0:28]
print(my_slice.shape)

#To select 14 * 14 pixels in Bottom right corner
my_slice = train_images[10:100, 14:,14:]
print(my_slice.shape)

#To select pixels centered in the middle
my_slice = train_images[10:100,7:-7,7:-7]
print(my_slice.shape)

#Relu function
def naive_relu(x):
    assert len(x.shape)== 2
    x = x.copy()
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] = max(x[i,j],0)
    return x

#Addition
def naive_add(x,y):
    assert len(x.shape) == 2
    assert x.shape == y.shape
    
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] += y[i,j]
    return x

#Same in numpy

import numpy as np
x = np.array([2,4])
y = np.array([3,6])
z = x + y
z = np.maximum(z,0.)

#Broadcasting
def naive_add_matrix_and_vector(x,y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] += y[j]
    return x
            
#Broadcasting in Numpy
x = np.random.random((64,3,32,10))
print(x.shape)
y = np.random.random((32,10))
print(y.shape)
z = np.maximum(x,y)
print(z.shape)
print(z)

#Numpy Dot
x = 10
y = 5
z = np.dot(x,y)
print(z)

#Vector Multiplication in Tensor
def naive_vector_dot(x,y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    z=0
    for i in range(x.shape[0]):
        z+= x[i] * y[i]
    return z

def naive_matrix_vector_dot(x,y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i][j] * y[j]
    return z

x= np.array([[2,4],[6,8]])
y = np.array([2,2])
print(np.dot(x,y))

def matrix_dot(x,y):
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]
    
    z= np.zeros((x.shape[0]. y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            row_x = x[i,:]
            column_y = y[:,j]
            z[i,j] = naive_vector_dot(row_x, column_y)
    return z

#Reshaping
x = np.array([[0.,1.], 
             [2.,3.], 
             [4.,5.]])
print(x.shape)

x = x.reshape((6,1))
print(x)

x = x.reshape((2,3))
print(x)

x = np.zeros((300,20))
x = np.transpose(x)
print(x.shape)


            
