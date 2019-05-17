# -*- coding: utf-8 -*-
"""

@author: karan bhargava
"""

import re
import keras
from sklearn.metrics import classification_report
from keras.models import Sequential#
from keras.layers import Dense, Dropout, Flatten#
from keras.layers import Conv2D, MaxPooling2D#
from keras.optimizers import SGD
from keras.datasets import mnist
import tensorflow as tf
import keras.utils#
from keras.utils import *

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import warnings
import os
from os import listdir, getcwd
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from skimage.color import rgb2gray
from skimage.filters import roberts


#IMPORT DATA, CHANGE TO GRAYSCALE, AND RESHAPE IT

trainPath = ('../train/')
dataPath = loadmat('../train_32x32.mat')
y_train_data = dataPath['y']
data= listdir(trainPath)
#x_train.sort()
#print(x_train)

#sorting code from stackoverflow to deal with listdir's bizarre sorting
#https://stackoverflow.com/questions/4813061/non-alphanumeric-list-order-from-os-listdir

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

x_train = sorted_alphanumeric(data)

print(x_train)

#create the y train label data
y_train = []

for i in range(10000):
    y_train.append(y_train_data[i])
    
y_train = np.array(y_train)

x_train[1]

x_train_orig = []
x_train_arr = []

#transform 10000 images into the train data
stopper = 0
for pics in x_train:
    if stopper == 10000:
        break
    else:
        pic = cv2.imread(trainPath+pics)
        x_train_orig.append(pic)
        gray_pic = rgb2gray(pic)
        x_train_arr.append(gray_pic)
        stopper = stopper + 1
    
x_train_orig = np.array(x_train_orig)
x_train_arr = np.array(x_train_arr)

x_train_orig.shape
x_train_arr.shape

plt.imshow(x_train_orig[8])

print(y_train[8])

x_train_arr[8]

plt.imshow(x_train_arr[8], cmap = 'gray')

x_train_arr.shape

x_train4d = x_train_arr.reshape(10000,32,32,1)

##CHANGE STREETVIEW TEST IMAGES TO CONFORM TO CNN

testPath = ('../test/')
testdataPath = loadmat('../test_32x32.mat')
y_test_data = testdataPath['y']
data= listdir(testPath)
x_test = sorted_alphanumeric(data)
print(x_test)

y_test = []

for i in range(26031):
    y_test.append(y_test_data[i])
    
y_test = np.array(y_test)
x_test[1]


x_test_orig = []
x_test_arr = []

#pull test data, convert to grayscale and store into a test array
for pics in x_test:
    pic = cv2.imread(testPath+pics)
    x_test_orig.append(pic)
    gray_pic = rgb2gray(pic)
    x_test_arr.append(gray_pic)
    
x_test_orig = np.array(x_test_orig)
x_test_arr = np.array(x_test_arr)

x_test_orig.shape

plt.imshow(x_test_orig[8])

print(y_test[8])

plt.imshow(x_test_arr[8], cmap = 'gray')

x_test_arr[8].shape

x_test_arr.shape

x_test4d = x_test_arr.reshape(26031,32,32,1)

#CHANGE ALL Y TRAIN AND Y TEST VALUES THAT HAVE A 
#LABEL '10' TO '0'

y_train[y_train > 9] = 0
y_test[y_test > 9] = 0

#RUN VISUALIZATIONS ON IMAGES TO UNDERSTAND
#IMAGE DISTRIBUTION, NOISE, ETC
data_matrix = x_train_arr
data_matrix.shape

#create array that will become 32x32
mean_matrix = x_train_arr.reshape(10000, 32 * 32)
mean_matrix.shape

len(range(1024))
mean_matrix.shape

comparing_matrix = mean_matrix.sum(axis = 0)
comparing_matrix.shape

for val in range(1024):
    comparing_matrix[val] = comparing_matrix[val]/10000

comparing_matrix

visualizing_mean_matrix = comparing_matrix.reshape(32,32)

#Visual of the mean of all the individual mean pixel values of the images in the dataset
plt.imshow(visualizing_mean_matrix, cmap = 'gray')

new_mean_matrix = mean_matrix

new_mean_matrix.shape

new_mean_matrix = data_matrix - visualizing_mean_matrix    
new_mean_matrix

new_mean_matrix = new_mean_matrix.reshape(10000,32*32)

normalized_mean_matrix = preprocessing.normalize(new_mean_matrix, norm = 'l2', axis = 1)

normalized_mean_matrix = normalized_mean_matrix.reshape(10000,32,32)

plt.imshow(normalized_mean_matrix[1], cmap = 'gray')

#NOW APPLY NORMALIZATION ON TEST DATA

test_data_matrix = x_test_arr
test_data_matrix.shape

#create array that will become 32x32
test_mean_matrix = x_test_arr.reshape(26031, 32 * 32)
test_mean_matrix.shape

test_comparing_matrix = test_mean_matrix.sum(axis = 0)
test_comparing_matrix.shape

test_new_mean_matrix = test_mean_matrix

test_new_mean_matrix = test_data_matrix - visualizing_mean_matrix    
test_new_mean_matrix

test_new_mean_matrix = test_new_mean_matrix.reshape(26031,32*32)

test_normalized_mean_matrix = preprocessing.normalize(test_new_mean_matrix, norm = 'l2', axis = 1)

test_normalized_mean_matrix = test_normalized_mean_matrix.reshape(26031,32,32)

#NOW SAVE IMAGES THAT HAVE HAD CONTRAST STRETCHING APPLIED
#pic = 0
#for img in normalized_mean_matrix:
#    plt.imsave('../CS_train/{}.png'.format(pic),img)
#    pic = pic + 1

#pic = 26032
#for img in test_normalized_mean_matrix:
#    plt.imsave('../CS_SV_test/{}.png'.format(pic),img)
#    pic = pic + 1

#ROBERT'S EDGE DETECTION

edge_roberts = roberts(normalized_mean_matrix[1])

fig, ax = plt.subplots(ncols = 2, sharex = True, sharey = True,
                      figsize = (8,4))

ax[0].imshow(edge_roberts, cmap = 'gray')
ax[0].set_title('Roberts edge detection')

ax[1].imshow(normalized_mean_matrix[1], cmap = 'gray')
ax[1].set_title('without edge detection')

plt.tight_layout()
plt.show()

roberts_train_data= []
for img in normalized_mean_matrix:
    edge_detect = roberts(img)
    roberts_train_data.append(edge_detect)
    
roberts_train_data = np.array(roberts_train_data)
roberts_train_data.shape

plt.imshow(roberts_train_data[2])

roberts_test_data = []
for img in test_normalized_mean_matrix:
    test_edge_detect = roberts(img)
    roberts_test_data.append(test_edge_detect)

roberts_test_data = np.array(roberts_test_data)

roberts_test_data.shape

#NOW SAVE ROBERTS DATA TO FOLDER
#pic = 0
#for img in roberts_train_data:
#    plt.imsave('../roberts_train/{}.png'.format(pic),img)
#    pic = pic + 1

#pic = 0
#for img in roberts_test_data:
#    plt.imsave('../roberts_test/{}.png'.format(pic),img)
#    pic = pic + 1

normalized_mean_matrix_4d = normalized_mean_matrix.reshape(10000,32,32,1)
test_normalized_mean_matrix_4d = test_normalized_mean_matrix.reshape(26031,32,32,1)
roberts_normalized_mean_matrix_4d = roberts_train_data.reshape(10000,32,32,1)
test_roberts_normalized_mean_matrix_4d = roberts_test_data.reshape(26031,32,32,1)

#CREATE CNN MODEL AND TRAIN IT
y_train = y_train.flatten()
y_test = y_test.flatten()

y_train.shape
y_test.shape

y_train

type(y_train[1])

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(y_train[1:62])

#Verify model before running it
model = Sequential()#[
  #  keras.layers.Flatten(input_shape = (28, 28))])
model.add(Conv2D(32, kernel_size = (5,5), padding = 'same',
                 activation = 'relu', input_shape = (32,32,1)))
#input_shape = (32,32,1)
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(64, kernel_size = (5,5), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(1024, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

print("[INFO] training network...")
model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop',
             metrics = ['accuracy'])

#fit model to the data
history = model.fit(roberts_normalized_mean_matrix_4d, y_train, batch_size = 16, epochs = 5)
model.evaluate(test_roberts_normalized_mean_matrix_4d, y_test)

#EVALUATE MODEL


print("[INFO] evaluating network...")
predictions = model.predict(test_roberts_normalized_mean_matrix_4d, batch_size = 128)
print(classification_report(y_test.argmax(axis = 1),predictions.argmax(axis = 1)))

plt.plot(history.history['acc'])
plt.plot(history.history['loss'])
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()

#NOW TAKE THE MNIST DATASET AND CHANGE IT TO 32X32

mnist = tf.keras.datasets.mnist

(mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = mnist.load_data()
mnist_x_train, mnist_x_test = mnist_x_train/255.0, mnist_x_test/255.0

#let's look at an image first
first_Image = mnist_x_train[3]
plt.gray()
plt.imshow(first_Image)

mnist_y_test = to_categorical(mnist_y_test, 10)

mnist_x_test.shape

#increase size of the mnist data set to be 32x32
new_mnist_x_test = []
dim = (32,32)
for img in mnist_x_test:
    new_mnist_x_test.append(cv2.resize(img, dim, interpolation = cv2.INTER_AREA))
    
####unsure if the reshaping should occur here
#mnist_x_test = mnist_x_test.reshape(10000,32,32,1)
new_mnist_x_test = np.array(new_mnist_x_test)

new_mnist_x_test.shape

#subtract mean image from MNIST
mean_subtract_mnist = new_mnist_x_test - visualizing_mean_matrix

roberts_mnist_test = []
for img in mean_subtract_mnist:
    robert_edge_mnist = roberts(img)
    roberts_mnist_test.append(robert_edge_mnist)

roberts_mnist_test = np.array(roberts_mnist_test)
roberts_mnist_test.shape

plt.imshow(roberts_mnist_test[1])

mean_subtract_mnist_4d = mean_subtract_mnist.reshape(10000,32,32,1)

roberts_mnist_test_4d = roberts_mnist_test.reshape(10000,32,32,1)

#PERFORM EVALUATION OF MODEL ON MNIST DATA
print("[INFO] evaluating network...")
predictions = model.predict(roberts_mnist_test_4d, batch_size = 128)
print(classification_report(mnist_y_test.argmax(axis = 1),predictions.argmax(axis = 1)))

