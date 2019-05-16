# -*- coding: utf-8 -*-
"""

@author: karan
"""

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

#load the test data into a separate test file with each individual
#.png image

testPath = loadmat('../test_32x32.mat')
x_test = testPath['X']
y_test = testPath['y']
print(type(x_test))
len(x_test[:,:,:])

x_test.shape

plt.imshow(x_test[:,:,:,1])

y_test[1]

print (len(range(26031)))


for img in range(26031):
    plt.imsave('../test/{}.png'.format(img),x_test[:,:,:,img])
    
#load each individual train pic into a train file with each
    #individual .png image
    
dataPath = loadmat('../train_32x32.mat')
x_train = dataPath['X']
y_train = dataPath['y']
print(type(x_train))
len(x_train[:,:,:])

x_train.shape

plt.imshow(x_train[:,:,:,5])

y_train[5]

for img in range(70000):
    plt.imsave('../train/{}.png'.format(img),x_train[:,:,:,img])
    
y_train[8]