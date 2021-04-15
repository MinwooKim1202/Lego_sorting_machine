#!/usr/bin/env python
# coding: utf-8

# In[109]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
get_ipython().run_line_magic('matplotlib', 'inline')

from tensorflow.keras.layers import Input, Activation, Dense, Flatten, RepeatVector, Reshape
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras import optimizers
import cv2


# In[110]:


import glob
import random


# In[107]:


images_resize = glob.glob('dataset/valid/base/*.jpg')
j = 0


# In[108]:


for fname_resize in images_resize:
    img = cv2.imread(fname_resize)
    img = cv2.resize(img, dsize=(197, 197), interpolation=cv2.INTER_LINEAR)
    file_name = 'resize_img/img' + str(j) + '.jpg'
    cv2.imwrite(file_name, img)
    j = j + 1
    if j > 1001:
        break


# In[111]:


def fill(img, h, w):
    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img


# In[112]:


def rotation(img, angle):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img


# In[113]:


def horizontal_shift(img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = w*ratio
    if ratio > 0:
        img = img[:, :int(w-to_shift), :]
    if ratio < 0:
        img = img[:, int(-1*to_shift):, :]
    img = fill(img, h, w)
    return img


# In[114]:


def vertical_flip(img, flag):
    if flag:
        return cv2.flip(img, 0)
    else:
        return img


# In[115]:


def horizontal_flip(img, flag):
    if flag:
        return cv2.flip(img, 1)
    else:
        return img


# In[132]:


images = glob.glob('dataset/train/1x1_brick/*.jpg')
i = 401


# In[133]:


for fname in images:
    img = cv2.imread(fname)
    img = cv2.resize(img, dsize=(197, 197), fx=0.3, fy=0.7, interpolation=cv2.INTER_LINEAR)
    img = rotation(img, 10)
    img = vertical_flip(img, 1)
    img = horizontal_flip(img, 1)
    #img = horizontal_shift(img, 0.2)
    file_name = 'aug_image/img' + str(i) + '.jpg'
    cv2.imwrite(file_name, img)
    i = i + 1
    if i > 1000:
        break

