# ELISA SALVI stage
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, Lambda
from os import listdir
from os.path import isfile, join
import shutil
from shutil import copyfile, copy2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tkinter as tk
import pandas as pd
import numpy as np
import seaborn as sns
import os
import tensorflow as tf
from tkinter import *
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
from pygame import mixer
from tkinter import messagebox
from sklearn.metrics import classification_report


import cv2 as cv  # https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
from sklearn import svm  # https://scikit-learn.org/stable/modules/svm.html
# https://medium.com/analytics-vidhya/image-classification-using-machine-learning-support-vector-machine-svm-dc7a0ec92e01
import os
from skimage.transform import resize
from skimage.io import imread


path_AllGood = '/Users/davidesalvi/Desktop/stage_project_py/images_notonlygood/AllGood'              # path folder AllGood


# CREATION OF A CSV FILE
categories = ['20211013 - Flipoff e Cap Parma - 10 - Trasparente - Nero - Oro - 32 mm - 250 ml',  # target = 0  shape 1024, 1280
              '20211013 - Flipoff e Cap Parma - 11 - Trasparente - Nero - Oro - 32 mm - 1000 ml',   # target = 1  shape 1024, 1280
              '20211013 - Flipoff e Cap Parma - 13 - Trasparente - Rosso - Oro - 32 mm - 1000 ml',  # target = 2   shape 1024, 1280
              '20211013 - Flipoff e Cap Parma - 18 - Trasparente - Rosso - Argento - 32 mm - 1000 ml',  # target = 3  shape 1024, 1280
              '20211013 - Flipoff e Cap Parma - 19 - Trasparente - Nero - Argento - 32 mm - 1000 ml',   # target = 4  shape 1024, 1280
              '20211013 - Flipoff e Cap Parma - 20  - Alluminio - Anello - Alluminio - 32 mm - 1000 ml',    # target = 5   shape 1024, 1280
              '20211013 - Flipoff e Cap Parma - 21 - Arancio - 32 mm 1000 ml',  # target = 6  shape 1024, 1280
              '20211110 - Flipoff e Cap Parma', # target = 7  shape 544, 707
              '20211217 - FlipOff Arancio con scritta (1)']     # target = 8 shape 600,800

flat_data_arr = []      # input array
target_arr = []         # output array
paths = []

for cat in categories:
    print(f'loading... category : {cat}')
    path = os.path.join(path_AllGood, cat)   # for each subdirectory in the directory AllGood
    for img in os.listdir(path):  # for each image in each directory
        if img.split(".")[-1].lower() in {"jpeg", "jpg", "png"}: # to avoide the creation of the files .DS_Store (only MacOS problem)
            image_path = os.path.join(path, img)
            img_array = cv.imread(image_path, cv.IMREAD_COLOR)  # read each image
            paths.append(image_path)
            #img_resized = resize(img_array,(150,150,3)) # resize?
            flat_data_arr.append(img_array.flatten())  # change in img_resized in case of resize
            target_arr.append(categories.index(cat))  # target = from 0 to 8 (identify the category)
    print(f'loaded category:{cat} successfully')


flat_data = np.array(flat_data_arr)
target = np.array(target_arr)      # target = from 0 to 8 (identify the category)
dfAllGood = pd.DataFrame()         # creation of a dataframe
dfAllGood['flat_data'] = flat_data
dfAllGood['Target'] = target
dfAllGood['Path'] = paths

# save shape of images:
height_arr = []
width_arr = []
channels_arr = []
for p in dfAllGood['Path']:
    if not p.startswith('.'):
        p = str(p)
        i = cv.imread(p, cv.IMREAD_COLOR)
        image_height, image_width, image_channels = i.shape
        height_arr.append(image_height)
        width_arr.append(image_width)
        channels_arr.append(image_channels)
dfAllGood['height'] = height_arr
dfAllGood['width'] = width_arr
dfAllGood['channels'] = channels_arr


dfAllGood.to_csv('df_notonly_AllGood.csv')


print('stop')
