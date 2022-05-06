# ELISA SALVI stage
# CLASSIFICATION OF TRANSPARENT CAPOFF, NO COLOR DEFECT!! (but include scratch)
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
from tkinter import messagebox
from sklearn.metrics import classification_report


import cv2 as cv  # https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
from sklearn import svm  # https://scikit-learn.org/stable/modules/svm.html
# https://medium.com/analytics-vidhya/image-classification-using-machine-learning-support-vector-machine-svm-dc7a0ec92e01
import os
from skimage.transform import resize
from skimage.io import imread
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
from mahotas import *
import mahotas.demos

path_folder = '/Users/davidesalvi/Desktop/stage_project_py'       # path folder stage_project_py
df_notonly_AllGood = pd.read_csv(path_folder + "/df_notonly_AllGood.csv")

df_trasparent = df_notonly_AllGood[df_notonly_AllGood['Target'] < 5]   # ONLY TRASPARENT CAPOFF
df_trasparent = df_trasparent.iloc[:, :-2]

def f(df_trasparent): # create a new column 'New Target' from 0 to 4 to grouped the Black-Gold of 250 ml and 1000ml that are equal
    val = 0
    if df_trasparent["Target"] == 1:
        val = 0
    elif df_trasparent["Target"] == 2:
        val = 1
    elif df_trasparent["Target"] == 3:
        val = 2
    elif df_trasparent["Target"] == 4:
        val = 3
    return val
df_trasparent['New Target'] = df_trasparent.apply(f, axis=1)

#names = ['Black-Gold-250ml-1000ml', 'Red-Gold', 'Red-Silver', 'Black-Silver']
def name(df_trasparent): # create a new column 'New Target Name'
    name = 'Black-Gold-250ml-1000ml'
    if df_trasparent["New Target"] == 1:
        name = 'Red-Gold'
    elif df_trasparent["New Target"] == 2:
        name = 'Red-Gold'
    elif df_trasparent["New Target"] == 3:
        name = 'Red-Silver'
    elif df_trasparent["New Target"] == 4:
        name = 'Black-Silver'
    return name
df_trasparent['New Target Name'] = df_trasparent.apply(name, axis=1)

df_trasparent.to_csv(path_folder + "/df_trasparent_notonly.csv", index=False)   # add column area in file csv