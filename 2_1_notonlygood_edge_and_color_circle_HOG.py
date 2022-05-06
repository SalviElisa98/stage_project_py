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
df_AllGood = pd.read_csv(path_folder + "/df_notonly_AllGood.csv")

# CREATION OF A CSV FILE
categories = ['20211013 - Flipoff e Cap Parma - 10 - Trasparente - Nero - Oro - 32 mm - 250 ml',             # target = 0  shape 1024, 1280
              '20211013 - Flipoff e Cap Parma - 11 - Trasparente - Nero - Oro - 32 mm - 1000 ml',            # target = 1  shape 1024, 1280
              '20211013 - Flipoff e Cap Parma - 13 - Trasparente - Rosso - Oro - 32 mm - 1000 ml',           # target = 2   shape 1024, 1280
              '20211013 - Flipoff e Cap Parma - 18 - Trasparente - Rosso - Argento - 32 mm - 1000 ml',       # target = 3  shape 1024, 1280
              '20211013 - Flipoff e Cap Parma - 19 - Trasparente - Nero - Argento - 32 mm - 1000 ml',        # target = 4  shape 1024, 1280
              '20211013 - Flipoff e Cap Parma - 20  - Alluminio - Anello - Alluminio - 32 mm - 1000 ml',     # target = 5   shape 1024, 1280
              '20211013 - Flipoff e Cap Parma - 21 - Arancio - 32 mm 1000 ml',                               # target = 6  shape 1024, 1280
              '20211110 - Flipoff e Cap Parma',                                                              # target = 7  shape 544, 707
              '20211217 - FlipOff Arancio con scritta (1)']                                                  # target = 8 shape 600,800

# INITIALIZATION
list_area = []
x_circle_list = []
y_circle_list = []
r_circle_list = []

# Canny Edge Detection and HoughCircles:
for trg in range(9):    # NB 7 e 8 hanno grandezza differente
    df_Target = df_AllGood.loc[(df_AllGood['Target'] == trg)]   # select the category
    for image_path in df_Target['Path']:  # for each image in the directory
        img_array = cv.imread(image_path, cv.IMREAD_COLOR)  # read each image in b&w

    # Canny Edge Detection:
        if trg == 8:   # target = 8 shape 600,800
            adjusted = cv.convertScaleAbs(img_array, alpha=3.0, beta=0)  # ADD CONTRAST: alpha = Contrast control (1.0 - 3.0), beta = Brightness control (0 - 100)
            edges = cv.Canny(adjusted, 100, 200)
            edges = cv.dilate(edges, None)

        else:
            adjusted = cv.convertScaleAbs(img_array, alpha=2.3, beta=0)
            edges = cv.Canny(adjusted, 300, 450)  # High and low threshold value of intensity gradient. The High value dictates that any contrast above its value will be immediately classified as an edge
            edges = cv.dilate(edges, None)        # make edges them more pronounced
            edges = cv.erode(edges, None)

        # save images in EdgeDetection folder:
        name_list = image_path.split('/')
        cv.imwrite(path_folder + '/images_notonlygood/prediction/EdgeDetection/' + str(trg) + '/' + name_list[-1], edges)

    # HoughCircles:
        cimg = cv.cvtColor(img_array, cv.IMREAD_COLOR)
        if trg == 7 or trg == 8:  # target = 7  shape 544,707   # target = 8 shape 600,800
            circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1, 500, param1=50, param2=30, minRadius=20, maxRadius=400)   # param 1 t is the higher threshold of the two passed to the Canny edge detector
        else:
            circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1, 500, param1=50, param2=30, minRadius=200, maxRadius=400)  # param 1 t is the higher threshold of the two passed to the Canny edge detector
        circles = np.uint16(np.rint(circles))           # or circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)  # draw the outer circle
            cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)     # draw the center of the circle
            area = 3.14159 * i[2] * i[2]                        # circle area = pi * r^2
            list_area.append(np.around(area, 2))
            x_circle_list.append(i[0])
            y_circle_list.append(i[1])
            r_circle_list.append(i[2])
        cv.imwrite(path_folder + '/images_notonlygood/prediction/HoughCircles/' + str(trg) + '/' + name_list[-1], cimg)

# SAVE IN THE DATASET THE AREA OF THE CAP, THE COORDINATES AND RADIUS OF THE CIRCLE DETECTED ON THE CAP
df_AllGood["CircleArea"] = list_area
df_AllGood["x_circle"] = x_circle_list
df_AllGood["y_circle"] = y_circle_list
df_AllGood["r_circle"] = r_circle_list
df_AllGood.to_csv(path_folder + "/df_notonly_AllGood.csv", index=False)   # add column area in file csv


# CALCULATE THE AVERAGE COLOR OF THE 80% OF THE CAP MAKING A MASK:
list_b_mean = []
list_g_mean = []
list_r_mean = []
list_b_var = []
list_g_var = []
list_r_var = []
list_b_std = []
list_g_std = []
list_r_std = []
list_HOG = []
list_HOG_max = []
list_HOG_min = []
list_HOG_mean = []
list_HOG_std = []
des_list = []
des_list_mean = []
des_list_std = []
features = []

for trg in range(9):    # NB 7 e 8 hanno grandezza differente
    df_Target = df_AllGood.loc[(df_AllGood['Target'] == trg)]   # select the category
    for image_path, x, y, r in zip(df_Target["Path"], df_Target["x_circle"], df_Target["y_circle"], df_Target["r_circle"]):
        img_original = cv.imread(image_path, cv.IMREAD_COLOR)
        img = img_original.copy()
        img_forhog = img_original.copy()

    # MASK WHITE FILLED CIRCLE
        r2 = np.rint(r * 4 / 5)    # take the 80% of the radius in order to consider only the internal part of the cap
        cv.circle(img, (x, y), int(r2), (255, 255, 255), -1)   # -1 significa che mi colora l'interno del cerchio di bianco
        name_list = image_path.split('/')
        cv.imwrite(path_folder + '/images_notonlygood/prediction/80%mask/' + str(trg) + '/' + name_list[-1], img)   #save in directory mask
        # ora devo trovare dove sono i pixel bianchi (cioè tutte le coordinate all'interno del cerchio)
        argwhere = np.argwhere(img == 255)    # [[x, y, chan], ...]

    # MEAN COLOR IN THE CIRCLE
        unique_coord = argwhere[range(0, len(argwhere), 3), 0:2]
        coord = unique_coord[:, 0], unique_coord[:, 1]
        intensity_values_from_original = img_original[coord]   # all intensity in circle  BGR
        b_mean = np.rint(np.mean(intensity_values_from_original[:, 0]))
        g_mean = np.rint(np.mean(intensity_values_from_original[:, 1]))
        r_mean = np.rint(np.mean(intensity_values_from_original[:, 2]))
        list_b_mean.append(b_mean)
        list_g_mean.append(g_mean)
        list_r_mean.append(r_mean)
    # COLOR VARIANCE IN THE CIRCLE
        b_var = np.rint(np.var(intensity_values_from_original[:, 0]))
        g_var = np.rint(np.var(intensity_values_from_original[:, 1]))
        r_var = np.rint(np.var(intensity_values_from_original[:, 2]))
        list_b_var.append(b_var)
        list_g_var.append(g_var)
        list_r_var.append(r_var)
    # COLOR STANDARD DEVIATION IN THE CIRCLE
        b_std = np.rint(np.std(intensity_values_from_original[:, 0]))
        g_std = np.rint(np.std(intensity_values_from_original[:, 1]))
        r_std = np.rint(np.std(intensity_values_from_original[:, 2]))
        list_b_std.append(b_std)
        list_g_std.append(g_std)
        list_r_std.append(r_std)

    # MASK: ONLY THE 80% CAP, OTHER BLACK
        mask = np.zeros_like(img_forhog)  # draw filled circle in white on black background as mask
        mask = cv.circle(mask, (x, y), int(r2), (255, 255, 255), -1)   # -1 significa che mi colora l'interno del cerchio di bianco  (80% of the radius)
        res = cv.bitwise_and(img_forhog, mask)   # apply mask to image  -> black outside the cap
        result = res.copy()    # for HOG
        img_SIFT = res.copy()  # for SIFT
        img_texture = res.copy()  # for TEXTURE FEATURE with mahotas

        # HOG FEATURE (Histogram of Oriented Gradients)
        feature, hog_image = hog(result, orientations=9, pixels_per_cell=(16, 16), feature_vector=True, cells_per_block=(2, 2), block_norm='L2', visualize=True, multichannel=True)
        feature2 = feature.ravel()  # numpy.ravel() <==> numpy.reshape(-1) -> crea un array da una lista di liste (riduce la dimensione)
        list_HOG.append(feature2)
        list_HOG_max.append(max(feature2))
        list_HOG_mean.append(np.mean(feature2))
        list_HOG_std.append(np.std(feature2))
        feature_nozero = feature2[feature2 != 0.]
        list_HOG_min.append(min(feature_nozero))   # do not consider 0!
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))   # Rescale histogram for better display
        plt.imsave(path_folder + '/images_notonlygood/prediction/HOG/' + str(trg) + '/' + name_list[-1], hog_image_rescaled, cmap="gray")

    #  SIFT (Scale-Invariant Feature Transform) KEY POINT   https://docs.opencv.org/3.4/da/df5/tutorial_py_sift_intro.html
        gray = cv.cvtColor(img_SIFT, cv.COLOR_BGR2GRAY)
        sift = cv.SIFT_create()
        kp = sift.detect(gray, None)
        imgsift = cv.drawKeypoints(gray, kp, img_SIFT, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imsave(path_folder + '/images_notonlygood/prediction/SIFT/' + str(trg) + '/' + name_list[-1], imgsift, cmap="gray")
        kp, des = sift.compute(gray, kp)   # computes the descriptors from the keypoints we have found, 'kp' will be a list of keypoints and 'des' is a numpy array of shape (Number of Keypoints)×128
        descriptors = des.flatten()
        des_list.append(descriptors)
        des_list_mean.append(np.mean(descriptors))
        des_list_std.append(np.std(descriptors))

    # TEXTURE FEATURE with mahotas
        gray = cv.cvtColor(img_texture, cv.COLOR_BGR2GRAY)
        textures = mahotas.features.haralick(gray)       # calculate haralick texture features for 4 types of adjacency
        ht_mean = textures.mean(axis=0)                  # take the mean of it
        features.append(ht_mean)                         # appends the 13-dim feature vector to the training features list.


# SAVE IN THE DATASET THE MEAN COLOR OF THE CAP IN BGR:
df_AllGood['b_mean_circle'] = list_b_mean  # b_mean, g_mean, r_mean -> mean color in the circle
df_AllGood['g_mean_circle'] = list_g_mean
df_AllGood['r_mean_circle'] = list_r_mean
df_AllGood['b_var_circle'] = list_b_var  # b_var, g_var, r_var -> variance color in the circle
df_AllGood['g_var_circle'] = list_g_var
df_AllGood['r_var_circle'] = list_r_var
df_AllGood['b_std_circle'] = list_b_std  # b_std, g_std, r_std -> standard deviation color in the circle
df_AllGood['g_std_circle'] = list_g_std
df_AllGood['r_std_circle'] = list_r_std

df_AllGood['HOG'] = list_HOG            # for each image there is an array of 170k number -> not all 0, I swear!
df_AllGood['HOG_max'] = list_HOG_max    # max of the array of HOG
df_AllGood['HOG_min'] = list_HOG_min    # min of the array of HOG (do not consider 0!)
df_AllGood['HOG_mean'] = list_HOG_mean  # mean of the array of HOG
df_AllGood['HOG_std'] = list_HOG_std    # std of the array of HOG

df_AllGood['SIFT'] = des_list              # SIFT (Scale-Invariant Feature Transform) KEY POINT
df_AllGood['SIFT_mean'] = des_list_mean    # SIFT mean
df_AllGood['SIFT_std'] = des_list_std      # SIFT std

df_AllGood['TEXTURE'] = features          # TEXTURE FEATURE mean with mahotas

df_AllGood.to_csv(path_folder + "/df_notonly_AllGood.csv", index=False)   # add column area in file csv

