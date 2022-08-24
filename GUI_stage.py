# ELISA SALVI IT-CODING PT. 2    NB se si cambiano le feature useful nei file 3 e 4, bisogna cambiarlo anche nelle funz. FEATURE_EXTRACTION
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, Lambda
from os import listdir
from os.path import isfile, join
import shutil
from time import time
from shutil import copyfile, copy2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tkinter as tk
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import seaborn as sns
import os
import tensorflow as tf
from tkinter import *
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import glob
#from pygame import mixer
from tkinter import messagebox, PhotoImage
from sklearn.metrics import classification_report
import csv
import os
import random
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.image as mpimg
from pandastable import Table
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sklearn.model_selection as model_selection
import cv2 as cv
#from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer

path_folder = '/Users/davidesalvi/Desktop/stage_project_py'       # path folder stage_project_py
df_AllGood = pd.read_csv(path_folder + "/df_notonly_AllGood.csv")
path_prediction = path_folder + "/images_notonlygood/prediction"
path_how_work = path_folder + "/images_notonlygood/HOW_WORK"
path_test_models = path_folder + "/images_notonlygood/test_models"
model_accuracy = pd.read_csv(path_test_models + "/model_accuracy.csv")
predizioni_test = pd.read_csv(path_test_models + "/predizioni_test.csv")
importances_linreg_tot = pd.read_csv(path_test_models + "/importances_tot.csv")

df_AllGood_transparent = pd.read_csv(path_folder + "/df_trasparent_notonly.csv")
path_test_models_tras = path_folder + "/images_notonlygood/test_models_trasparent"
model_accuracy_transparent = pd.read_csv(path_test_models_tras + "/model_accuracy_trasparent.csv")
predizioni_test_transparent = pd.read_csv(path_test_models_tras + "/predizioni_test_trasparent.csv")
importances_linreg_tot_transparent = pd.read_csv(path_test_models_tras + "/importances_tot_trasparent.csv")
image_dir_NN = path_folder + "/images_notonlygood/allimagesNN"

# IMAGE ANALYSIS
def EDA():
    root = tk.Tk()
    root.title('IMAGE ANALYSIS')
    canvas1 = tk.Canvas(root, width=700, height=600)
    canvas1.configure(bg='RoyalBlue4')
    canvas1.pack()

    buttonn = tk.Button(root, text='BACK', command=root.destroy, fg='red', highlightbackground='red')
    canvas1.create_window(350, 550, window=buttonn)

    lbl = ['Trasparente-Nero\nOro-250 ml',  # target = 0  shape 1024, 1280
           'Trasparente-Nero-Oro-1000 ml',   # target = 1  shape 1024, 1280
           'Trasparente-Rosso-Oro',  # target = 2   shape 1024, 1280
           'Trasparente-Rosso-Argento',  # target = 3  shape 1024, 1280
           'Trasparente-Nero-Argento',   # target = 4  shape 1024, 1280
           'Alluminio-Anello-Alluminio',    # target = 5   shape 1024, 1280
           'FlipOff Arancio',  # target = 6  shape 1024, 1280
           'Flipoff Blu', # target = 7  shape 544, 707
           'FlipOff Arancio con scritta']  # target = 8

    def pieplot():
        colors = ['darkorchid', 'aqua', 'lightpink', 'red', 'green', 'coral', 'yellow', 'blue', 'orange']
        count_list_targ = df_AllGood.groupby('Target').count()   # count how many images for each category
        my_dict = {'NAME':lbl, 'Count':count_list_targ['Path']}
        df = pd.DataFrame(data=my_dict)
        df.plot.pie(title="THE DATASET", y='Count', figsize=(8, 5), labeldistance=1.2, shadow=True, colors=colors, labels=lbl, legend=None, startangle=90, autopct='%1.1f%%', explode=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)).get_figure()
        plt.show()
    button1 = tk.Button(root, text='THE DATASET', command=pieplot, fg='red', highlightbackground='red')
    canvas1.create_window(350, 90, window=button1)

    # EDGE DETECTION
    def edgedetection():
        folder_edgedetection = path_folder + '/images_notonlygood/prediction/EdgeDetection'
        images = []  # list of path, one random image for each category
        for n in range(9):
            path = folder_edgedetection + '/%s'%n
            files = os.listdir(path)  # list of images names.png in the directory
            d = random.choice(files)  # name of a random image.png
            img_path = path + '/' + d
            img = mpimg.imread(img_path)
            scale_percent = 85   # zoom or crop the images for a better display (use proportion because every image has a different size)
            xx = int(img.shape[0] * scale_percent / 100)
            x = int(img.shape[0] * (100 - scale_percent) / 100)
            yy = int(img.shape[1] * scale_percent / 100)
            y = int(img.shape[1] * (100 - scale_percent) / 100)
            cropped_image = img[x:xx, y:yy]
            images.append(cropped_image)
        f, axarr = plt.subplots(3, 3, figsize=(12, 7))  # plot 9 random images, one for each type
        [axi.set_axis_off() for axi in axarr.ravel()]  # no axis
        [axi.set_title(lbl[n]) for axi, n in zip(axarr.ravel(), range(9))]  # categories titles
        f.suptitle('EDGE DETECTION', fontweight="bold")
        axarr[0, 0].imshow(images[0], cmap='gray')
        axarr[0, 1].imshow(images[1], cmap='gray')
        axarr[0, 2].imshow(images[2], cmap='gray')
        axarr[1, 0].imshow(images[3], cmap='gray')
        axarr[1, 1].imshow(images[4], cmap='gray')
        axarr[1, 2].imshow(images[5], cmap='gray')
        axarr[2, 0].imshow(images[6], cmap='gray')
        axarr[2, 1].imshow(images[7], cmap='gray')
        axarr[2, 2].imshow(images[8], cmap='gray')
        f.tight_layout()
        plt.show()
    button1 = tk.Button(root, text='EDGE DETECTION', command=edgedetection, fg='red', highlightbackground='red')
    canvas1.create_window(350, 150, window=button1)


    # HOUGH CIRCLE
    def houghcircle():
        folder_HoughCircles = path_folder + '/images_notonlygood/prediction/HoughCircles'
        images = []  # list of path, one random image for each category
        for n in range(9):
            path = folder_HoughCircles + '/%s'%n
            files = os.listdir(path)  # list of images names.png in the directory
            d = random.choice(files)  # name of a random image.png
            img_path = path + '/' + d
            img = mpimg.imread(img_path)
            scale_percent = 88   # zoom or crop the images for a better display (use proportion because every image has a different size)
            xx = int(img.shape[0] * scale_percent / 100)
            x = int(img.shape[0] * (100 - scale_percent) / 100)
            yy = int(img.shape[1] * scale_percent / 100)
            y = int(img.shape[1] * (100 - scale_percent) / 100)
            cropped_image = img[x:xx, y:yy]
            images.append(cropped_image)
        f, axarr = plt.subplots(3, 3, figsize=(12, 7))  # plot 9 random images, one for each type
        [axi.set_axis_off() for axi in axarr.ravel()]  # no axis
        [axi.set_title(lbl[n]) for axi, n in zip(axarr.ravel(), range(9))]  # categories titles
        f.suptitle('HOUGH CIRCLE', fontweight="bold")
        axarr[0, 0].imshow(images[0])
        axarr[0, 1].imshow(images[1])
        axarr[0, 2].imshow(images[2])
        axarr[1, 0].imshow(images[3])
        axarr[1, 1].imshow(images[4])
        axarr[1, 2].imshow(images[5])
        axarr[2, 0].imshow(images[6])
        axarr[2, 1].imshow(images[7])
        axarr[2, 2].imshow(images[8])
        f.tight_layout()
        plt.show()
    button1 = tk.Button(root, text='HOUGH CIRCLE', command=houghcircle, fg='red', highlightbackground='red')
    canvas1.create_window(250, 200, window=button1)

    # 80% MASK
    def mask():
        folder_mask = path_folder + '/images_notonlygood/prediction/80%mask'
        images = []  # list of path, one random image for each category
        for n in range(9):
            path = folder_mask + '/%s'%n
            files = os.listdir(path)  # list of images names.png in the directory
            d = random.choice(files)  # name of a random image.png
            img_path = path + '/' + d
            img = mpimg.imread(img_path)
            scale_percent = 85   # zoom or crop the images for a better display (use proportion because every image has a different size)
            xx = int(img.shape[0] * scale_percent / 100)
            x = int(img.shape[0] * (100 - scale_percent) / 100)
            yy = int(img.shape[1] * scale_percent / 100)
            y = int(img.shape[1] * (100 - scale_percent) / 100)
            cropped_image = img[x:xx, y:yy]
            images.append(cropped_image)
        f, axarr = plt.subplots(3, 3, figsize=(12, 7))  # plot 9 random images, one for each type
        [axi.set_axis_off() for axi in axarr.ravel()]  # no axis
        [axi.set_title(lbl[n]) for axi, n in zip(axarr.ravel(), range(9))]  # categories titles
        f.suptitle('80% MASK', fontweight="bold")
        axarr[0, 0].imshow(images[0])
        axarr[0, 1].imshow(images[1])
        axarr[0, 2].imshow(images[2])
        axarr[1, 0].imshow(images[3])
        axarr[1, 1].imshow(images[4])
        axarr[1, 2].imshow(images[5])
        axarr[2, 0].imshow(images[6])
        axarr[2, 1].imshow(images[7])
        axarr[2, 2].imshow(images[8])
        f.tight_layout()
        plt.show()
    button1 = tk.Button(root, text='80% MASK', command=mask, fg='red', highlightbackground='red')
    canvas1.create_window(420, 200, window=button1)

    # HOG - HISTOGRAM OF ORIENTED GRADIENTS
    def HOG():
        folder_HOG = path_folder + '/images_notonlygood/prediction/HOG'
        images = []  # list of path, one random image for each category
        for n in range(9):
            path = folder_HOG + '/%s'%n
            files = os.listdir(path)  # list of images names.png in the directory
            d = random.choice(files)  # name of a random image.png
            img_path = path + '/' + d
            img = mpimg.imread(img_path)
            scale_percent = 80   # zoom or crop the images for a better display (use proportion because every image has a different size) se 80 vuol dire che si toglie il 20% da sopra e sotto e da destra e sinistra
            xx = int(img.shape[0] * scale_percent / 100)
            x = int(img.shape[0] * (100 - scale_percent) / 100)
            yy = int(img.shape[1] * scale_percent / 100)
            y = int(img.shape[1] * (100 - scale_percent) / 100)
            cropped_image = img[x:xx, y:yy]
            images.append(cropped_image)
        f, axarr = plt.subplots(3, 3, figsize=(12, 7))  # plot 9 random images, one for each type
        [axi.set_axis_off() for axi in axarr.ravel()]  # no axis
        [axi.set_title(lbl[n]) for axi, n in zip(axarr.ravel(), range(9))]  # categories titles
        f.suptitle('HISTOGRAM OF ORIENTED GRADIENTS', fontweight="bold")
        axarr[0, 0].imshow(images[0])
        axarr[0, 1].imshow(images[1])
        axarr[0, 2].imshow(images[2])
        axarr[1, 0].imshow(images[3])
        axarr[1, 1].imshow(images[4])
        axarr[1, 2].imshow(images[5])
        axarr[2, 0].imshow(images[6])
        axarr[2, 1].imshow(images[7])
        axarr[2, 2].imshow(images[8])
        f.tight_layout()
        plt.show()
    button1 = tk.Button(root, text='HISTOGRAM OF ORIENTED GRADIENTS', command=HOG, fg='red', highlightbackground='red')
    canvas1.create_window(330, 250, window=button1)

    def imageHOG():  # VISUALIZE IMAGES TO UNDERSTAND
        img = mpimg.imread(path_how_work + "/hog.png")
        img2 = mpimg.imread(path_how_work + "/hog2.png")
        f, axarr = plt.subplots(1, 2, figsize=(14, 5))
        [axi.set_axis_off() for axi in axarr]  # no axis
        f.suptitle('HOG - HISTOGRAM OF ORIENTED GRADIENTS', fontweight="bold")
        axarr[0].imshow(img, aspect='auto')
        axarr[1].imshow(img2, aspect='auto')
        plt.show()
    button9 = tk.Button(root, text='HOG?', command=imageHOG, fg='red', highlightbackground='red')
    canvas1.create_window(100, 250, window=button9)

    # SIFT- SCALE INVARIANT FEATURE TRANSFORM
    def SIFT():
        folder_SIFT = path_folder + '/images_notonlygood/prediction/SIFT'
        images = []  # list of path, one random image for each category
        for n in range(9):
            path = folder_SIFT + '/%s'%n
            files = os.listdir(path)  # list of images names.png in the directory
            d = random.choice(files)  # name of a random image.png
            img_path = path + '/' + d
            img = mpimg.imread(img_path)
            scale_percent = 80   # zoom or crop the images for a better display (use proportion because every image has a different size)
            xx = int(img.shape[0] * scale_percent / 100)
            x = int(img.shape[0] * (100 - scale_percent) / 100)
            yy = int(img.shape[1] * scale_percent / 100)
            y = int(img.shape[1] * (100 - scale_percent) / 100)
            cropped_image = img[x:xx, y:yy]
            images.append(cropped_image)
        f, axarr = plt.subplots(3, 3, figsize=(12, 7))  # plot 9 random images, one for each type
        [axi.set_axis_off() for axi in axarr.ravel()]  # no axis
        [axi.set_title(lbl[n]) for axi, n in zip(axarr.ravel(), range(9))]  # categories titles
        f.suptitle('SCALE INVARIANT FEATURE TRANSFORM', fontweight="bold")
        axarr[0, 0].imshow(images[0])
        axarr[0, 1].imshow(images[1])
        axarr[0, 2].imshow(images[2])
        axarr[1, 0].imshow(images[3])
        axarr[1, 1].imshow(images[4])
        axarr[1, 2].imshow(images[5])
        axarr[2, 0].imshow(images[6])
        axarr[2, 1].imshow(images[7])
        axarr[2, 2].imshow(images[8])
        f.tight_layout()
        plt.show()
    button1 = tk.Button(root, text='SCALE INVARIANT FEATURE TRANSFORM', command=SIFT, fg='red', highlightbackground='red')
    canvas1.create_window(330, 300, window=button1)

    def imageSIFT():  # VISUALIZE IMAGES TO UNDERSTAND
        img = Image.open(path_how_work + "/SIFT.png")
        img2 = mpimg.imread(path_how_work + "/SIFT2.png")
        f, axarr = plt.subplots(1, 2, figsize=(12, 5))
        [axi.set_axis_off() for axi in axarr]  # no axis
        f.suptitle('SIFT - SCALE INVARIANT FEATURE TRANSFORM', fontweight="bold")
        axarr[0].imshow(img, aspect='auto')
        axarr[1].imshow(img2, aspect='auto')
        plt.show()
    button9 = tk.Button(root, text='SIFT?', command=imageSIFT, fg='red', highlightbackground='red')
    canvas1.create_window(100, 300, window=button9)

    # FEATURE EXTRACTION
    def FEATURE_EXTRACTION():  # show the feature used in a df
        root1 = tk.Tk()
        root1.title('FEATURE EXTRACTION')
        root1.geometry("1200x600")
        df = df_AllGood.iloc[:, :-2]  # tolgo new feature
        frame = tk.Frame(root1)
        frame.pack(fill='both', expand=True, side=TOP)
        pt = Table(frame, dataframe=df)
        pt.show()
    button1 = tk.Button(root, text='FEATURE EXTRACTION', command=FEATURE_EXTRACTION, fg='red', highlightbackground='red')
    canvas1.create_window(330, 350, window=button1)

    from sklearn.preprocessing import StandardScaler
    # CALCULATE THE IMPORTANCE OF ALL THE FEATURE USING A LINEAR REGRESSION (transparent grupped)
    XX = df_AllGood.iloc[:, 2:-2]  # drop list feature and useless feature:
    XX = XX.drop(['Path', 'channels', 'height', 'width', 'HOG', 'SIFT', 'TEXTURE', 'y_circle', 'x_circle', 'Target'], axis=1)
    yy = df_AllGood["Target"]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(XX, yy, train_size=0.70, test_size=0.30, random_state=101)  # divsione randomica in train e test
    ss = StandardScaler()
    X_train_scaled = ss.fit_transform(X_train)

# PLOT IMPORTANCES OF THE FEATURE USING LINEAR REGRESSION
    def importanceslinreg():
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)
        Importance_linreg = model.coef_[0]
        Importance_linreg = abs(Importance_linreg)
        importances_linreg = pd.DataFrame(data={'Attribute': X_train.columns, 'Importance_linreg': Importance_linreg})
        importances_linreg = importances_linreg.sort_values(by='Importance_linreg', ascending=False)
        plt.subplots(figsize=(12, 7))
        plt.bar(x=importances_linreg['Attribute'], height=importances_linreg['Importance_linreg'], color='#087E8B')
        plt.title('Feature importance obtained from linear regression coefficients (abs)\n (no list feature)', size=20, fontweight="bold")
        plt.xticks(rotation=30)
        plt.ylabel('Importance')
        plt.xlabel('Feature')
        plt.show()
    button9 = tk.Button(root, text='FEATURE IMPORTANCE lin_reg', command=importanceslinreg, fg='red', highlightbackground='red')
    canvas1.create_window(330, 400, window=button9)

    # PLOT IMPORTANCES OF THE FEATURE USING TREE-LIKE STRUCTURE
    def importancestree():
        # CALCULATE THE IMPORTANCE OF A FEATURE USING A TREE-BASED MODEL
        from xgboost import XGBClassifier
        model2 = XGBClassifier()
        model2.fit(X_train_scaled, y_train)
        Importance_tree = model2.feature_importances_
        importances_linreg = pd.DataFrame(data={'Attribute': X_train.columns, 'Importance_tree': Importance_tree})
        importances_linreg = importances_linreg.sort_values(by='Importance_tree', ascending=False)

        plt.subplots(figsize=(12, 7))
        sns.barplot(x='Attribute', y='Importance_tree', data=importances_linreg, color='#087E8B',
                order=importances_linreg.sort_values('Importance_tree', ascending=False).Attribute)
        plt.title('Feature importance obtained using XGBClassifier (no list feature)\nExtreme Gradient Boosting Classifier, a tree-based model', size=20, fontweight="bold")
        plt.ylabel('Importance')
        plt.xlabel('Feature')
        plt.xticks(rotation=30)
        #plt.text(1.6, 1.4, 'The larger the coefficient is (both positive and negative),\nthe more influence it has on a prediction.', fontsize=15)
        plt.show()
    button9 = tk.Button(root, text='FEATURE IMPORTANCE XGBClassifier', command=importancestree, fg='red', highlightbackground='red')
    canvas1.create_window(330, 450, window=button9)




# MODELS (transparent grouped)
def models():
    root = tk.Tk()
    root.title('Models Prediction')
    canvas1 = tk.Canvas(root, width=700, height=600)
    canvas1.configure(bg='RoyalBlue4')
    canvas1.pack()

    buttonn = tk.Button(root, text='BACK', command=root.destroy, fg='red', highlightbackground='red')
    canvas1.create_window(350, 550, window=buttonn)

    def pieplot():
        names = ['Flipoff Trasparente', 'Anello Alluminio','Flipoff Arancio', 'Flipoff Blu', 'Flipoff Arancio con Scritta']
        colors = ['aqua', 'lightpink', 'yellow', 'blue', 'orange']
        count_list_targ = df_AllGood.groupby('New Target').count()   # count how many images for each category
        my_dict = {'NAME':names, 'Count':count_list_targ['Path']}
        df = pd.DataFrame(data=my_dict)
        df.plot.pie(title="THE DATASET\n (transparent grouped)", y='Count', figsize=(8, 5), labeldistance=1.2, shadow=True, colors=colors, labels=names, legend=None, startangle=90, autopct='%1.1f%%', explode=(0.1, 0.1, 0.1, 0.1, 0.1)).get_figure()
        plt.show()
    button1 = tk.Button(root, text='THE DATASET\n(transparent grouped)', command=pieplot, fg='red', highlightbackground='red')
    canvas1.create_window(350, 40, window=button1)

    # PLOT IMPORTANCES OF THE FEATURE USING LINEAR REGRESSION
    def importanceslinreg():
        plt.subplots(figsize=(12, 7))
        plt.bar(x=importances_linreg_tot['Attribute'], height=importances_linreg_tot['Importance_linreg'], color='#087E8B')
        plt.title('Feature importance obtained from linear regression coefficients (abs)\n (no list feature)', size=20, fontweight="bold")
        plt.xticks(rotation=30)
        plt.ylabel('Importance')
        plt.xlabel('Feature')
        #plt.text(1.6, 1.4, 'The larger the coefficient is (both positive and negative),\nthe more influence it has on a prediction.', fontsize=15)
        plt.show()
    button9 = tk.Button(root, text='FEATURE IMPORTANCE lin_reg', command=importanceslinreg, fg='red', highlightbackground='red')
    canvas1.create_window(150, 90, window=button9)

    # PLOT IMPORTANCES OF THE FEATURE USING TREE-LIKE STRUCTURE
    def importancestree():
        plt.subplots(figsize=(12, 7))
        sns.barplot(x='Attribute', y='Importance_tree', data=importances_linreg_tot, color='#087E8B',
                    order=importances_linreg_tot.sort_values('Importance_tree', ascending=False).Attribute)
        plt.title('Feature importance obtained using XGBClassifier (no list feature)\nExtreme Gradient Boosting Classifier, a tree-based model', size=20, fontweight="bold")
        plt.ylabel('Importance')
        plt.xlabel('Feature')
        plt.xticks(rotation=30)
        #plt.text(1.6, 1.4, 'The larger the coefficient is (both positive and negative),\nthe more influence it has on a prediction.', fontsize=15)
        plt.show()
    button9 = tk.Button(root, text='FEATURE IMPORTANCE XGBClassifier', command=importancestree, fg='red', highlightbackground='red')
    canvas1.create_window(550, 90, window=button9)

    # RANDOM FOREST
    def RANDOM_FOREST():
        folder_RANDOM_FOREST = path_folder + '/images_notonlygood/test_models/rf_pred'
        files = os.listdir(folder_RANDOM_FOREST)  # list of images names.png in the directory
        images = []  # list of path, one random image for each category
        n = 0
        for n in range(9):
            d = random.choice(files)  # name of a random image.png
            img_path = folder_RANDOM_FOREST + '/' + d
            img = mpimg.imread(img_path)
            images.append(img)
            n += 1
        f, axarr = plt.subplots(3, 3, figsize=(12, 7))  # plot 9 random images, one for each type
        [axi.set_axis_off() for axi in axarr.ravel()]  # no axis
        f.suptitle('RANDOM FOREST', fontweight="bold")
        axarr[0, 0].imshow(images[0])
        axarr[0, 1].imshow(images[1])
        axarr[0, 2].imshow(images[2])
        axarr[1, 0].imshow(images[3])
        axarr[1, 1].imshow(images[4])
        axarr[1, 2].imshow(images[5])
        axarr[2, 0].imshow(images[6])
        axarr[2, 1].imshow(images[7])
        axarr[2, 2].imshow(images[8])
        f.tight_layout()
        plt.show()
    button1 = tk.Button(root, text='RANDOM FOREST', command=RANDOM_FOREST, fg='red', highlightbackground='red')
    canvas1.create_window(330, 150, window=button1)

    def visualize_RF():
        img = Image.open(path_test_models + "/rf_5trees.png")
        plt.figure(figsize=(165, 85))
        plt.imshow(img, aspect='auto')
        plt.title('RANDOM FOREST')
        plt.axis(False)
        plt.show()
    button = tk.Button(root, text='visualize RF', command=visualize_RF, fg='red', highlightbackground='red')
    canvas1.create_window(550, 150, window=button)

    def imageRF():  # VISUALIZE IMAGES TO UNDERSTAND
        img = Image.open(path_how_work + "/RF.png")
        img2 = mpimg.imread(path_how_work + "/rf2.png")
        f, axarr = plt.subplots(1, 2, figsize=(12, 5))
        [axi.set_axis_off() for axi in axarr]  # no axis
        f.suptitle('RANDOM FOREST', fontweight="bold")
        axarr[0].imshow(img, aspect='auto')
        axarr[1].imshow(img2, aspect='auto')
        plt.show()
    button9 = tk.Button(root, text='RANDOM FOREST?', command=imageRF, fg='red', highlightbackground='red')
    canvas1.create_window(100, 150, window=button9)

    # DECISION TREE
    def DECISION_TREE():
        folder_DECISION_TREE = path_folder + '/images_notonlygood/test_models/DT_pred'
        files = os.listdir(folder_DECISION_TREE)  # list of images names.png in the directory
        images = []  # list of path, one random image for each category
        n = 0
        for n in range(9):
            d = random.choice(files)  # name of a random image.png
            img_path = folder_DECISION_TREE + '/' + d
            img = mpimg.imread(img_path)
            images.append(img)
            n += 1
        f, axarr = plt.subplots(3, 3, figsize=(12, 7))  # plot 9 random images, one for each type
        [axi.set_axis_off() for axi in axarr.ravel()]  # no axis
        f.suptitle('DECISION TREE', fontweight="bold")
        axarr[0, 0].imshow(images[0])
        axarr[0, 1].imshow(images[1])
        axarr[0, 2].imshow(images[2])
        axarr[1, 0].imshow(images[3])
        axarr[1, 1].imshow(images[4])
        axarr[1, 2].imshow(images[5])
        axarr[2, 0].imshow(images[6])
        axarr[2, 1].imshow(images[7])
        axarr[2, 2].imshow(images[8])
        f.tight_layout()
        plt.show()
    button1 = tk.Button(root, text='DECISION TREE', command=DECISION_TREE, fg='red', highlightbackground='red')
    canvas1.create_window(330, 200, window=button1)

    def visualize_DT():
        img = Image.open(path_test_models + "/tree.png")
        plt.figure(figsize=(165, 85))
        plt.imshow(img, aspect='auto')
        plt.title('DECISION TREE')
        plt.axis(False)
        plt.show()
    button = tk.Button(root, text='visualize DT', command=visualize_DT, fg='red', highlightbackground='red')
    canvas1.create_window(550, 200, window=button)


    def imagedt():  # VISUALIZE IMAGES TO UNDERSTAND
        img = Image.open(path_how_work + "/dt.png")
        img2 = mpimg.imread(path_how_work + "/dt2.png")
        f, axarr = plt.subplots(1, 2, figsize=(12, 5))
        [axi.set_axis_off() for axi in axarr]  # no axis
        f.suptitle('DECISION TREE', fontweight="bold")
        axarr[0].imshow(img, aspect='auto')
        axarr[1].imshow(img2, aspect='auto')
        plt.show()
    button9 = tk.Button(root, text='DECISION TREE?', command=imagedt, fg='red', highlightbackground='red')
    canvas1.create_window(100, 200, window=button9)

    # KNN
    def KNN():
        folder_KNN = path_folder + '/images_notonlygood/test_models/knn_pred'
        files = os.listdir(folder_KNN)  # list of images names.png in the directory
        images = []  # list of path, one random image for each category
        n = 0
        for n in range(9):
            d = random.choice(files)  # name of a random image.png
            img_path = folder_KNN + '/' + d
            img = mpimg.imread(img_path)
            images.append(img)
            n += 1
        f, axarr = plt.subplots(3, 3, figsize=(12, 7))  # plot 9 random images, one for each type
        [axi.set_axis_off() for axi in axarr.ravel()]  # no axis
        f.suptitle('K-NEAREST NEIGHBORS', fontweight="bold")
        axarr[0, 0].imshow(images[0])
        axarr[0, 1].imshow(images[1])
        axarr[0, 2].imshow(images[2])
        axarr[1, 0].imshow(images[3])
        axarr[1, 1].imshow(images[4])
        axarr[1, 2].imshow(images[5])
        axarr[2, 0].imshow(images[6])
        axarr[2, 1].imshow(images[7])
        axarr[2, 2].imshow(images[8])
        f.tight_layout()
        plt.show()
    button1 = tk.Button(root, text='K-NEAREST NEIGHBORS', command=KNN, fg='red', highlightbackground='red')
    canvas1.create_window(330, 250, window=button1)

    def confusionmatrix_knn():   # plot confusion matrix for knn
        img = Image.open(path_test_models + "/knn_confusion_matrix.png")
        plt.figure(figsize=(80, 100))
        plt.imshow(img, aspect='auto')
        plt.axis(False)
        plt.show()
    button = tk.Button(root, text='confusion matrix', command=confusionmatrix_knn, fg='red', highlightbackground='red')
    canvas1.create_window(540, 250, window=button)

    def eda():  # EDA knn
        img = Image.open(path_test_models + "/eda.png")
        plt.figure(figsize=(80, 100))
        plt.imshow(img, aspect='auto')
        plt.axis(False)
        plt.show()
    button = tk.Button(root, text='EDA', command=eda, fg='red', highlightbackground='red')
    canvas1.create_window(650, 250, window=button)

    def imageknn():  # VISUALIZE IMAGES TO UNDERSTAND
        img = Image.open(path_how_work + "/knn.png")
        img2 = mpimg.imread(path_how_work + "/knn2.png")
        f, axarr = plt.subplots(1, 2, figsize=(12, 5))
        [axi.set_axis_off() for axi in axarr]  # no axis
        f.suptitle('K-NEAREST NEIGHBORS', fontweight="bold")
        axarr[0].imshow(img, aspect='auto')
        axarr[1].imshow(img2, aspect='auto')
        plt.show()
    button9 = tk.Button(root, text='KNN?', command=imageknn, fg='red', highlightbackground='red')
    canvas1.create_window(100, 250, window=button9)


    # 3 DEGREE POLYNOMIAL SVM
    def POLY_SVM():
        folder_POLY_SVM = path_folder + '/images_notonlygood/test_models/poly_pred'
        files = os.listdir(folder_POLY_SVM)  # list of images names.png in the directory
        images = []  # list of path, one random image for each category
        n = 0
        for n in range(9):
            d = random.choice(files)  # name of a random image.png
            img_path = folder_POLY_SVM + '/' + d
            img = mpimg.imread(img_path)
            images.append(img)
            n += 1
        f, axarr = plt.subplots(3, 3, figsize=(12, 7))  # plot 9 random images, one for each type
        [axi.set_axis_off() for axi in axarr.ravel()]  # no axis
        f.suptitle('3 DEGREE POLYNOMIAL SVM', fontweight="bold")
        axarr[0, 0].imshow(images[0])
        axarr[0, 1].imshow(images[1])
        axarr[0, 2].imshow(images[2])
        axarr[1, 0].imshow(images[3])
        axarr[1, 1].imshow(images[4])
        axarr[1, 2].imshow(images[5])
        axarr[2, 0].imshow(images[6])
        axarr[2, 1].imshow(images[7])
        axarr[2, 2].imshow(images[8])
        f.tight_layout()
        plt.show()
    button1 = tk.Button(root, text='3 DEGREE POLYNOMIAL SVM', command=POLY_SVM, fg='red', highlightbackground='red')
    canvas1.create_window(330, 300, window=button1)

    def imagesvm():  # VISUALIZE IMAGES TO UNDERSTAND
        img = Image.open(path_how_work + "/svm.png")
        img2 = mpimg.imread(path_how_work + "/svm2.png")
        f, axarr = plt.subplots(1, 2, figsize=(12, 5))
        [axi.set_axis_off() for axi in axarr]  # no axis
        f.suptitle('3 DEGREE POLYNOMIAL SVM', fontweight="bold")
        axarr[0].imshow(img, aspect='auto')
        axarr[1].imshow(img2, aspect='auto')
        plt.show()
    button9 = tk.Button(root, text='SVM?', command=imagesvm, fg='red', highlightbackground='red')
    canvas1.create_window(100, 300, window=button9)

    # FEATURE USED FOR THE MODELS
    def FEATURE_EXTRACTION():  # show the feature used in a df
        root1 = tk.Tk()
        root1.title('FEATURE USED')
        root1.geometry("1200x600")
        useful = ['New Target Name', 'Path', 'r_var_circle', 'b_var_circle', 'g_var_circle', 'HOG_std', 'CircleArea', 'SIFT_mean'] # only numeric, not list
        df = df_AllGood.loc[:, [col for col in useful]]
        frame = tk.Frame(root1)
        frame.pack(fill='both', expand=True, side=TOP)
        pt = Table(frame, dataframe=df)
        pt.show()
    button1 = tk.Button(root, text='FEATURE USED', command=FEATURE_EXTRACTION, fg='red', highlightbackground='red')
    canvas1.create_window(330, 350, window=button1)

    def barplot():
        colors2 = ['aqua', 'lightpink', 'mediumseagreen', 'coral', 'khaki']
        model_accuracy.plot.bar(x='Model', y='Accuracy', color=colors2, edgecolor='black', figsize=(8, 5), legend=None,)
        plt.title('THE MODELS ACCURACY', fontsize=20, fontweight="bold")
        plt.xticks(rotation=10, horizontalalignment="center")
        plt.xlabel("MODEL")
        plt.ylabel("ACCURACY (%)")
        for i in range(len(model_accuracy['Model'])):
            plt.text(i, model_accuracy['Accuracy'][i], model_accuracy['Accuracy'][i].round(), ha='center', Bbox=dict(facecolor='red', alpha=.8))
        plt.show()
    button1 = tk.Button(root, text='THE MODELS ACCURACY', command=barplot, fg='red', highlightbackground='red')
    canvas1.create_window(350, 400, window=button1)




# MODELS TRANSPARENT ONLY
def TRANSPARENT():
    root = tk.Tk()
    root.title('Models Prediction for Transparent Flipoff')
    canvas1 = tk.Canvas(root, width=700, height=600)
    canvas1.configure(bg='RoyalBlue4')
    canvas1.pack()

    buttonn = tk.Button(root, text='BACK', command=root.destroy, fg='red', highlightbackground='red')
    canvas1.create_window(350, 550, window=buttonn)

    def pieplot():
        names = ['Black-Gold-250ml-1000ml', 'Red-Gold', 'Red-Silver', 'Black-Silver']
        colors = ['orange', 'lightpink', 'yellow', 'aqua', 'blue']
        count_list_targ = df_AllGood_transparent.groupby('New Target').count()   # count how many images for each category
        my_dict = {'NAME': names, 'Count': count_list_targ['Path']}
        df = pd.DataFrame(data=my_dict)
        df.plot.pie(title="THE DATASET\n Transparent Flipoff", y='Count', figsize=(8, 5), labeldistance=1.1, shadow=True, colors=colors, labels=names, legend=None, startangle=90, autopct='%1.1f%%', explode=(0.1, 0.1, 0.1, 0.1)).get_figure()
        plt.show()
    button1 = tk.Button(root, text='DATASET of Transparent Flipoff', command=pieplot, fg='red', highlightbackground='red')
    canvas1.create_window(350, 20, window=button1)

    # PLOT IMPORTANCES OF THE FEATURE USING LINEAR REGRESSION
    def importanceslinreg():
        plt.subplots(figsize=(12, 7))
        plt.bar(x=importances_linreg_tot_transparent['Attribute'], height=importances_linreg_tot_transparent['Importance_linreg'], color='#087E8B')
        plt.title('Feature importance obtained from linear regression coefficients (abs)\n (no list feature)', size=20, fontweight="bold")
        plt.xticks(rotation=30)
        plt.ylabel('Importance')
        plt.xlabel('Feature')
        plt.show()
    button9 = tk.Button(root, text='FEATURE IMPORTANCE lin_reg', command=importanceslinreg, fg='red', highlightbackground='red')
    canvas1.create_window(150, 90, window=button9)

    # PLOT IMPORTANCES OF THE FEATURE USING TREE-LIKE STRUCTURE
    def importancestree():
        plt.subplots(figsize=(12, 7))
        sns.barplot(x='Attribute', y='Importance_tree', data=importances_linreg_tot_transparent, color='#087E8B',
                    order=importances_linreg_tot_transparent.sort_values('Importance_tree', ascending=False).Attribute)
        plt.title('Feature importance obtained using XGBClassifier (no list feature)\nExtreme Gradient Boosting Classifier, a tree-based model', size=20, fontweight="bold")
        plt.ylabel('Importance')
        plt.xlabel('Feature')
        plt.xticks(rotation=30)
        #plt.text(1.6, 1.4, 'The larger the coefficient is (both positive and negative),\nthe more influence it has on a prediction.', fontsize=15)
        plt.show()
    button9 = tk.Button(root, text='FEATURE IMPORTANCE XGBClassifier', command=importancestree, fg='red', highlightbackground='red')
    canvas1.create_window(550, 90, window=button9)

    # RANDOM FOREST
    def RANDOM_FOREST():
        folder_RANDOM_FOREST = path_test_models_tras + '/rf_pred'
        files = os.listdir(folder_RANDOM_FOREST)  # list of images names.png in the directory
        images = []  # list of path, one random image for each category
        n = 0
        for n in range(9):
            d = random.choice(files)  # name of a random image.png
            img_path = folder_RANDOM_FOREST + '/' + d
            img = mpimg.imread(img_path)
            images.append(img)
            n += 1
        f, axarr = plt.subplots(3, 3, figsize=(12, 7))  # plot 9 random images, one for each type
        [axi.set_axis_off() for axi in axarr.ravel()]  # no axis
        f.suptitle('RANDOM FOREST', fontweight="bold")
        axarr[0, 0].imshow(images[0])
        axarr[0, 1].imshow(images[1])
        axarr[0, 2].imshow(images[2])
        axarr[1, 0].imshow(images[3])
        axarr[1, 1].imshow(images[4])
        axarr[1, 2].imshow(images[5])
        axarr[2, 0].imshow(images[6])
        axarr[2, 1].imshow(images[7])
        axarr[2, 2].imshow(images[8])
        f.tight_layout()
        plt.show()
    button1 = tk.Button(root, text='RANDOM FOREST', command=RANDOM_FOREST, fg='red', highlightbackground='red')
    canvas1.create_window(330, 150, window=button1)

    def visualize_RF():
        img = Image.open(path_test_models_tras + "/rf_5trees_tras.png")
        plt.figure(figsize=(165, 85))
        plt.imshow(img, aspect='auto')
        plt.title('RANDOM FOREST')
        plt.axis(False)
        plt.show()
    button = tk.Button(root, text='visualize RF', command=visualize_RF, fg='red', highlightbackground='red')
    canvas1.create_window(550, 150, window=button)

    def imageRF():  # VISUALIZE IMAGES TO UNDERSTAND
        img = Image.open(path_how_work + "/RF.png")
        img2 = mpimg.imread(path_how_work + "/rf2.png")
        f, axarr = plt.subplots(1, 2, figsize=(12, 5))
        [axi.set_axis_off() for axi in axarr]  # no axis
        f.suptitle('RANDOM FOREST', fontweight="bold")
        axarr[0].imshow(img, aspect='auto')
        axarr[1].imshow(img2, aspect='auto')
        plt.show()
    button9 = tk.Button(root, text='RANDOM FOREST?', command=imageRF, fg='red', highlightbackground='red')
    canvas1.create_window(100, 150, window=button9)

    # DECISION TREE
    def DECISION_TREE():
        folder_DECISION_TREE = path_test_models_tras + '/DT_pred'
        files = os.listdir(folder_DECISION_TREE)  # list of images names.png in the directory
        images = []  # list of path, one random image for each category
        n = 0
        for n in range(9):
            d = random.choice(files)  # name of a random image.png
            img_path = folder_DECISION_TREE + '/' + d
            img = mpimg.imread(img_path)
            images.append(img)
            n += 1
        f, axarr = plt.subplots(3, 3, figsize=(12, 7))  # plot 9 random images, one for each type
        [axi.set_axis_off() for axi in axarr.ravel()]  # no axis
        f.suptitle('DECISION TREE', fontweight="bold")
        axarr[0, 0].imshow(images[0])
        axarr[0, 1].imshow(images[1])
        axarr[0, 2].imshow(images[2])
        axarr[1, 0].imshow(images[3])
        axarr[1, 1].imshow(images[4])
        axarr[1, 2].imshow(images[5])
        axarr[2, 0].imshow(images[6])
        axarr[2, 1].imshow(images[7])
        axarr[2, 2].imshow(images[8])
        f.tight_layout()
        plt.show()
    button1 = tk.Button(root, text='DECISION TREE', command=DECISION_TREE, fg='red', highlightbackground='red')
    canvas1.create_window(330, 200, window=button1)

    def visualize_DT():
        img = Image.open(path_test_models_tras + "/tree_trasparent.png")
        plt.figure(figsize=(165, 85))
        plt.imshow(img, aspect='auto')
        plt.title('DECISION TREE')
        plt.axis(False)
        plt.show()
    button = tk.Button(root, text='visualize DT', command=visualize_DT, fg='red', highlightbackground='red')
    canvas1.create_window(550, 200, window=button)

    def imagedt():  # VISUALIZE IMAGES TO UNDERSTAND
        img = Image.open(path_how_work + "/dt.png")
        img2 = mpimg.imread(path_how_work + "/dt2.png")
        f, axarr = plt.subplots(1, 2, figsize=(12, 5))
        [axi.set_axis_off() for axi in axarr]  # no axis
        f.suptitle('DECISION TREE', fontweight="bold")
        axarr[0].imshow(img, aspect='auto')
        axarr[1].imshow(img2, aspect='auto')
        plt.show()
    button9 = tk.Button(root, text='DECISION TREE?', command=imagedt, fg='red', highlightbackground='red')
    canvas1.create_window(100, 200, window=button9)

    # KNN
    def KNN():
        folder_KNN = path_test_models_tras + '/knn_pred'
        files = os.listdir(folder_KNN)  # list of images names.png in the directory
        images = []  # list of path, one random image for each category
        n = 0
        for n in range(9):
            d = random.choice(files)  # name of a random image.png
            img_path = folder_KNN + '/' + d
            img = mpimg.imread(img_path)
            images.append(img)
            n += 1
        f, axarr = plt.subplots(3, 3, figsize=(12, 7))  # plot 9 random images, one for each type
        [axi.set_axis_off() for axi in axarr.ravel()]  # no axis
        f.suptitle('K-NEAREST NEIGHBORS', fontweight="bold")
        axarr[0, 0].imshow(images[0])
        axarr[0, 1].imshow(images[1])
        axarr[0, 2].imshow(images[2])
        axarr[1, 0].imshow(images[3])
        axarr[1, 1].imshow(images[4])
        axarr[1, 2].imshow(images[5])
        axarr[2, 0].imshow(images[6])
        axarr[2, 1].imshow(images[7])
        axarr[2, 2].imshow(images[8])
        f.tight_layout()
        plt.show()
    button1 = tk.Button(root, text='K-NEAREST NEIGHBORS', command=KNN, fg='red', highlightbackground='red')
    canvas1.create_window(330, 250, window=button1)

    def confusionmatrix_knn(): # plot confusion matrix for knn
        img = Image.open(path_test_models_tras + "/knn_confusion_matrix_tras.png")
        plt.figure(figsize=(80, 100))
        plt.imshow(img, aspect='auto')
        plt.axis(False)
        plt.show()
    button = tk.Button(root, text='confusion matrix', command=confusionmatrix_knn, fg='red', highlightbackground='red')
    canvas1.create_window(540, 250, window=button)

    def eda():   # EDA knn
        img = Image.open(path_test_models_tras + "/eda_tras.png")
        plt.figure(figsize=(80, 100))
        plt.imshow(img, aspect='auto')
        plt.axis(False)
        plt.show()
    button = tk.Button(root, text='EDA', command=eda, fg='red', highlightbackground='red')
    canvas1.create_window(650, 250, window=button)



    def imageknn():  # VISUALIZE IMAGES TO UNDERSTAND
        img = Image.open(path_how_work + "/knn.png")
        img2 = mpimg.imread(path_how_work + "/knn2.png")
        f, axarr = plt.subplots(1, 2, figsize=(12, 5))
        [axi.set_axis_off() for axi in axarr]  # no axis
        f.suptitle('K-NEAREST NEIGHBORS', fontweight="bold")
        axarr[0].imshow(img, aspect='auto')
        axarr[1].imshow(img2, aspect='auto')
        plt.show()
    button9 = tk.Button(root, text='KNN?', command=imageknn, fg='red', highlightbackground='red')
    canvas1.create_window(100, 250, window=button9)


    # 3 DEGREE POLYNOMIAL SVM
    def POLY_SVM():
        folder_POLY_SVM = path_test_models_tras + '/poly_pred'
        files = os.listdir(folder_POLY_SVM)  # list of images names.png in the directory
        images = []  # list of path, one random image for each category
        n = 0
        for n in range(9):
            d = random.choice(files)  # name of a random image.png
            img_path = folder_POLY_SVM + '/' + d
            img = mpimg.imread(img_path)
            images.append(img)
            n += 1
        f, axarr = plt.subplots(3, 3, figsize=(12, 7))  # plot 9 random images, one for each type
        [axi.set_axis_off() for axi in axarr.ravel()]  # no axis
        f.suptitle('3 DEGREE POLYNOMIAL SVM', fontweight="bold")
        axarr[0, 0].imshow(images[0])
        axarr[0, 1].imshow(images[1])
        axarr[0, 2].imshow(images[2])
        axarr[1, 0].imshow(images[3])
        axarr[1, 1].imshow(images[4])
        axarr[1, 2].imshow(images[5])
        axarr[2, 0].imshow(images[6])
        axarr[2, 1].imshow(images[7])
        axarr[2, 2].imshow(images[8])
        f.tight_layout()
        plt.show()
    button1 = tk.Button(root, text='3 DEGREE POLYNOMIAL SVM', command=POLY_SVM, fg='red', highlightbackground='red')
    canvas1.create_window(330, 300, window=button1)

    def imagesvm():  # VISUALIZE IMAGES TO UNDERSTAND
        img = Image.open(path_how_work + "/svm.png")
        img2 = mpimg.imread(path_how_work + "/svm2.png")
        f, axarr = plt.subplots(1, 2, figsize=(12, 5))
        [axi.set_axis_off() for axi in axarr]  # no axis
        f.suptitle('3 DEGREE POLYNOMIAL SVM', fontweight="bold")
        axarr[0].imshow(img, aspect='auto')
        axarr[1].imshow(img2, aspect='auto')
        plt.show()
    button9 = tk.Button(root, text='SVM?', command=imagesvm, fg='red', highlightbackground='red')
    canvas1.create_window(100, 300, window=button9)

    # FEATURE USED FOR THE MODELS
    def FEATURE_EXTRACTION():  # show the feature used in a df
        root1 = tk.Tk()
        root1.title('FEATURE USED')
        root1.geometry("1200x600")
        useful = ['New Target', 'Path', 'r_mean_circle', 'g_mean_circle', 'b_std_circle', 'SIFT_std', 'HOG_min', 'r_var_circle'] # only numeric, not list
        df = df_AllGood_transparent.loc[:, [col for col in useful]]
        frame = tk.Frame(root1)
        frame.pack(fill='both', expand=True, side=TOP)
        pt = Table(frame, dataframe=df)
        pt.show()
    button1 = tk.Button(root, text='FEATURE USED', command=FEATURE_EXTRACTION, fg='red', highlightbackground='red')
    canvas1.create_window(330, 350, window=button1)

    def barplot():
        colors2 = ['aqua', 'lightpink', 'mediumseagreen', 'coral', 'khaki']
        model_accuracy_transparent.plot.bar(x='Model', y='Accuracy', color=colors2, edgecolor='black', figsize=(8, 5), legend=None,)
        plt.title('THE MODELS ACCURACY', fontsize=20, fontweight="bold")
        plt.xticks(rotation=10, horizontalalignment="center")
        plt.xlabel("MODEL")
        plt.ylabel("ACCURACY (%)")
        for i in range(len(model_accuracy_transparent['Model'])):
            plt.text(i, model_accuracy_transparent['Accuracy'][i], model_accuracy_transparent['Accuracy'][i].round(), ha='center', Bbox=dict(facecolor='red', alpha=.8))
        plt.show()
    button1 = tk.Button(root, text='THE MODELS ACCURACY', command=barplot, fg='red', highlightbackground='red')
    canvas1.create_window(350, 400, window=button1)


# CONVOLUTIONAL NEURAL NETWORK
def NN():
    root = tk.Tk()
    root.title('NEURAL NETWORK')
    canvas1 = tk.Canvas(root, width=1000, height=750)
    canvas1.pack()
    canvas1.configure(bg='RoyalBlue4')
    entry = tk.Entry(root)
    canvas1.create_window(500, 150, window=entry)
    buttonn = tk.Button(root, text='BACK', command=root.destroy, fg='red', highlightbackground='red')
    canvas1.create_window(500, 600, window=buttonn)
    pathNN = path_folder + "/images_notonlygood/NN/"

    def image():  # VISUALIZE IMAGES TO UNDERSTAND HOW A CNN WORKS
        img0 = Image.open(pathNN + "bird.png")
        img1 = Image.open(pathNN + "rr.png")
        fig, ax = plt.subplots(2, figsize=(12, 7))
        [axi.set_axis_off() for axi in ax.ravel()]  # no axis
        ax[0].imshow(img0)
        ax[1].imshow(img1)
        plt.show()
    button9 = tk.Button(root, text='How does it work?', command=image, fg='red', highlightbackground='red')
    canvas1.create_window(500, 400, window=button9)


    def neurals():  # CNN
        # DATASET MANIPULATION:
        # Create new data
        imag = []
        for path in df_AllGood['Path']:
            name_list = path.split('/')
            imag.append(name_list[-1])
        tg_list = df_AllGood['Target'].to_list()
        images = {'Image': imag, 'Target': tg_list}   # rename the attributes
        img_data = pd.DataFrame(images)  # dataset with image name and gt
        indice_da_drop = img_data.index[img_data['Image'] == 'Product_Prod0091_Cam00_Pre00_Img000_ResM001_ResImg000.png']  # droppo questa immagine perche fuori posto
        img_data = img_data.drop(indice_da_drop[0], axis=0)
        print(indice_da_drop)
        IMAGE_WIDTH = 250   # image shape
        IMAGE_HEIGHT = 250

        for img, x, y, r in zip(img_data['Image'], df_AllGood['x_circle'], df_AllGood['y_circle'], df_AllGood['r_circle']):    # images have different sizes, so I decided to resize them
            imagePath = path_folder + "/images_notonlygood/all_images" + '/' + img
            image = cv.imread(imagePath, cv.IMREAD_COLOR)
            # MASK WHITE FILLED CIRCLE
            r2 = np.rint(r * 4 / 5)    # take the 80% of the radius in order to consider only the internal part of the cap
            mask = np.zeros(image.shape[:2], dtype="uint8")
            cv.circle(mask, (x, y), int(r2), (255, 255, 255), -1)   # -1 significa che mi colora l'interno del cerchio di bianco
            masked = cv.bitwise_and(image, image, mask=mask)
            crop = masked[y-int(r2):y+int(r2), x-int(r2):x+int(r2)]
            res = cv.resize(crop, (IMAGE_WIDTH, IMAGE_HEIGHT))
            cv.imwrite(image_dir_NN + '/' + img, res)  # save all the images masked ad cropped in the folder allimagesNN

        # Divide the dataset
        # Delate output directory
        if os.path.exists(pathNN+'train/'):  # for training set
            shutil.rmtree(pathNN+'train/')   # rmtree is a function to delete all the contents of a directory
        if os.path.exists(pathNN+'val/'):    # for validation set
            shutil.rmtree(pathNN+'val/')

        # Create training set, test set and directory for validation set
        if not os.path.exists(pathNN+'train/'):  # for training set
            os.mkdir(pathNN+'train/')            # mkdir creates a directory named train
        if not os.path.exists(pathNN+'val/'):    # for validation set
            os.mkdir(pathNN+'val/')

        # Create a sub-directory based on the Target 0, 1, 2, ..., 8
        for class_ in img_data['Target'].unique():
            if not os.path.exists(pathNN+'train/'+str(class_)+'/'):  # for training set
                os.mkdir(pathNN+'train/'+str(class_)+'/')
            if not os.path.exists(pathNN+'val/'+str(class_)+'/'):    # for validation set
                os.mkdir(pathNN+'val/'+str(class_)+'/')

        # Split training set and validation set, 70% for training set, 30% for validation set
        X_train, X_val, y_train, y_val = train_test_split(img_data, img_data['Target'], test_size=0.30, stratify=img_data['Target'], random_state=2023)


        # Save the divided data sets into folders according to the first attribute
        for image, type_ in zip(image_dir_NN + '/' + X_train['Image'], y_train):  # for training set
            copy2(image, pathNN+'train/' + str(type_))   # salva le immagini dalla cartella generale, nelle cartelle val e train

        for image, type_ in zip(image_dir_NN + '/' + X_val['Image'], y_val):       # for validation set
            copy2(image, pathNN+'val/' + str(type_))

        # DATA AUGMENTATION
        datagen = ImageDataGenerator(zoom_range=0.2, rotation_range=10, fill_mode='nearest')   # Picture Generator Initialization and augmented images (rotated, zoomed)
        train = datagen.flow_from_directory(pathNN+'train/', class_mode='categorical', color_mode="rgb", shuffle=True, seed=2023)  # flow_from_directory means that is able to take the images from different subdirectories
        val = datagen.flow_from_directory(pathNN+'val/', target_size=(IMAGE_WIDTH, IMAGE_HEIGHT), class_mode='categorical', color_mode="rgb", shuffle=False, seed=2023)  # if True prediction mixed and no match with gt

        # the CNN
        def build(): # Build a classification model to divide the 18 classes. Use 3 convolutional layers, 2D input maximum pooling layer and dropout method (Dropout)
            model = Sequential()   # Sequential model
            IMAGE_CHANNELS = 3     # Three Channels
            model.add(Lambda(lambda x: x, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)))  # Call the new layer.

            # FIRST CONVOLUTION
            model.add(Conv2D(32, (3, 3), activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(3, 3)))
            model.add(Dropout(0.3))

            # SECOND CONVOLUTION: Convert 32 images to 64 images
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(3, 3)))
            model.add(Dropout(0.4))

            # THIRD CONVOLUTION: Turn 64 images into 128 images
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(3, 3)))
            model.add(Dropout(0.4))

            model.add(Flatten())

            # Fully connected layer (classified according to the combination of features, greatly reducing the impact of feature positions on classification), a total of 512 neurons
            model.add(Dense(512, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))

            # Output layer, 9 neurons (corresponding to 9 first attributes) are used to distinguish 9 classes
            model.add(Dense(9, activation='softmax'))
            model.summary()  # View model summary
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[tf.keras.metrics.CategoricalAccuracy()])
            return model

        model = build()  # Start training


        def grafici():  # SINGLE FUNCTION THAT CONTAINS ALL THE FUNCTION FOR THE CNN ACCURACY ANALYSIS (only if the user inserts the right number in the box)
            root1 = tk.Tk()
            root1.title('ACCURACY ANALYSIS')
            canvas5 = tk.Canvas(root1, width=1000, height=900)
            canvas5.pack()
            canvas5.configure(bg='RoyalBlue4')

            def graph():    # View the accuracy of the training set and validation set
                plt.style.use('ggplot')
                acc = history.history['categorical_accuracy']
                val_acc = history.history['val_categorical_accuracy']
                epochs = range(len(acc))

                plt.figure(figsize=(6, 5))
                plt.plot(epochs, acc, 'r', label='training_categorical_accuracy')
                plt.plot(epochs, val_acc, 'b', label='validation_categorical_accuracy')
                plt.title('Training and Validation Accuracy')
                plt.xlabel('-----epochs--->')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.show()

            def graph2():   # View the loss of the training set and validation set
                acc = history.history['categorical_accuracy']
                loss = history.history['loss']
                val_loss = history.history['val_loss']
                epochs = range(len(acc))
                plt.figure(figsize=(6, 5))
                plt.plot(epochs, loss, 'r', label='training_loss')
                plt.plot(epochs, val_loss, 'b', label='validation_loss')
                plt.title('Training and Validation Loss')
                plt.xlabel('----epochs--->')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()

            button5 = tk.Button(root1, text='Training and Validation Accuracy', command=graph, fg='red', highlightbackground='red')
            canvas5.create_window(500, 200, window=button5)
            button6 = tk.Button(root1, text='Training and Validation Loss', command=graph2, fg='red', highlightbackground='red')
            canvas5.create_window(500, 300, window=button6)

            def time():
                acc = history.history['categorical_accuracy']
                epochs = len(acc)
                tim = round(stop-start, 1)
                lablee = tk.Label(root1, text='The time needed for the CNN training with an accuracy of ' + str(round(entrata * 100)) + '% is:\n'+ str(tim) + ' sec. with ' + str(epochs) + ' epochs.')
                canvas5.create_window(500, 400, window=(lablee))
            button99 = tk.Button(root1, text='TIME', command=time, fg='red', highlightbackground='red')
            canvas5.create_window(500, 350, window=button99)

            def image_pred_folders():             # PLOT THE IMAGES WITH PREDICTION THAT ARE SAVED IN A FOLDER AND SHOW PLOT
                images = []  # list of path, one random image for each category
                for n in range(9):
                    path = pathNN + 'predictionNN'
                    files = os.listdir(path)  # list of images names.png in the directory
                    d = random.choice(files)  # name of a random image.png
                    img_path = path + '/' + d
                    img = mpimg.imread(img_path)
                    images.append(img)
                    n += 1
                f, axarr = plt.subplots(3, 3, figsize=(12, 7))  # plot 9 random images
                [axi.set_axis_off() for axi in axarr.ravel()]  # no axis
                f.suptitle('CNN PREDICTIONS (validation set)', fontweight="bold")
                axarr[0, 0].imshow(images[0])
                axarr[0, 1].imshow(images[1])
                axarr[0, 2].imshow(images[2])
                axarr[1, 0].imshow(images[3])
                axarr[1, 1].imshow(images[4])
                axarr[1, 2].imshow(images[5])
                axarr[2, 0].imshow(images[6])
                axarr[2, 1].imshow(images[7])
                axarr[2, 2].imshow(images[8])
                f.tight_layout()
                plt.show()
            button8 = tk.Button(root1, text='CNN model prediction', command=image_pred_folders, fg='red', highlightbackground='red')
            canvas5.create_window(500, 500, window=button8)

            def GTvsPred():
                root1 = tk.Tk()
                root1.title('GT vs PREDICTION for the validation set')
                root1.geometry("1200x600")
                df = predict_frame.iloc[:, 2:]  # tolgo new feature
                frame = tk.Frame(root1)
                frame.pack(fill='both', expand=True, side=TOP)
                pt = Table(frame, dataframe=df)
                pt.show()

            button9 = tk.Button(root1, text='GT vs CNN PREDICTIONS', command=GTvsPred, fg='red', highlightbackground='red')
            canvas5.create_window(500, 600, window=button9)


            butto = tk.Button(root1, text='BACK', command=root1.destroy, fg='red', highlightbackground='red')
            canvas5.create_window(500, 700, window=butto)

        try:    # CNN works only if the user inserts the right number (for ex. 0.1 for a 10% accuracy) in the entry box
            entrata = float(entry.get())
            if entrata > 1.0 or entrata < 0.0:  # error message if the user inserts something wrong in the entry box
                tk.messagebox.showerror(title="ERROR", message="The number insert is NOT between 0 and 1. \n Please insert a float such as 0.8 for a 80% accuracy.")
            else:
                class myCallback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs={}):
                        if(logs.get('categorical_accuracy') > entrata) and (logs.get('val_categorical_accuracy') > entrata):   # SET HERE THE ACCURACY
                            self.model.stop_training = True
                            grafici()
        except ValueError:  # error message if the user inserts something wrong in the entry box
            tk.messagebox.showerror(title="ERROR", message="Wrong digit insert! \n You should insert a float like: 0.1 or 0.34")

        # Creating a callback to stop the training when a particular accuarcy is reached
        callbacks = myCallback()

        start = time()
        history = model.fit(train, validation_data=val, batch_size=20, epochs=100,     # CNN doesn'do all the 100 epoch, but it finishes when the desired accuracy is reached.
            callbacks=[tf.keras.callbacks.ReduceLROnPlateau(), callbacks])
        stop = time()
        predict = model.predict(val)      # prob di appartrnenza alla classi
        print(stop - start)
        predict_frame = pd.DataFrame([])  # Check forecast accuracy
        predict_frame['indexes'] = val.index_array
        predict_frame['column_split'] = val.filenames  # for ex. 0/name_image.png
        predict_frame['category'] = np.argmax(predict, axis=-1)
        #predict_frame = predict_frame.sort_values(by='indexes', ascending=True)
        labels = ['Trasparente-Nero-Oro-250ml',  # target = 0  shape 1024, 1280
                          'Trasparente-Nero-Oro-1000ml',   # target = 1  shape 1024, 1280
                          'Trasparente-Rosso-Oro',  # target = 2   shape 1024, 1280
                          'Trasparente-Rosso-Argento',  # target = 3  shape 1024, 1280
                          'Trasparente-Nero-Argento',   # target = 4  shape 1024, 1280
                          'Alluminio-Anello-Alluminio',    # target = 5   shape 1024, 1280
                          'FlipOff Arancio',  # target = 6  shape 1024, 1280
                          'Flipoff Blu', # target = 7  shape 544, 707
                          'FlipOff Arancio-scritta']  # target = 8
        predict_frame[['GroundTruth', 'file_name']] = predict_frame['column_split'].str.split('/', 1, expand=True)
        path_val = pathNN +'val/'
        predict_frame.to_csv(pathNN+'predict_frame.csv', index=False)
        import glob
        for cat in range(9):  # save images in folder predictionNN with the NN prediction and gt
            for img_path, gt, img_name, pred in zip(predict_frame['column_split'], predict_frame['GroundTruth'], predict_frame['file_name'], predict_frame['category']):
                image = cv.imread(path_val+img_path, cv.IMREAD_COLOR)
                image = cv.resize(image, (800, 600))  # resize for better visualization in plot
                cv.putText(image, 'Prediction NN:', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                cv.putText(image, str(labels[pred]), (10, 100), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                cv.putText(image, 'Ground Truth:', (10, 150), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv.putText(image, str(labels[int(gt)]), (10, 200), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                image_path = pathNN + '/predictionNN/' + img_name
                cv.imwrite(image_path, image)


    button1 = tk.Button(root, text='Insert the accuracy in the box above \n (0.1 for 10%)\nand PRESS this button', command=neurals, fg='red', highlightbackground='red')
    canvas1.create_window(500, 200, window=button1)


# VISION TRANSFORMERS
def TRANS():
    root = tk.Tk()
    root.title('VISION TRANSFORMER')
    canvas1 = tk.Canvas(root, width=1000, height=750)
    canvas1.configure(bg='RoyalBlue4')
    canvas1.pack()
    entry = tk.Entry(root)
    canvas1.create_window(500, 150, window=entry)
    buttonn = tk.Button(root, text='BACK', command=root.destroy, fg='red', highlightbackground='red')
    canvas1.create_window(500, 600, window=buttonn)
    image_dir_tran = path_folder + "/images_notonlygood/allimagesTrans"
    path_Transformers = path_folder + "/images_notonlygood/Transformers/"

    # RIFARE SPIEGAZIONE VISION TRANS
    def image():  # VISUALIZE IMAGES TO UNDERSTAND HOW A transformers WORKS
        img1 = mpimg.imread(path_Transformers + "trans.png")
        img2 = mpimg.imread(path_Transformers + "trans_ex.png")
        f, axarr = plt.subplots(1, 2, figsize=(12, 12))
        [axi.set_axis_off() for axi in axarr]  # no axis
        f.suptitle('VISION TRANSFORMER', fontweight="bold")
        axarr[0].imshow(img1, aspect='auto')
        axarr[1].imshow(img2, aspect='auto')
        plt.show()
    button9 = tk.Button(root, text='How does it work?', command=image, fg='red', highlightbackground='red')
    canvas1.create_window(500, 400, window=button9)

    def transformer():
        # DATASET MANIPULATION:
        # Create new data
        imag = []
        for path in df_AllGood['Path']:
            name_list = path.split('/')
            imag.append(name_list[-1])

        tg_list = df_AllGood['Target'].to_list()
        images = {'Image': imag, 'Target': tg_list}   # rename the attributes
        img_data = pd.DataFrame(images)  # dataset with image name and gt
        indice_da_drop = img_data.index[img_data['Image'] == 'Product_Prod0091_Cam00_Pre00_Img000_ResM001_ResImg000.png']  # droppo questa immagine perche fuori posto
        img_data = img_data.drop(indice_da_drop[0], axis=0)
        print(indice_da_drop)

        IMAGE_WIDTH = 224   # image shape 224
        IMAGE_HEIGHT = 224

        for img, x, y, r in zip(img_data['Image'], df_AllGood['x_circle'], df_AllGood['y_circle'], df_AllGood['r_circle']):    # images have different sizes, so I decided to resize them
            imagePath = path_folder + "/images_notonlygood/all_images" + '/' + img
            image = cv.imread(imagePath, cv.IMREAD_COLOR)
            # MASK WHITE FILLED CIRCLE
            r2 = np.rint(r * 4 / 5)    # take the 80% of the radius in order to consider only the internal part of the cap
            mask = np.zeros(image.shape[:2], dtype="uint8")
            cv.circle(mask, (x, y), int(r2), (255, 255, 255), -1)   # -1 significa che mi colora l'interno del cerchio di bianco
            masked = cv.bitwise_and(image, image, mask=mask)
            crop = masked[y-int(r2):y+int(r2), x-int(r2):x+int(r2)]
            res = cv.resize(crop, (IMAGE_WIDTH, IMAGE_HEIGHT))
            cv.imwrite(image_dir_tran + '/' + img, res)   # save all the images masked ad cropped in the folder allimagesTrans

        # Divide the dataset
        # Delate output directory
        if os.path.exists(path_Transformers+'train/'):  # for training set
            shutil.rmtree(path_Transformers+'train/')   # rmtree is a function to delete all the contents of a directory
        if os.path.exists(path_Transformers+'val/'):    # for validation set
            shutil.rmtree(path_Transformers+'val/')

        # Create training set, test set and directory for validation set
        if not os.path.exists(path_Transformers+'train/'):  # for training set
            os.mkdir(path_Transformers+'train/')            # mkdir creates a directory named train
        if not os.path.exists(path_Transformers+'val/'):    # for validation set
            os.mkdir(path_Transformers+'val/')

        # Create a sub-directory based on the Target 0, 1, 2, ..., 8 -> in this way the proportions of every class in the training and val set are respected
        for class_ in img_data['Target'].unique():
            if not os.path.exists(path_Transformers+'train/'+str(class_)+'/'):  # for training set
                os.mkdir(path_Transformers+'train/'+str(class_)+'/')
            if not os.path.exists(path_Transformers+'val/'+str(class_)+'/'):    # for validation set
                os.mkdir(path_Transformers+'val/'+str(class_)+'/')

        # Split training set and validation set, 70% for training set, 30% for validation set
        X_train, X_val, y_train, y_val = train_test_split(img_data, img_data['Target'], test_size=0.30, stratify=img_data['Target'], random_state=2023)
        # X_train 252 images and X_val 109 are a df with columns: Image for the name.png and Target
        # y_train and y_val solo target

        imgs_array_train = []  # Convert single image to a batch. # collection of images shape (n of images, 224, 224, 3)
        for p in X_train['Image']:
            path = image_dir_tran + '/' + p
            img = tf.keras.preprocessing.image.load_img(path)
            input_arr = tf.keras.preprocessing.image.img_to_array(img)
            input_arr = np.array([input_arr])
            imgs_array_train.append(input_arr)  # descrive le immagini come array per essere leggibile da keras
        XXtrain = np.vstack(imgs_array_train) # shape: (252, 224, 224 ,3)


        imgs_array_X_val = []  # Convert single image to a batch. # collection of images shape (n of images, 224, 224, 3)
        for p in X_val['Image']:
            path = image_dir_tran + '/' + p
            img = tf.keras.preprocessing.image.load_img(path)
            input_arr = tf.keras.preprocessing.image.img_to_array(img)
            input_arr = np.array([input_arr])
            imgs_array_X_val.append(input_arr)  # descrive le immagini come array per essere leggibile da keras
        XXval = np.vstack(imgs_array_X_val) # shape: (252, 224, 224 ,3)

        # Save the divided data sets into folders according to the class
        for image, type_ in zip(image_dir_tran + '/' + X_train['Image'], y_train):  # for training set
            copy2(image, path_Transformers+'train/' + str(type_))   # salva le immagini dalla cartella generale, nelle cartelle val e train

        for image, type_ in zip(image_dir_tran + '/' + X_val['Image'], y_val):       # for validation set
            copy2(image, path_Transformers+'val/' + str(type_))

        # Configure the hyperparameters
        input_shape = (224, 224, 3)
        num_classes = 9
        batch_size = 256
        patch_size = 16  # Size of the patches to be extract from the input images
        num_patches = (IMAGE_WIDTH // patch_size) ** 2  # 224/16
        projection_dim = 64
        # projection_dim : size of the hidden dimension feature vectors in the model: we have 64 dimensional vectors of the projection dimension
        # to project the patches into these 64-dimensional feature vectors that are then concatenated together
        num_heads = 4  # number of attention heads describes having n different parameterizations of the query key and value matrices so we have four separate transformations that then aggregate the outputs of these four separate self-attention parameterizations from the previous layer
        transformer_units = [projection_dim * 2, projection_dim,]  # Size of the transformer layers
        transformer_layers = 4
        mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier

        # DATA AUGMENTATION
        data_augmentation = keras.Sequential([
            layers.Normalization(),
            layers.Resizing(IMAGE_WIDTH, IMAGE_HEIGHT),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.02),
            layers.RandomZoom(height_factor=0.2, width_factor=0.2), ], name="data_augmentation",)
        # Compute the mean and the variance of the training data for normalization.
        data_augmentation.layers[0].adapt(XXtrain)

        # Implement multilayer perceptron (MLP)
        def mlp(x, hidden_units, dropout_rate):
            for units in hidden_units:
                x = layers.Dense(units, activation=tf.nn.gelu)(x)
                x = layers.Dropout(dropout_rate)(x)
            return x

        # Implement patch creation as a layer
        class Patches(layers.Layer):
            def __init__(self, patch_size):
                super(Patches, self).__init__()
                self.patch_size = patch_size

            def call(self, images):
                batch_size = tf.shape(images)[0]
                patches = tf.image.extract_patches(
                    images=images,
                    sizes=[1, self.patch_size, self.patch_size, 1],
                    strides=[1, self.patch_size, self.patch_size, 1],
                    rates=[1, 1, 1, 1],
                    padding="VALID",)
                patch_dims = patches.shape[-1]
                patches = tf.reshape(patches, [batch_size, -1, patch_dims])
                return patches

        # Implement the patch encoding layer
        # The PatchEncoder layer will linearly transform a patch by projecting it into a vector of size projection_dim.
        # In addition, it adds a learnable position embedding to the projected vector.
        class PatchEncoder(layers.Layer):
            def __init__(self, num_patches, projection_dim):
                super(PatchEncoder, self).__init__()
                self.num_patches = num_patches
                self.projection = layers.Dense(units=projection_dim)
                self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

            def call(self, patch):
                positions = tf.range(start=0, limit=self.num_patches, delta=1)
                encoded = self.projection(patch) + self.position_embedding(positions)
                return encoded

        # Build the ViT model
        def create_vit_classifier():
            inputs = layers.Input(shape=input_shape)
            augmented = data_augmentation(inputs)  # Augment data.
            patches = Patches(patch_size)(augmented)  # Create patches.
            encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)  # Encode patches.
            # Create multiple layers of the Transformer block:
            for _ in range(transformer_layers):
                x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)  # Layer normalization 1.
                attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1) # Create a multi-head attention layer.
                x2 = layers.Add()([attention_output, encoded_patches])  # Skip connection 1.
                x3 = layers.LayerNormalization(epsilon=1e-6)(x2)  # Layer normalization 2.
                x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)  # MLP.
                encoded_patches = layers.Add()([x3, x2]) # Skip connection 2.
            representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches) # Create a [batch_size, projection_dim] tensor.
            representation = layers.Flatten()(representation)
            representation = layers.Dropout(0.5)(representation)
            features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)  # Add MLP.
            logits = layers.Dense(num_classes)(features)  # Classify outputs.
            model = keras.Model(inputs=inputs, outputs=logits)  # Create the Keras model.
            return model

        vit_classifier = create_vit_classifier()

        def grafici():
            root1 = tk.Tk()
            root1.title('ACCURACY ANALYSIS')
            canvas5 = tk.Canvas(root1, width=1000, height=900)
            canvas5.pack()
            canvas5.configure(bg='RoyalBlue4')

            def graph():    # View the accuracy of the training set and validation set
                plt.style.use('ggplot')
                acc = history.history['sparse_categorical_accuracy']
                val_acc = history.history['val_sparse_categorical_accuracy']
                epochs = range(len(acc))

                plt.figure(figsize=(6, 5))
                plt.plot(epochs, acc, 'r', label='training_sparse_categorical_accuracy')
                plt.plot(epochs, val_acc, 'b', label='validation_sparse_categorical_accuracy')
                plt.title('Training and Validation Accuracy')
                plt.xlabel('-----epochs--->')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.show()

            def graph2():   # View the loss of the training set and validation set
                acc = history.history['sparse_categorical_accuracy']
                loss = history.history['loss']
                val_loss = history.history['val_loss']
                epochs = range(len(acc))
                plt.figure(figsize=(6, 5))
                plt.plot(epochs, loss, 'r', label='training_loss')
                plt.plot(epochs, val_loss, 'b', label='validation_loss')
                plt.title('Training and Validation Loss')
                plt.xlabel('----epochs--->')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()

            button5 = tk.Button(root1, text='Training and Validation Accuracy', command=graph, fg='red', highlightbackground='red')
            canvas5.create_window(500, 200, window=button5)
            button6 = tk.Button(root1, text='Training and Validation Loss', command=graph2, fg='red', highlightbackground='red')
            canvas5.create_window(500, 300, window=button6)

            def image_pred_folders():             # PLOT THE IMAGES WITH PREDICTION THAT ARE SAVED IN A FOLDER AND SHOW PLOT
                images = []  # list of path, one random image for each category
                for n in range(9):
                    path = path_Transformers + 'predictionTRAN'
                    files = os.listdir(path)  # list of images names.png in the directory
                    d = random.choice(files)  # name of a random image.png
                    img_path = path + '/' + d
                    img = mpimg.imread(img_path)
                    images.append(img)
                    n += 1
                f, axarr = plt.subplots(3, 3, figsize=(12, 7))  # plot 9 random images
                [axi.set_axis_off() for axi in axarr.ravel()]  # no axis
                f.suptitle('VI-TRANSFORMER PREDICTIONS (validation set)', fontweight="bold")
                axarr[0, 0].imshow(images[0])
                axarr[0, 1].imshow(images[1])
                axarr[0, 2].imshow(images[2])
                axarr[1, 0].imshow(images[3])
                axarr[1, 1].imshow(images[4])
                axarr[1, 2].imshow(images[5])
                axarr[2, 0].imshow(images[6])
                axarr[2, 1].imshow(images[7])
                axarr[2, 2].imshow(images[8])
                f.tight_layout()
                plt.show()
            button8 = tk.Button(root1, text='VI-TRANSFORMER model prediction', command=image_pred_folders, fg='red', highlightbackground='red')
            canvas5.create_window(500, 500, window=button8)


            def GTvsPred():
                root1 = tk.Tk()
                root1.title('GT vs PREDICTION for the validation set')
                root1.geometry("1200x600")
                frame = tk.Frame(root1)
                frame.pack(fill='both', expand=True, side=TOP)
                pt = Table(frame, dataframe=predict_frame)
                pt.show()
            button9 = tk.Button(root1, text='GT vs PREDICTIONS', command=GTvsPred, fg='red', highlightbackground='red')
            canvas5.create_window(500, 600, window=button9)

            def time():
                acc = history.history['sparse_categorical_accuracy']
                epochs = len(acc)
                tim = round(stop_trans - start_trans, 1)
                lablee = tk.Label(root1, text='The time needed for the Transformer training with an accuracy of ' + str(round(entrata * 100)) + '% is:\n' + str(tim) + ' sec. with ' + str(epochs) + ' epochs.')
                canvas5.create_window(500, 400, window=(lablee))
            button99 = tk.Button(root1, text='TIME', command=time, fg='red', highlightbackground='red')
            canvas5.create_window(500, 350, window=button99)

            butto = tk.Button(root1, text='BACK', command=root1.destroy, fg='red', highlightbackground='red')
            canvas5.create_window(500, 700, window=butto)


        def run_experiment(model):
            #model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True, name="sparse_categorical_crossentropy"), metrics=[
                #keras.metrics.SparseCategoricalAccuracy(name="sparse_categorical_accuracy"),],)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="sparse_categorical_accuracy")])
            model.summary()
            #_, val_accuracy = model.evaluate(XXval, y_val)
            #print(f"Validation accuracy: {round(val_accuracy * 100, 2)}%")

            try:    # CNN works only if the user inserts the right number (for ex. 0.1 for a 10% accuracy) in the entry box
                entrata = float(entry.get())
                if entrata > 1.0 or entrata < 0.0:  # error message if the user inserts something wrong in the entry box
                    tk.messagebox.showerror(title="ERROR", message="The number insert is NOT between 0 and 1. \n Please insert a float such as 0.8 for a 80% accuracy.")
                else:
                    class myCallback(tf.keras.callbacks.Callback):
                        def on_epoch_end(self, epoch, logs={}):
                            if(logs.get('sparse_categorical_accuracy') > entrata) and (logs.get('val_sparse_categorical_accuracy') > entrata):   # SET HERE THE ACCURACY
                                self.model.stop_training = True
                                grafici()
            except ValueError:  # error message if the user inserts something wrong in the entry box
                tk.messagebox.showerror(title="ERROR", message="Wrong digit insert! \n You should insert a float like: 0.1 or 0.34")
            # Creating a callback to stop the training when a particular accuarcy is reached
            callbacks = myCallback()
            start_trans = time()
            history = vit_classifier.fit(x=XXtrain, y=y_train, batch_size=batch_size, epochs=150, validation_data=(XXval, y_val),     # doesn'do all the 100 epoch, but it finishes when the desired accuracy is reached.
                                         callbacks=[tf.keras.callbacks.ReduceLROnPlateau(), callbacks])
            stop_trans = time()

            predict = model.predict(XXval)      # prob di appartrnenza alla classi

            return history, start_trans, stop_trans, entrata, predict

        history, start_trans, stop_trans, entrata, predict = run_experiment(vit_classifier)

        predict_frame = pd.DataFrame([])  # Check forecast accuracy
        predict_frame['Image'] = X_val['Image']
        predict_frame['Prediction'] = np.argmax(predict, axis=-1)
        predict_frame['GroundTruth'] = y_val
        predict_frame = predict_frame.sort_values(by='GroundTruth', ascending=True)
        predict_frame.to_csv(path_Transformers + 'predict_frame.csv', index=False)
        labels = ['Trasparente-Nero-Oro-250ml',  # target = 0  shape 1024, 1280
                  'Trasparente-Nero-Oro-1000ml',   # target = 1  shape 1024, 1280
                  'Trasparente-Rosso-Oro',  # target = 2   shape 1024, 1280
                  'Trasparente-Rosso-Argento',  # target = 3  shape 1024, 1280
                  'Trasparente-Nero-Argento',   # target = 4  shape 1024, 1280
                  'Alluminio-Anello-Alluminio',    # target = 5   shape 1024, 1280
                  'FlipOff Arancio',  # target = 6  shape 1024, 1280
                  'Flipoff Blu', # target = 7  shape 544, 707
                  'FlipOff Arancio-scritta']  # target = 8
        import glob
        for cat in range(9):  # save images in folder predictionNN with the NN prediction and gt
            for gt, img_name, pred in zip(predict_frame['GroundTruth'], predict_frame['Image'], predict_frame['Prediction']):
                image = cv.imread(path_Transformers + 'val/' + str(gt) + '/' + img_name, cv.IMREAD_COLOR)
                image = cv.resize(image, (800, 600))  # resize for better visualization in plot
                cv.putText(image, 'Prediction Vi-Tran.:', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                cv.putText(image, str(labels[pred]), (10, 100), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                cv.putText(image, 'Ground Truth:', (10, 150), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv.putText(image, str(labels[int(gt)]), (10, 200), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                image_path = path_Transformers + '/predictionTRAN/' + img_name
                cv.imwrite(image_path, image)


    button11 = tk.Button(root, text='Insert the accuracy in the box above \n (0.1 for 10%)\nand PRESS this button', command=transformer, fg='red', highlightbackground='red')
    canvas1.create_window(500, 200, window=button11)



# CREATE THE GUI:
MAIN = Tk()
MAIN.title('STAGE ELISA SALVI')
MAIN.geometry('900x700')
background_image = tk.PhotoImage(file=path_folder + '/antar.png')  # set an image in background
background_label = tk.Label(MAIN, image=background_image)
#background_label = tk.Label(MAIN)
background_label.place(x=0, y=0, relwidth=1, relheight=1)


# SET ALL THE BUTTON IN THE MAIN CANVAS
button22 = Button(MAIN, text='IMAGE ANALYSIS', command=EDA, fg='red', highlightbackground='red')
button22.place(x=300, y=100)
button33 = Button(MAIN, text='Models Prediction', command=models, fg='red', highlightbackground='red')
button33.place(x=290, y=200)
button44 = Button(MAIN, text='Models Prediction for Transparent Flipoff', command=TRANSPARENT, fg='red', highlightbackground='red')
button44.place(x=250, y=300)
button55 = Button(MAIN, text='Neural Network', command=NN, fg='red', highlightbackground='red')
button55.place(x=300, y=400)
button66 = Button(MAIN, text='Vision Transformer', command=TRANS, fg='red', highlightbackground='red')
button66.place(x=290, y=500)


def exx():  # WARNING MESSAGE WHEN YOU TRY TO QUIT THE PROGRAM
    mess = tk.messagebox.askquestion(title="Warning", message="Are you sure to quit the program?")
    if mess == 'yes':
        MAIN.quit()

buttonfin = Button(MAIN, text='EXIT', command=exx, fg='red', highlightbackground='red')
buttonfin.place(x=350, y=600)


# SET THE MENU BAR (alternative to the buttons)
menubar = Menu(MAIN)
fun = Menu(menubar, tearoff=0)
fun.add_command(label="IMAGE ANALYSIS", command=EDA)
fun.add_command(label="Models prediction", command=models)
fun.add_command(label="Models prediction for Transparent Flipoff", command=TRANSPARENT)
fun.add_command(label="Neural Network", command=NN)
fun.add_command(label="Vision Transformer", command=TRANS)
fun.add_separator()

fun.add_command(label="Exit", command=MAIN.quit)
menubar.add_cascade(label="Functions", menu=fun)
edit = Menu(menubar, tearoff=0)

edit.add_command(label="Undo")
edit.add_separator()
menubar.add_cascade(label="Edit", menu=edit)
help = Menu(menubar, tearoff=0)
help.add_command(label="About")
menubar.add_cascade(label="Help", menu=help)
MAIN.config(menu=menubar)
MAIN.mainloop()