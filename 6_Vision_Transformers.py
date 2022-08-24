# ELISA SALVI FILE NUMBER 6 TRANSFORMERS MODEL FOR IMAGE CLASSIFICATION
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
from shutil import copyfile, copy2
import cv2 as cv
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange
#from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
#from torchsummary import summary
from PIL import Image as img
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

path_folder = '/Users/davidesalvi/Desktop/stage_project_py'       # path folder stage_project_py
df_AllGood = pd.read_csv(path_folder + "/df_notonly_AllGood.csv")
image_dir_tran = path_folder + "/images_notonlygood/allimagesTrans"
path_Transformers = path_folder + "/images_notonlygood/Transformers/"

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
print('stop')



# Save the divided data sets into folders according to the class
for image, type_ in zip(image_dir_tran + '/' + X_train['Image'], y_train):  # for training set
    copy2(image, path_Transformers+'train/' + str(type_))   # salva le immagini dalla cartella generale, nelle cartelle val e train

for image, type_ in zip(image_dir_tran + '/' + X_val['Image'], y_val):       # for validation set
    copy2(image, path_Transformers+'val/' + str(type_))






# https://keras.io/examples/vision/image_classification_with_vision_transformer/

# Configure the hyperparameters
input_shape = (224, 224, 3)
num_classes = 9
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 2
patch_size = 16  # Size of the patches to be extract from the input images
num_patches = (IMAGE_WIDTH // patch_size) ** 2 # 224/16
projection_dim = 64
num_heads = 4
transformer_units = [projection_dim * 2, projection_dim,]  # Size of the transformer layers
transformer_layers = 8   # sarebbe 8
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
# The ViT model consists of multiple Transformer blocks, which use the layers.MultiHeadAttention layer as a self-attention mechanism applied to the sequence of patches.
# The Transformer blocks produce a [batch_size, num_patches, projection_dim] tensor, which is processed via an classifier head with softmax to produce the final class probabilities output.
# Unlike the technique described in the paper, which prepends a learnable embedding to the sequence of encoded patches to serve as the image representation,
# all the outputs of the final Transformer block are reshaped with layers.Flatten() and used as the image representation input to the classifier head.
# Note that the layers.GlobalAveragePooling1D layer could also be used instead to aggregate the outputs of the Transformer block, especially when the number of patches and the projection dimensions are large.
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

# Compile, train, and evaluate the mode
def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

    model.compile(optimizer=optimizer, loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),],)
    model.summary()

    checkpoint_filepath = "/tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor="val_accuracy", save_best_only=True, save_weights_only=True,)

    history = model.fit(x=XXtrain, y=y_train, batch_size=batch_size, epochs=num_epochs, validation_split=0.1, callbacks=[checkpoint_callback],)

    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(XXval, y_val)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history

vit_classifier = create_vit_classifier()
history = run_experiment(vit_classifier)


"""
model = create_vit_classifier()
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc') > 0.7) and (logs.get('val_acc') > 0.7):
            print('\n reached 70% accuarcy so stopping training')
            self.model.stop_training = True
callbacks = myCallback()
history = model.fit(x=XXtrain, y=y_train, batch_size=20, epochs=100, callbacks=[tf.keras.callbacks.ReduceLROnPlateau(), callbacks])  # doesn'do all the 100 epoch, but it finishes when the desired accuracy is reached.
"""











