# Import package
import argparse
from collections import Counter
import logging
import math
import os
import random
import re
import shutil
from shutil import copyfile
import sys
import threading
import time
#import utils
import zipfile
# Data Science packages
import keras
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
# from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.io import loadmat

### SET UP GPU WITH HOROVOD
# import horovod.tensorflow as hvd
# hvd.init()
# ==> For TF 1.X
# config = tf.configProto()
# print(f"str(hvd.local_rank()): {str(hvd.local_rank())}")
# config.gpu_options.visible_device_list = str(hvd.local_rank())
# ==> For TF 2.X
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# if gpus:
#     tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

print("====> Start running imdb_age_distributed_model1_v2.0.py")
# Understand the file structure of dataset
DATA_PATH = '/fs/scratch/PAA0023/dong760/PlantVillage-Dataset/raw/color/'
# train_dir = "/fs/scratch/PAS1777/imdb_wiki/dataset/{}/train".format(db)
# test_dir = "/fs/scratch/PAS1777/imdb_wiki/dataset/{}/test".format(db)

# Understand the # of classes:
Classes = os.listdir(DATA_PATH)
print(f"Classes: {Classes}")

# Display an images as sample:
# import PIL
# import PIL.Image
# roses = list(data_dir.glob('Corn_(maize)___Common_rust_/*'))
# PIL.Image.open(str(roses[0]))

# Let's figure out a way to understand the shape of image
first_folder_path =os.path.join(DATA_PATH, os.listdir(DATA_PATH)[0])
print(DATA_PATH)
img_path = os.path.join(first_folder_path, os.listdir(first_folder_path)[0])
print(f"sample image path: {img_path}")
img = mpimg.imread(img_path)
print(f"Image size: {img.shape}") # ==> Image size: (256, 256, 3)


# Model configuration
BASE_DIR = "/users/PAA0023/dong760/plant_leaves_diagnosis"
MODEL_NAME = 'baseline_sparse_categorical_model'
batch_size = 32
no_epochs = 2
img_width, img_height, img_num_channels = 256, 256, 3
loss_function = 'sparse_categorical_crossentropy' # 'mean_squared_error', 'mean_absolute_error', 'binary_crossentropy', 'sparse_categorical_crossentropy'
metrics = 'accuracy' # 'precision', 'recall'
no_classes = len(os.listdir(DATA_PATH))
optimizer = tf.keras.optimizers.Adam(0.001) # SGD, Adagrad, RMSprop
validation_split = 0.2
verbosity = 2 # 0 = silent, 1 = progress bar, 2 = one line per epoch. 
steps_per_epoch = 10 # Total number of steps for one epochs
lr = 0.001
momentum=0.9

DESIRED_ACCURACY = 0.99
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>DESIRED_ACCURACY):
            print("\nReached ",DESIRED_ACCURACY, "% accuracy so cancelling training!")
            self.model.stop_training = True
callbacks = myCallback()


# Setting up the ImageGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=validation_split)
validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=validation_split)

train_generator = train_datagen.flow_from_directory(
    DATA_PATH,
    target_size=(img_width, img_height),  # (256, 256, 3)
    color_mode='rgb', #  "grayscale", "rgb", "rgba", Default: "rgb". Whether the images will be converted to have 1, 3, or 4 channels.
    class_mode = 'sparse', # One of "categorical", "binary", "sparse", "input", or None.
#     label_mode='int', # 'int': encoded as integer for sparse_categorical_crossentropy loss, 'categorical': encoded as categorical vector for categorical_crossentropy loss, 'binary': encoded as 0 or 1 for binary_crossentropy
    batch_size=batch_size,
    shuffle=False,
#     save_to_dir = '/users/PAA0023/dong760/plant_leaves_diagnosis/tmp/augmented_train', # This allows you to optionally specify a directory to which to save the augmented pictures being generated (useful for visualizing what you are doing).
    seed=123,
    subset="training",
    interpolation='nearest' #  Supported methods are "nearest", "bilinear", and "bicubic"
)
# print(type(train_generator))
# print(train_generator.shape)

validation_generator = train_datagen.flow_from_directory(
    DATA_PATH,
    target_size=(img_width, img_height),  # (256, 256, 3)
    color_mode='rgb', #  "grayscale", "rgb", "rgba", Default: "rgb". Whether the images will be converted to have 1, 3, or 4 channels.
    class_mode = 'sparse', # One of "categorical", "binary", "sparse", "input", or None.
#     label_mode='int', # 'int': encoded as integer for sparse_categorical_crossentropy loss, 'categorical': encoded as categorical vector for categorical_crossentropy loss, 'binary': encoded as 0 or 1 for binary_crossentropy
    batch_size=batch_size,
    shuffle=False,
#     save_to_dir = '/users/PAA0023/dong760/plant_leaves_diagnosis/tmp/augmented_valid', # This allows you to optionally specify a directory to which to save the augmented pictures being generated (useful for visualizing what you are doing).
    seed=123,
    subset="validation",
    interpolation='nearest' #  Supported methods are "nearest", "bilinear", and "bicubic"
)
# print(type(validation_generator))
# print(validation_generator.shape)

# Define some variable for Horovod
train_iterator = train_generator
train_size = train_iterator.n # OR len(train_iterator.filepaths), len(train_iterator.classes), len(train_iterator.filenames)
# batch_size = train_iterator.batch_size
# val_size = validation_iterator.n
# len(train_iterator.filepaths)
# len(train_iterator.classes)
# train_iterator.num_classes
# train_iterator.image_shape
# train_iterator.batch_size
# train_iterator.dtype
print(f"Train size: {train_size}")

# For vgg19
IMG_SHAPE = (img_width, img_height, img_num_channels)
VGG19_model = tf.keras.applications.VGG19(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
# For ResNet50
ResNet50_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
# For InceptionV3
InceptionV3_model = InceptionV3(input_shape=IMG_SHAPE,
                                 include_top = False,
                                 weights = 'imagenet') 
# ResNet50_model.summary()

# Lock each layer in pre_trained_model
for layer in InceptionV3_model.layers:
    layer.trainable = False

# If you want to do fine-tunnning
# For VGG19
# VGG19_last_layer = VGG19_model.get_layer('block5_pool')
# For InceptionV3
# pre_trained_model.load_weights(local_weights_file) # Load the weights from previously downloaded file
InceptionV3_mixed7_layer = InceptionV3_model.get_layer('mixed7')

# Fine tuning the model: Define the new model by adding extra fully connected layers
last_output = InceptionV3_mixed7_layer.output
x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
output_layer = tf.keras.layers.Dense(no_classes, activation='softmax')(x)
model = tf.keras.Model(inputs=InceptionV3_model.input, outputs=output_layer)
# model.summary()

# Instantiate the Model object:
# More detail about compile: https://www.tensorflow.org/guide/keras/train_and_evaluate
# optimizer = tf.keras.optimizers.SGD(lr=lr, momentum=momentum)
# optimizer = tf.keras.optimizers.RMSprop(lr=0.0001)
# Horovod: adjust learning rate based on number of GPUs.
optimizer = tf.keras.optimizers.SGD(lr=lr*hvd.size(), momentum=momentum)

loss = tf.losses.SparseCategoricalCrossentropy()

# Define the checkpoint directory:
checkpoint_dir = './checkpoints/'
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

# Training the model
# history = model.fit(train_generator, epochs=no_epochs, verbose=verbosity, steps_per_epoch=steps_per_epoch)
history = model.fit(train_generator, 
                    epochs=no_epochs, 
                    verbose=verbosity,  
#                     callbacks=[callbacks],
#                     steps_per_epoch=steps_per_epoch,                    
#                     validation_steps=steps_per_epoch,
                    validation_data = validation_generator
                   )
print(history.history)

# Evaluate the performance
print("Prediction Result: ")
valid_loss, valid_acc = model.evaluate(validation_generator)
print('Validation accuracy:', valid_acc, "Validation accuracy:", valid_loss)

# Get the timestamp
from datetime import datetime 
now = datetime.now()
# print("now =", now)
# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y_%H%M%S")
print("date and time =", dt_string)	
# dd/mm/YY
# today = date.today()
# d1 = today.strftime("%d/%m/%Y")
# print("d1 =", d1)


pd.DataFrame(history.history).plot()
plt.savefig('/users/PAA0023/dong760/plant_leaves_diagnosis/plots/'+"PANDA_"+MODEL_NAME+"_"+str(dt_string)+'.png')

# Plot the figure
plt.figure()
plt.ylabel("Loss (training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0,2])
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

plt.figure()
plt.ylabel("Accuracy (training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.savefig('/users/PAA0023/dong760/plant_leaves_diagnosis/plots/'+MODEL_NAME+"_"+str(dt_string)+'.png')

def plot_graphs(history, string, filename):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.savefig('/users/PAA0023/dong760/plant_leaves_diagnosis/plots/'+MODEL_NAME+"_"+str(dt_string)+'.png')
    # plt.show()
# plot_graphs(history, 'acc', 'test01')
# plot_graphs(history, 'loss', 'test01')

# Saving the model
saved_model_path = BASE_DIR+"/tmp/"+MODEL_NAME+"_"+str(dt_string)
tf.saved_model.save(model, saved_model_path)