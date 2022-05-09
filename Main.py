#Real-time road detection implementation 

import os
import cv2
import tqdm
import imageio
import cProfile
import numpy as np
import glob as glob
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
config.log_device_placement = True # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)
## ---- end Memory setting ----

img_height = 1024
img_width = 1232
batch_size= 1

# Image Data Generators for images and maks from training, respectively validation dataset
train_datagen_image = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_datagen_mask = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
valid_datagen_image = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
valid_datagen_mask = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Datagen Image flow for images and masks from the training dataset 
train_flow_image = train_datagen_image.flow_from_directory("../UHA_Dataset/img_train",
                                               target_size=(img_height,img_width),
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               shuffle=False,
                                               seed=5,
                                               class_mode = None)
train_flow_mask = train_datagen_image.flow_from_directory("../UHA_Dataset/mask_train",
                                               target_size=(img_height,img_width),
                                               color_mode='grayscale',
                                               batch_size=batch_size,
                                               shuffle=False,
                                               seed=5,
                                               class_mode = None)

# Datagen Image flow for images and masks from the validation dataset 
valid_flow_image = valid_datagen_image.flow_from_directory("../UHA_Dataset//img_valid",
                                               target_size=(img_height,img_width),
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               shuffle=False,
                                               seed=8,
                                               class_mode = None)
valid_flow_mask = valid_datagen_image.flow_from_directory("../UHA_Dataset//mask_valid",
                                               target_size=(img_height,img_width),
                                               color_mode='grayscale',
                                               shuffle=False,
                                               batch_size=batch_size,
                                               seed=8,
                                               class_mode = None)

N1 = len(train_flow_mask)
train_masks = np.zeros((N1,img_height, img_width, 1), dtype=np.float32)
for n in range(N1):
    mask = train_flow_mask[n]
    mask = np.reshape(mask,(img_height,img_width))
    mask_road = np.zeros((img_height,img_width, 1), dtype=np.float32)
    mask_road[np.where(mask==0.6666667)[0], np.where(mask==0.6666667)[1]]=1    
    mask_road = np.reshape(mask_road,(img_height,img_width,1))
    train_masks[n]=mask_road
train_flow_mask1 = valid_datagen_image.flow(train_masks,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               seed=5)  
N2 = len(valid_flow_mask)
valid_masks = np.zeros((N2,img_height, img_width, 1), dtype=np.float32)
for n in range(N2):
    mask1 = valid_flow_mask[n]
    mask1 = np.reshape(mask1,(img_height, img_width))
    mask_road1 = np.zeros((img_height, img_width, 1), dtype=np.float32)
    mask_road1[np.where(mask1==0.6666667)[0], np.where(mask1==0.6666667)[1]]=1                    
    valid_masks[n] = mask_road1
    
