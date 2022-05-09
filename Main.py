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
