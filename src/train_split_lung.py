import numpy as np 
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import os
from cv2 import imread, createCLAHE 
import cv2
from glob import glob
import matplotlib.pyplot as plt
from src.utils.functionsInCode import OutputModified, getData
from src.utils.PlotMask import plotMask
from src.utils.modelUnet import unet
from src.utils.metrics import dice_coef, dice_coef_loss
from tensorflow.keras.optimizers import Adam

image_path = os.path.join("/content/drive/MyDrive/Dataset/Medicine Image/Lung Segmentation/CXR_png")
mask_path = os.path.join("/content/drive/MyDrive/Dataset/Medicine Image/Lung Segmentation","masks/")

# we have 704 masks but 800 images. Hence we are going to
# make a 1-1 correspondance from mask to images, not the usual other way.
images = os.listdir(image_path)
mask = os.listdir(mask_path)
mask = [fName.split(".png")[0] for fName in mask]
image_file_name = [fName.split("_mask")[0] for fName in mask]

check = OutputModified(mask= mask)

testing_files = set(os.listdir(image_path)) & set(os.listdir(mask_path))
training_files = check


# Load training and testing data
dim = 256*2
X_train,y_train = getData(training_files= training_files, testing_files= testing_files,
                            image_path= image_path, mask_path= mask_path, 
                            X_shape= dim, flag="train")
X_test, y_test = getData(training_files= training_files, testing_files= testing_files,
                            image_path= image_path, mask_path= mask_path,
                            X_shape= dim)


print("training set")
plotMask(X_train,y_train)
print("testing set")
plotMask(X_test,y_test)

def StanderizeVarible():
    X_train = np.array(X_train).reshape(len(X_train),dim,dim,1)
    y_train = np.array(y_train).reshape(len(y_train),dim,dim,1)
    X_test = np.array(X_test).reshape(len(X_test),dim,dim,1)
    y_test = np.array(y_test).reshape(len(y_test),dim,dim,1)
    assert X_train.shape == y_train.shape
    assert X_test.shape == y_test.shape
    images = np.concatenate((X_train,X_test),axis=0)
    mask  = np.concatenate((y_train,y_test),axis=0)
    return images, mask



def ChooseModel():
    model = unet(input_size=(512,512,1))
    return model
model = ChooseModel()
model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss,
                  metrics=[dice_coef, 'binary_accuracy'])
model.summary()