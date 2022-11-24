from keras.layers import Conv2D,Conv2DTranspose,MaxPooling2D,Dropout,Concatenate,Input
from keras import Model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def double_conv_block(prev_layer, filter_count):
   new_layer = Conv2D(filter_count, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(prev_layer)
   new_layer = Conv2D(filter_count, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(new_layer)
   return new_layer

def downsample_block(prev_layer, filter_count):
   skip_features = double_conv_block(prev_layer, filter_count)
   down_sampled = MaxPooling2D(2)(skip_features)
   down_sampled = Dropout(0.3)(down_sampled)
   return skip_features, down_sampled

def upsample_block(prev_layer, skipped_features, n_filters):
   upsampled = Conv2DTranspose(n_filters, 3, 2, padding="same")(prev_layer)
   upsampled = Concatenate()([upsampled, skipped_features])
   upsampled = Dropout(0.3)(upsampled)
   upsampled = double_conv_block(upsampled, n_filters)
   return upsampled

def make_unet():
   inputs = Input(shape=(128,128,1))


   skipped_fmaps_1, downsample_1 = downsample_block(inputs, 64)
   skipped_fmaps_2, downsample_2 = downsample_block(downsample_1, 128)
   skipped_fmaps_3, downsample_3 = downsample_block(downsample_2, 256)
   skipped_fmaps_4, downsample_4 = downsample_block(downsample_3, 512)

   bottleneck = double_conv_block(downsample_4, 1024)
   
   upsample_1 = upsample_block(bottleneck, skipped_fmaps_4, 512)
   upsample_2 = upsample_block(upsample_1, skipped_fmaps_3, 256)
   upsample_3 = upsample_block(upsample_2, skipped_fmaps_2, 128)
   upsample_4 = upsample_block(upsample_3, skipped_fmaps_1, 64)


   outputs = Conv2D(1, 1, padding="same", activation = "sigmoid")(upsample_4)

   unet_model = Model(inputs, outputs, name="U-Net")

   return unet_model

