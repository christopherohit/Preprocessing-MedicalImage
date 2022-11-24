import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import plot_model
from PIL import Image
from matplotlib.pyplot import figure
import cv2
#perform sanity check

def plotMask(X,y):
    sample = []
    
    for i in range(6):
        left = X[i]
        right = y[i]
        combined = np.hstack((left,right))
        sample.append(combined)
        
        
    for i in range(0,6,3):

        plt.figure(figsize=(25,10))
        
        plt.subplot(2,3,1+i)
        plt.imshow(sample[i])
        
        plt.subplot(2,3,2+i)
        plt.imshow(sample[i+1])
        
        
        plt.subplot(2,3,3+i)
        plt.imshow(sample[i+2])
        
        plt.show()

def PlotMetric(loss_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
    ax1.plot(loss_history.history['loss'], '-', label = 'Loss')
    ax1.plot(loss_history.history['val_loss'], '-', label = 'Validation Loss')
    ax1.legend()

    ax2.plot(100*np.array(loss_history.history['binary_accuracy']), '-', 
            label = 'Accuracy')
    ax2.plot(100*np.array(loss_history.history['val_binary_accuracy']), '-',
            label = 'Validation Accuracy')
    ax2.legend()

def PlotTest(validation_vol, model, validation_seg):
    pred_candidates = np.random.randint(1,validation_vol.shape[0],10)
    preds = model.predict(validation_vol)

    plt.figure(figsize=(20,10))

    for i in range(0,9,3):
        plt.subplot(3,3,i+1)
        
        plt.imshow(np.squeeze(validation_vol[pred_candidates[i]]))
        plt.xlabel("Base Image")
        
        
        plt.subplot(3,3,i+2)
        plt.imshow(np.squeeze(validation_seg[pred_candidates[i]]))
        plt.xlabel("Mask")
        
        plt.subplot(3,3,i+3)
        plt.imshow(np.squeeze(preds[pred_candidates[i]]))
        plt.xlabel("Prediction")

def SegmentImage(model,path,img_shape = (512,512),threshold = 0.5):
    '''
    **********Input**************
    model: segmentation model (h5)
    path: filepath to image (string)
    img_shape: shape of the image(IMG_WIDTH,IMG_HEIGHT) used in segmenation model
    threshold: float value varing between 0 and 1, thresholding the mask
    *********Output*************
    return: Segment mask, segmented image, original image
    '''
    IMG_WIDTH,IMG_HEIGHT = img_shape
    ori_x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    ori_x = cv2.resize(ori_x, (IMG_HEIGHT,IMG_WIDTH))
    x = ori_x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    y_pred = model.predict(x)[0] > threshold
    y_pred = y_pred.astype(np.int32)
    plt.imsave('mask.jpeg',np.squeeze(y_pred),cmap='gray')
    maskapply = cv2.imread('mask.jpeg')
    maskapply = cv2.cvtColor(maskapply, cv2.COLOR_BGR2GRAY)
    chest_image = ori_x
    chest_image = cv2.resize(chest_image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_NEAREST)
    masked_image = cv2.bitwise_and(maskapply,chest_image)
    return maskapply,masked_image,chest_image
    
def model_plotter(model):
    plot_model(
      model,
      to_file="model.png",
      show_shapes=True,
      show_dtype=True,
      show_layer_names=True,
      rankdir="TB",
      expand_nested=True,
      dpi=96,
      layer_range=None,  )
    figure(figsize=(100,100))
    plt.imshow(np.asarray(Image.open("model.png")))
    plt.axis("off")
    plt.show()  