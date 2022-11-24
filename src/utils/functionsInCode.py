from tqdm.auto import tqdm
import cv2
import os
import numpy as np
from PIL import Image

def OutputModified(mask):
    check = [i for i in mask if "mask" in i]
    print("Total mask that has modified name:",len(check))
    return check

def getData(training_files, testing_files, image_path, mask_path, X_shape, flag = "test"):
    im_array = []
    mask_array = []
    
    if flag == "test":
        for i in tqdm(testing_files): 
            im = cv2.resize(cv2.imread(os.path.join(image_path,i)),(X_shape,X_shape))[:,:,0]
            mask = cv2.resize(cv2.imread(os.path.join(mask_path,i)),(X_shape,X_shape))[:,:,0]
            
            im_array.append(im)
            mask_array.append(mask)
        
        return im_array,mask_array
    
    if flag == "train":
        for i in tqdm(training_files): 
            im = cv2.resize(cv2.imread(os.path.join(image_path,i.split("_mask")[0]+".png")),(X_shape,X_shape))[:,:,0]
            mask = cv2.resize(cv2.imread(os.path.join(mask_path,i+".png")),(X_shape,X_shape))[:,:,0]

            im_array.append(im)
            mask_array.append(mask)

        return im_array,mask_array

def make_dataset(images, v_images, v_masks, masks, validation=False):
    x = []
    y = []
    if(validation):
        for i,(image,mask) in enumerate(zip(v_images[:10000],v_masks[:10000])):
            print("\r"+str(i)+"/"+str(len(v_images)),end="")

            image = Image.open(os.path.join("../input/lung-mask-image-dataset/ChestXray/val/image",image)).convert('L')
            mask = Image.open(os.path.join("../input/lung-mask-image-dataset/ChestXray/val/mask",mask)).convert('L')

            image = np.asarray(image.resize((128,128)))/255.
            mask = np.asarray(mask.resize((128,128)))/255.

            x.append(image)
            y.append(mask)
    else:
        for i,(image,mask) in enumerate(zip(images[:10000],masks[:10000])):
            print("\r"+str(i)+"/"+str(len(images)),end="")
            
            image = Image.open(os.path.join("../input/lung-mask-image-dataset/ChestXray/train/image",image)).convert('L')
            mask = Image.open(os.path.join("../input/lung-mask-image-dataset/ChestXray/train/mask",mask)).convert('L')

            image = np.asarray(image.resize((128,128)))/255.
            mask = np.asarray(mask.resize((128,128)))/255.

            x.append(image)
            y.append(mask)

    return np.array(x),np.array(y)