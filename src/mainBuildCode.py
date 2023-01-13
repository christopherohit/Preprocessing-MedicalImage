import os
import numpy as np
from PIL import Image
from src.utils.functionsInCode import make_dataset
import matplotlib.pyplot as plt
from src.architureUnet import make_unet
from src.utils.PlotMask import model_plotter

images = os.listdir("/content/drive/MyDrive/Dataset/Medicine Image/ChestXray/train/image")
masks = os.listdir("/content/drive/MyDrive/Dataset/Medicine Image/ChestXray/train/mask")

v_images = os.listdir("/content/drive/MyDrive/Dataset/Medicine Image/ChestXray/val/image")
v_masks = os.listdir("/content/drive/MyDrive/Dataset/Medicine Image/ChestXray/val/mask")

print(len(images)==len(masks))
print(len(v_images)==len(v_masks))

print(masks[:10])
print(images[:10])

print(masks.sort())
print(images.sort())
print(v_masks.sort())
print(v_images.sort())


temp = 0
while temp<20:
    print(images[temp].split("_")[1].split(".")[0] == masks[temp].split("_")[1].split(".")[0])
    temp+=1

import random

i = random.randint(0,len(images)-1)
img = np.asarray(Image.open(os.path.join("/content/drive/MyDrive/Dataset/Medicine Image/ChestXray/train/image",images[i])))
mask = np.asarray(Image.open(os.path.join("/content/drive/MyDrive/Dataset/Medicine Image/ChestXray/train/mask",masks[i])))
print(img.shape,mask.shape)


print(len(images))


plt.subplot(1,2,1)
plt.imshow(img[:,:,:3]) #rgba to rgb

plt.subplot(1,2,2)
plt.imshow(mask)

plt.show()

x,y = make_dataset(images= images, v_images= v_images, v_masks= v_masks,
                    masks= masks)

v_x,v_y = make_dataset(images= images, v_images= v_images, v_masks= v_masks,
                    masks= masks, validation=True)


x,y = np.expand_dims(x,axis=-1),np.expand_dims(y,axis=-1)
v_x,v_y = np.expand_dims(v_x,axis=-1),np.expand_dims(v_y,axis=-1)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_gen_args = dict(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode="nearest",
    horizontal_flip=True,
    vertical_flip=True,
)

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

seed = 1

image_generator = image_datagen.flow(
    x,
    batch_size=64,
    seed=seed)

mask_generator = mask_datagen.flow(
    y,
    batch_size=64,
    seed=seed)

train_generator = zip(image_generator, mask_generator)


from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_test_datagen = ImageDataGenerator()
mask_test_datagen = ImageDataGenerator()

seed = 1

image_test_generator = image_test_datagen.flow(
    v_x,
    batch_size=64,
    seed=seed)

mask_test_generator = mask_test_datagen.flow(
    v_y,
    batch_size=64,
    seed=seed)

valid_generator = zip(image_test_generator, mask_test_generator)


i = random.randint(0,len(x))

img = x[i]
mask = y[i]

plt.subplot(1,2,1)
plt.imshow(np.squeeze(img))

plt.subplot(1,2,2)
plt.imshow(np.squeeze(mask))

plt.show()

u_net = make_unet()
model_plotter(u_net)
u_net.compile(optimizer="adam",loss="binary_crossentropy",metrics="accuracy")
model_history = u_net.fit(train_generator,epochs=20,validation_data=valid_generator,steps_per_epoch = int(x.shape[0] / 64),validation_steps = int(v_x.shape[0] / 64))
u_net.save("u_net.h5")


i = random.randint(0,len(v_x)-1)

original = v_x[i].copy()
original_mask = v_y[i].copy()

mask = u_net.predict(np.expand_dims(original,axis=0))


segmented = np.squeeze(original).copy()
segmented[np.squeeze(mask)<0.2] = 0

plt.subplot(1,4,1)
plt.imshow(np.squeeze(original),cmap="gray")
plt.title("x-ray")
plt.axis("off")

plt.subplot(1,4,2)
plt.imshow(segmented,cmap="gray")
plt.title("segmented")
plt.axis("off")

plt.subplot(1,4,3)
plt.imshow(np.squeeze(mask[0]),cmap="gray")
plt.title("predicted mask")
plt.axis("off")

plt.subplot(1,4,4)
plt.imshow(np.squeeze(original_mask),cmap="gray")
plt.title("  original mask")
plt.axis("off")


plt.show()
