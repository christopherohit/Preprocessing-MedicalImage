from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np



def loadUnet(path):
    model = load_model(path)
    return model

def preprocessImage(pathImage):
    image = Image.open(pathImage)
    image = image.resize((128, 128))
    image = np.asarray(image)
    image = image.reshape(image.shape[0], image.shape[1], 1)
    return image