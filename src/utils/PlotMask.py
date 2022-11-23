import matplotlib.pyplot as plt
import numpy as np

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