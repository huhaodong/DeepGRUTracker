import matplotlib
import matplotlib.pyplot as plt 
import tensorflow as tf
import numpy as np
from skimage import io
import cv2
from skimage.transform import resize

def saveFeatureMap(img=None ,featureMap=None ,outputPath=None):
    '''
        dis: save the feature map to output path
        prem:
            img -- origin image
            featureMap -- the feature map for show  [x,y,c]
            outputPath -- the output picture

    '''
    cam = np.ones(featureMap.shape[0 : 2], dtype = np.float32)
    for i in range(featureMap.shape[2]):
        cam += featureMap[:,:,i]
    cam = np.maximum(cam,0)
    cam = cam / np.max(cam)
    cam = resize(cam, (img.shape[0],img.shape[1]))

    cam3 = np.expand_dims(cam ,axis=2)
    cam3 = np.tile(cam3 ,[1,1,3])

    cam3 = cv2.applyColorMap(np.uint8(255*cam3), cv2.COLORMAP_JET)
    cam3 = cv2.cvtColor(cam3, cv2.COLOR_BGR2RGB)
    
    img = img.astype(float)
    img /= img.max()

    alpha = 0.009
    new_img = img + alpha*cam3
    new_img /= new_img.max()

    io.imshow(new_img)
    io.imsave(outputPath ,new_img)
