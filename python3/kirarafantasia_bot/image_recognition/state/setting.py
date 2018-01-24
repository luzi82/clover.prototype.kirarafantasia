# avoid tensorflow dependency
import numpy as np
import cv2
import clover.image_recognition

NAME = 'state'
HEIGHT = 40
WIDTH  = 71

def should_ignore(img):
    IGNORE_STDDEV_MAX = 0.2
    IGNORE_DIFF_MAX   = 1

    img_diff = np.max(np.max(img,axis=(0,1))-np.min(img,axis=(0,1)))
    if img_diff > IGNORE_DIFF_MAX:
        return False
    img_stddev = np.max(np.std(img,axis=(0,1)))
    if img_stddev > IGNORE_STDDEV_MAX:
        return False
    return True

def preprocess_img(img):
    img = cv2.resize(img,dsize=(WIDTH,HEIGHT),interpolation=cv2.INTER_AREA)
    #img = np.append(img,clover.image_recognition.xy_layer(WIDTH,HEIGHT),axis=2)
    return img
