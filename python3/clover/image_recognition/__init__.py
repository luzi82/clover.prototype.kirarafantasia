import os
import numpy as np
import cv2
from functools import lru_cache
import csv
import shutil

def get_label_state_list():
    label_state_path = os.path.join('image_recognition','label','state')
    label_name_list = os.listdir(label_state_path)
    label_name_list = filter(lambda v:os.path.isfile(os.path.join(label_state_path,v)),label_name_list)
    label_name_list = filter(lambda v:v.endswith('.txt'),label_name_list)
    label_name_list = [ i[:-4] for i in label_name_list]
    label_name_list = sorted(label_name_list)
    return label_name_list

def load_img(fn):
    img = cv2.imread(fn).astype('float32')*2/255-1
    return img

def xy_layer(w,h):
    xx = np.array(list(range(w))).astype('float32')*2/(w-1)-1
    xx = np.tile(xx,h)
    xx = np.reshape(xx,(h,w,1))
    yy = np.array(list(range(h))).astype('float32')*2/(h-1)-1
    yy = np.repeat(yy,w)
    yy = np.reshape(yy,(h,w,1))
    xxyy = np.append(xx,yy,axis=2)
    return xxyy

def xy1_layer(w,h):
    ret = xy_layer(w,h)

    oo = np.ones(shape=(h,w,1),dtype=np.float)
    ret = np.append(ret,oo,axis=2)

    return ret

@lru_cache(maxsize=4)
def get_raw_image_timestamp_list():
    raw_image_timestamp_list = os.listdir(os.path.join('image_recognition','raw_image'))
    raw_image_timestamp_list = filter(lambda v:os.path.isdir(os.path.join('image_recognition','raw_image',v)),raw_image_timestamp_list)
    raw_image_timestamp_list = [ int(i) for i in raw_image_timestamp_list ]
    return raw_image_timestamp_list

def get_timestamp(v):
    return max(filter(lambda i:i<v,get_raw_image_timestamp_list()))
