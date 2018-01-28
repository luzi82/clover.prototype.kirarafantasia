import os
import json
import sys
import random
import cv2
import numpy as np
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import json
from . import model as model_setting
from . import classifier
import clover.common
import math
import time
import subprocess
from . import train as train
import clover.image_recognition

WIDTH  = model_setting.WIDTH
HEIGHT = model_setting.HEIGHT

sample_list_to_data_set = train.sample_list_to_data_set
#load_img_list = train.load_img_list
#load_img = train.load_img

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='state classifier trainer unit')
    parser.add_argument('train_unit_path', help="train_unit_path")
    args = parser.parse_args()

    train_unit_path = args.train_unit_path
    train_unit_data = clover.common.read_json(train_unit_path)

    label_count = train_unit_data['label_count']
    test_sample_count = train_unit_data['test_sample_count']
    sample_list = train_unit_data['sample_list']
    mirror_count = train_unit_data['mirror_count']

    test_sample_list        = sample_list[:test_sample_count]
    test_img_list,  test_label_onehot_list  = sample_list_to_data_set(test_sample_list ,label_count)
<<<<<<< HEAD
        
=======

>>>>>>> origin/v00a
    model = model_setting.create_model(label_count)

    for mirror_idx in range(mirror_count):
        hdf5_fn = os.path.join('image_recognition','model','state','weight.{}.hdf5'.format(mirror_idx))

        model.load_weights(hdf5_fn)
    
        test_predictions = [np.argmax(model.predict(np.expand_dims(img_list, axis=0))) for img_list in test_img_list]
        test_accuracy = np.sum(np.array(test_predictions)==np.argmax(test_label_onehot_list, axis=1))/len(test_predictions)
        print('Mirror {0}/{1} test accuracy: {2:.4f}'.format(mirror_idx,mirror_count,test_accuracy))
