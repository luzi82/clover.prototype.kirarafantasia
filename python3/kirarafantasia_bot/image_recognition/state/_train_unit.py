import os
import json
import sys
import random
import cv2
import numpy as np
#from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, CSVLogger
import json
from . import model as model_setting
#from . import classifier
import clover.common
import math
import time
from . import train
import gc
import clover.image_recognition

WIDTH  = model_setting.WIDTH
HEIGHT = model_setting.HEIGHT

#sample_list_to_data_set = train.sample_list_to_data_set
#load_img_list = train.load_img_list
#load_img = train.load_img

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='state classifier trainer unit')
    parser.add_argument('train_unit_path', help="train_unit_path")
    parser.add_argument('mirror_idx', type=int, help="mirror_idx")
    args = parser.parse_args()
    
    train_unit_path = args.train_unit_path
    mirror_idx = args.mirror_idx
    
    train_unit_data = clover.common.read_json(train_unit_path)
    mirror_data = train_unit_data['mirror_data_list'][mirror_idx]

    label_count = train_unit_data['label_count']
    test_sample_count = train_unit_data['test_sample_count']
    sample_list = train_unit_data['sample_list']
    train_valid_sample_data_dir_path = train_unit_data['train_valid_sample_data_dir_path']
    hdf5_fn = mirror_data['hdf5_fn']
    csvlog_fn = mirror_data['csvlog_fn']

    valid_start = mirror_data['valid_start']
    valid_end   = mirror_data['valid_end']

    train_sample_count = len(sample_list) - test_sample_count - valid_end + valid_start
    valid_sample_count = valid_end - valid_start
    
#    train_valid_sample_list = sample_list[test_sample_count:]
#
#    valid_start = mirror_data['valid_start']
#    valid_end   = mirror_data['valid_end']
#    train_sample_list = train_valid_sample_list[:valid_start]+train_valid_sample_list[valid_end:]
#    valid_sample_list = train_valid_sample_list[valid_start:valid_end]
#
#    random.shuffle(train_sample_list)
#        
#    train_img_list, train_label_onehot_list = sample_list_to_data_set(train_sample_list,label_count)
#    valid_img_list, valid_label_onehot_list = sample_list_to_data_set(valid_sample_list,label_count)

    train_valid_img_list, train_valid_label_onehot_list = train.load_data_set(train_valid_sample_data_dir_path)

    train_img_list = np.concatenate((train_valid_img_list[:valid_start],train_valid_img_list[valid_end:]))
    assert(train_img_list.shape==(train_sample_count,HEIGHT,WIDTH,3))
    train_label_onehot_list = np.concatenate((train_valid_label_onehot_list[:valid_start],train_valid_label_onehot_list[valid_end:]))
    assert(train_label_onehot_list.shape==(train_sample_count,label_count))
    valid_img_list = train_valid_img_list[valid_start:valid_end]
    assert(valid_img_list.shape==(valid_sample_count,HEIGHT,WIDTH,3))
    valid_label_onehot_list = train_valid_label_onehot_list[valid_start:valid_end]
    assert(valid_label_onehot_list.shape==(valid_sample_count,label_count))

    del train_valid_img_list, train_valid_label_onehot_list
    gc.collect()
    
    # shuffle train
    p = np.random.permutation(train_sample_count)
    train_img_list = train_img_list[p]
    train_label_onehot_list = train_label_onehot_list[p]
    
    # destory np link, free mem
    gc.collect()
    train_img_list = np.copy(train_img_list)
    gc.collect()
    train_label_onehot_list = np.copy(train_label_onehot_list)
    gc.collect()
    valid_img_list = np.copy(valid_img_list)
    gc.collect()
    valid_label_onehot_list = np.copy(valid_label_onehot_list)
    gc.collect()

    # xy layer
    xy_layer = clover.image_recognition.xy_layer(WIDTH,HEIGHT)
    train_xy_layer = np.broadcast_to(xy_layer,(train_sample_count,HEIGHT,WIDTH,2))
    valid_xy_layer = np.broadcast_to(xy_layer,(valid_sample_count,HEIGHT,WIDTH,2))

    # create model
    model = model_setting.create_model(label_count)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        
    checkpointer = ModelCheckpoint(filepath=hdf5_fn, verbose=1, save_best_only=True)
    csv_logger = CSVLogger(filename=csvlog_fn)
    
    #train_turn_count = math.floor(len(train_sample_list)**(1/3))
    #batch_size = math.ceil(len(train_sample_list)/train_turn_count)
    batch_size = 100
    
    epochs = train_unit_data['epochs']
    model.fit([train_img_list, train_xy_layer], train_label_onehot_list,
        validation_data=([valid_img_list, valid_xy_layer], valid_label_onehot_list),
        epochs=epochs, batch_size=batch_size, callbacks=[checkpointer], verbose=1)
