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

WIDTH  = model_setting.WIDTH
HEIGHT = model_setting.HEIGHT

def sample_list_to_data_set(sample_list, label_count):
    fn_list = [ sample['fn'] for sample in sample_list ]
    img_list = load_img_list(fn_list)
    label_idx_list = np.array([ sample['label_idx'] for sample in sample_list ])
    label_onehot_list = np_utils.to_categorical(label_idx_list, label_count)
    return img_list, label_onehot_list

def load_img_list(fn_list):
    img_list = [ load_img(fn) for fn in fn_list ]
    return np.array(img_list)

def load_img(fn):
    img = classifier.load_img(fn)
    img = classifier.preprocess_img(img)
    return img

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
    hdf5_fn = mirror_data['hdf5_fn']
    
    train_valid_sample_list = sample_list[test_sample_count:]

    valid_start = mirror_data['valid_start']
    valid_end   = mirror_data['valid_end']
    train_sample_list = train_valid_sample_list[:valid_start]+train_valid_sample_list[valid_end:]
    valid_sample_list = train_valid_sample_list[valid_start:valid_end]
        
    random.shuffle(train_sample_list)

    # create model
    model = model_setting.create_model(label_count)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        
    train_img_list, train_label_onehot_list = sample_list_to_data_set(train_sample_list,label_count)
    valid_img_list, valid_label_onehot_list = sample_list_to_data_set(valid_sample_list,label_count)

    checkpointer = ModelCheckpoint(filepath=hdf5_fn, verbose=1, save_best_only=True)
    
    train_turn_count = math.floor(len(train_sample_list)**(1/3))
    batch_size = math.ceil(len(train_sample_list)/train_turn_count)
    batch_size = 100
    
    epochs = train_unit_data['epochs']
    model.fit(train_img_list, train_label_onehot_list,
        validation_data=(valid_img_list, valid_label_onehot_list),
        epochs=epochs, batch_size=batch_size, callbacks=[checkpointer], verbose=1)
