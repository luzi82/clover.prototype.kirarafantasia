import os
import json
import sys
import random
import cv2
import numpy as np
#from keras.utils import np_utils
#from keras.callbacks import ModelCheckpoint
import json
#from . import model as model_setting
#from . import classifier
import clover.common
import clover.image_recognition
import math
import time
import subprocess
import gc
from . import setting

#WIDTH  = model_setting.WIDTH
#HEIGHT = model_setting.HEIGHT

def save_data_set(dir_path, img_list, label_onehot_list):
    clover.common.reset_dir(dir_path)
    #img_list, label_onehot_list = sample_list_to_data_set(sample_list,label_count)
    img_list_fn          = os.path.join(dir_path,'img_list.npy')
    label_onehot_list_fn = os.path.join(dir_path,'label_onehot_list.npy')
    np.save(img_list_fn,img_list)
    np.save(label_onehot_list_fn,label_onehot_list)

def load_data_set(dir_path):
    img_list_fn          = os.path.join(dir_path,'img_list.npy')
    label_onehot_list_fn = os.path.join(dir_path,'label_onehot_list.npy')
    img_list = np.load(img_list_fn)
    label_onehot_list = np.load(label_onehot_list_fn)
    return img_list, label_onehot_list

def sample_list_to_data_set(sample_list, label_count):
    fn_list = [ sample['fn'] for sample in sample_list ]
    img_list = load_img_list(fn_list)
    label_idx_list = np.array([ sample['label_idx'] for sample in sample_list ])
    #label_onehot_list = np_utils.to_categorical(label_idx_list, label_count)
    label_onehot_list = np.zeros((len(label_idx_list),label_count),dtype=np.float32)
    label_onehot_list[np.arange(len(label_idx_list)), label_idx_list] = 1
    return img_list, label_onehot_list

def load_img_list(fn_list):
    start_time = time.time()
    img_list = [ load_img(fn) for fn in fn_list ]
    end_time = time.time()
    diff_time = end_time-start_time
    print('load_img_list: start_time={}, end_time={}, diff_time={}'.format(start_time,end_time,diff_time));
    return np.array(img_list)

def load_img(fn):
    cache_fn = os.path.join('cache','state',fn)
    cache_fn = cache_fn + '.npy'
    if os.path.exists(cache_fn):
        return np.load(cache_fn)
    img = clover.image_recognition.load_img(fn)
    img = setting.preprocess_img(img)
    clover.common.makedirs(os.path.dirname(cache_fn))
    np.save(cache_fn,img)
    return img

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='state classifier trainer')
    parser.add_argument('epochs', nargs='?', type=int, help="epochs")
    #parser.add_argument('batch_size', nargs='?', type=int, help="batch_size")
    parser.add_argument('mirror_count', nargs='?', type=int, help="mirror count")
    parser.add_argument('--summaryonly', action='store_true', help="summary only")
    args = parser.parse_args()

    assert(((args.epochs!=None)and(args.mirror_count!=None))or(args.summaryonly))

    now = int(time.time())

    TRAIN_UNIT_PATH_FORMAT = '/tmp/IWNFWFKRMF-{}.json'
    train_unit_path = TRAIN_UNIT_PATH_FORMAT.format(now)
    
    TRAIN_PREPARE_PATH_FORMAT = '/tmp/MGOKDVXPAS-{}.json'
    train_prepare_path = TRAIN_PREPARE_PATH_FORMAT.format(now)
    
    train_prepare_data = {
        'train_unit_path':  train_unit_path,
        'summaryonly':      args.summaryonly,
        'mirror_count':     args.mirror_count,
        'epochs':           args.epochs,
    }
    clover.common.write_json(train_prepare_path,train_prepare_data)
    
    proc = subprocess.Popen([
            sys.executable,
            '-m','kirarafantasia_bot.image_recognition.state._train_prepare',
            train_prepare_path,
        ],
        stderr=None,
        stdout=None
    )
    ret_code = proc.wait()
    if ret_code != 0:
        print('_train_prepare quit with err code: {}'.format(ret_code))
        exit(ret_code)
        
    if args.summaryonly:
        quit()
    
    for mirror_idx in range(args.mirror_count):
        proc = subprocess.Popen([
                sys.executable,
                '-m','kirarafantasia_bot.image_recognition.state._train_unit',
                train_unit_path,
                str(mirror_idx)
            ],
            stderr=None,
            stdout=None
        )
        ret_code = proc.wait()
        if ret_code != 0:
            print('_train_unit quit with err code: {}'.format(ret_code))
            exit(ret_code)
        time.sleep(1)
        
    proc = subprocess.Popen([
            sys.executable,
            '-m','kirarafantasia_bot.image_recognition.state._train_test',
            train_unit_path,
        ],
        stderr=None,
        stdout=None
    )
    ret_code = proc.wait()
    if ret_code != 0:
        print('_train_test quit with err code: {}'.format(ret_code))
        exit(ret_code)
