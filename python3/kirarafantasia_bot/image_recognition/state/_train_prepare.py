import os
import json
import sys
import random
import cv2
import numpy as np
import json
import clover.common
import clover.image_recognition
import math
import time
import subprocess
import gc
from . import setting
from . import train
import kirarafantasia_bot.image_recognition.state as ir_state

save_data_set = train.save_data_set
sample_list_to_data_set = train.sample_list_to_data_set

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='state classifier trainer')
    parser.add_argument('train_prepare_path', help="train_prepare_path")
    args = parser.parse_args()
    
    train_prepare_path = args.train_prepare_path
    train_prepare_data = clover.common.read_json(train_prepare_path)

    train_unit_path = train_prepare_data['train_unit_path']
    summaryonly     = train_prepare_data['summaryonly']
    mirror_count    = train_prepare_data['mirror_count']
    epochs          = train_prepare_data['epochs']

    # get label list
    label_name_list = ir_state.get_label_list()
    label_count = len(label_name_list)

    # fast quit if summaryonly
    if summaryonly:
        from . import model as model_setting
        model = model_setting.create_model(label_count)
        model.summary()
        quit()

    # clear session
    SESSION_CACHE_DIR_PATH = os.path.join('cache','session','state.train')
    clover.common.reset_dir(SESSION_CACHE_DIR_PATH)

    # load sample list
    sample_list = []
    for label_idx in range(label_count):
        label_name = label_name_list[label_idx]
        img_fn_list_fn = os.path.join(ir_state.LABEL_STATE_PATH,'{}.txt'.format(label_name))
        with open(img_fn_list_fn, mode='rt', encoding='utf-8') as fin:
            img_fn_list = fin.readlines()
        img_fn_list = [ img_fn.strip() for img_fn in img_fn_list ]
        sample_list += [{'fn':img_fn, 'label_idx':label_idx, 'label_name': label_name} for img_fn in img_fn_list ]

    # randomize sample order
    random.shuffle(sample_list)
    #sample_list = sample_list[:100]

    # clean dir
    clover.common.reset_dir(os.path.join('image_recognition','model','state'))

    # write data
    j = {
        'label_name_list': label_name_list,
        'mirror_count':    mirror_count
    }
    with open(os.path.join('image_recognition','model','state','data.json'),'w') as fout:
        json.dump(j, fp=fout, indent=2, sort_keys=True)
        fout.write('\n')

    test_sample_count        = int(len(sample_list)/10)
    test_sample_list         = sample_list[:test_sample_count]
    train_valid_sample_list  = sample_list[test_sample_count:]
    train_valid_sample_count = len(train_valid_sample_list)

    # load data set, save to cache
    img_list, label_onehot_list = sample_list_to_data_set(train_valid_sample_list,label_count)
    train_valid_sample_data_dir_path = os.path.join(SESSION_CACHE_DIR_PATH,'train_valid_sample_data')
    save_data_set(train_valid_sample_data_dir_path, img_list, label_onehot_list)
    del img_list, label_onehot_list

    train_unit_data = {
        'label_count':                      label_count,
        'sample_list':                      sample_list,
        'train_valid_sample_data_dir_path': train_valid_sample_data_dir_path,
        'test_sample_count':                test_sample_count,
        'epochs':                           epochs,
        'mirror_count':                     mirror_count,
        'mirror_data_list':                 [],
    }

    for mirror_idx in range(mirror_count):

        hdf5_fn = os.path.join('image_recognition','model','state','weight.{}.hdf5'.format(mirror_idx))

        valid_start = int(train_valid_sample_count*(mirror_idx+0)/float(mirror_count))
        valid_end   = int(train_valid_sample_count*(mirror_idx+1)/float(mirror_count))
        
        mirror_data = {
            'hdf5_fn': hdf5_fn,
            'valid_start': valid_start,
            'valid_end': valid_end,
        }
        train_unit_data['mirror_data_list'].append(mirror_data)

    clover.common.write_json(train_unit_path,train_unit_data)
