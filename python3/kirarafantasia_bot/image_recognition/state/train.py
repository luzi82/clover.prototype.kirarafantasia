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
import math
import time
import subprocess

#WIDTH  = model_setting.WIDTH
#HEIGHT = model_setting.HEIGHT

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='state classifier trainer')
    parser.add_argument('epochs', nargs='?', type=int, help="epochs")
    #parser.add_argument('batch_size', nargs='?', type=int, help="batch_size")
    parser.add_argument('mirror_count', nargs='?', type=int, help="mirror count")
    parser.add_argument('--summaryonly', action='store_true', help="summary only")
    args = parser.parse_args()

    assert(((args.epochs!=None)and(args.mirror_count!=None))or(args.summaryonly))

    # get label list
    label_state_path = os.path.join('image_recognition','label','state')
    label_name_list = os.listdir(label_state_path)
    label_name_list = filter(lambda v:os.path.isfile(os.path.join(label_state_path,v)),label_name_list)
    label_name_list = filter(lambda v:v.endswith('.txt'),label_name_list)
    label_name_list = [ i[:-4] for i in label_name_list]
    label_name_list = sorted(label_name_list)
    
    label_count = len(label_name_list)

    # fast quit if summaryonly
    if args.summaryonly:
        from . import model as model_setting
        model = model_setting.create_model(label_count)
        model.summary()
        quit()

    # load sample
    sample_list = []
    for label_idx in range(label_count):
        label_name = label_name_list[label_idx]
        img_fn_list_fn = os.path.join(label_state_path,'{}.txt'.format(label_name))
        with open(img_fn_list_fn, mode='rt', encoding='utf-8') as fin:
            img_fn_list = fin.readlines()
        img_fn_list = [ img_fn.strip() for img_fn in img_fn_list ]
        sample_list += [{'fn':img_fn, 'label_idx':label_idx, 'label_name': label_name} for img_fn in img_fn_list ]

    # randomize sample order
    random.shuffle(sample_list)

    # clean dir
    clover.common.reset_dir(os.path.join('image_recognition','model','state'))

    # write data
    j = {
        'label_name_list': label_name_list,
        'mirror_count':    args.mirror_count
    }
    with open(os.path.join('image_recognition','model','state','data.json'),'w') as fout:
        json.dump(j, fp=fout, indent=2, sort_keys=True)
        fout.write('\n')

    test_sample_count       = int(len(sample_list)/10)
    test_sample_list        = sample_list[:test_sample_count]
    train_valid_sample_list = sample_list[test_sample_count:]

    train_unit_data = {
        'label_count':          label_count,
        'sample_list':          sample_list,
        'test_sample_count':    test_sample_count,
        'epochs':               args.epochs,
        'mirror_count':         args.mirror_count,
        'mirror_data_list':     [],
    }

    for mirror_idx in range(args.mirror_count):

        hdf5_fn = os.path.join('image_recognition','model','state','weight.{}.hdf5'.format(mirror_idx))

        valid_start = int(len(train_valid_sample_list)*(mirror_idx+0)/float(args.mirror_count))
        valid_end   = int(len(train_valid_sample_list)*(mirror_idx+1)/float(args.mirror_count))
        
        mirror_data = {
            'hdf5_fn': hdf5_fn,
            'valid_start': valid_start,
            'valid_end': valid_end,
        }
        train_unit_data['mirror_data_list'].append(mirror_data)
        
#        train_sample_list = train_valid_sample_list[:valid_start]+train_valid_sample_list[valid_end:]
#        valid_sample_list = train_valid_sample_list[valid_start:valid_end]
#        
#        random.shuffle(train_sample_list)
#
#        # create model
#        model = model_setting.create_model(label_count)
#        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#            
#        train_img_list, train_label_onehot_list = sample_list_to_data_set(train_sample_list,label_count)
#        valid_img_list, valid_label_onehot_list = sample_list_to_data_set(valid_sample_list,label_count)
#    
#        checkpointer = ModelCheckpoint(filepath=hdf5_fn, verbose=1, save_best_only=True)
#        
#        train_turn_count = math.floor(len(train_sample_list)**(1/3))
#        batch_size = math.ceil(len(train_sample_list)/train_turn_count)
#        
#        epochs = args.epochs
#        model.fit(train_img_list, train_label_onehot_list,
#            validation_data=(valid_img_list, valid_label_onehot_list),
#            epochs=epochs, batch_size=batch_size, callbacks=[checkpointer], verbose=1)

    TRAIN_UNIT_PATH_FORMAT = '/tmp/IWNFWFKRMF-{}.json'
    train_unit_path = TRAIN_UNIT_PATH_FORMAT.format(int(time.time()))
    clover.common.write_json(train_unit_path,train_unit_data)
    
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
