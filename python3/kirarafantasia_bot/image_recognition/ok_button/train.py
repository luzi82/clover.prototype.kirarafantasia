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
import clover.image_recognition

INPUT_WH = model_setting.WIDTH,model_setting.HEIGHT

TRAIN_CROP_XYWH   = model_setting.TRAIN_CROP_XYWH
TRAIN_BOUND_LU_WH = model_setting.TRAIN_BOUND_LU_WH
TRAIN_STEP_XY     = model_setting.TRAIN_STEP_XY

def sample_list_to_data_set(sample_list):
    bound_box_list = clover.image_recognition.cal_bound_box_list(TRAIN_CROP_XYWH,TRAIN_BOUND_LU_WH,TRAIN_STEP_XY)

    img_list = []
    score_list = []
    
    for sample in sample_list:
        sample_img = clover.image_recognition.load_img(sample['fn'])
        sample_img_list = clover.image_recognition.create_bound_box_img_list(
            img,bound_box_list,INPUT_WH
        )
        
        sample_obj_xywh = (sample['x'],sample['y'],sample['w'],sample['h'])
        sample_score_list = [
            clover.image_recognition.cal_bound_box_score(bound_box,sample_obj_xywh)
            for bound_box in bound_box_list
        ]
        
        assert(len(sample_img_list)==len(sample_score_list))
        img_list  +=sample_img_list
        score_list+=sample_score_list
    
    return img_list, score_list

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='state classifier trainer')
    parser.add_argument('epochs', nargs='?', type=int, help="epochs")
    parser.add_argument('batch_size', nargs='?', type=int, help="batch_size")
    parser.add_argument('mirror_count', nargs='?', type=int, help="batch_size")
    parser.add_argument('--summaryonly', action='store_true', help="summary only")
    args = parser.parse_args()

    assert(((args.epochs!=None)and(args.batch_size!=None)and(args.mirror_count!=None))or(args.summaryonly))

    # get label list
    label_state_path = os.path.join('image_recognition','label','state')
    label_name_list = os.listdir(label_state_path)
    label_name_list = filter(lambda v:os.path.isfile(os.path.join(label_state_path,v)),label_name_list)
    label_name_list = filter(lambda v:v.endswith('.txt'),label_name_list)
    label_name_list = [ i[:-4] for i in label_name_list]
    label_name_list = sorted(label_name_list)
    
    label_count = len(label_name_list)

    # fast quit if summaryonly
    model = model_setting.create_model(label_count)
    model.summary()
    if args.summaryonly:
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

    for mirror_idx in range(args.mirror_count):
        hdf5_fn = os.path.join('image_recognition','model','state','weight.{}.hdf5'.format(mirror_idx))

        valid_start = int(len(train_valid_sample_list)*(mirror_idx+0)/float(args.mirror_count))
        valid_end   = int(len(train_valid_sample_list)*(mirror_idx+1)/float(args.mirror_count))
        train_sample_list = train_valid_sample_list[:valid_start]+train_valid_sample_list[valid_end:]
        valid_sample_list = train_valid_sample_list[valid_start:valid_end]

        # create model
        model = model_setting.create_model(label_count)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
            
        train_img_list, train_label_onehot_list = sample_list_to_data_set(train_sample_list,label_count)
        valid_img_list, valid_label_onehot_list = sample_list_to_data_set(valid_sample_list,label_count)
    
        checkpointer = ModelCheckpoint(filepath=hdf5_fn, verbose=1, save_best_only=True)
        
        epochs = args.epochs
        model.fit(train_img_list, train_label_onehot_list,
            validation_data=(valid_img_list, valid_label_onehot_list),
            epochs=epochs, batch_size=args.batch_size, callbacks=[checkpointer], verbose=1)
    
    model = model_setting.create_model(label_count)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    for mirror_idx in range(args.mirror_count):
        hdf5_fn = os.path.join('image_recognition','model','state','weight.{}.hdf5'.format(mirror_idx))

        model.load_weights(hdf5_fn)
    
        test_img_list,  test_label_onehot_list  = sample_list_to_data_set(test_sample_list ,label_count)
        test_predictions = [np.argmax(model.predict(np.expand_dims(img_list, axis=0))) for img_list in test_img_list]
        test_accuracy = np.sum(np.array(test_predictions)==np.argmax(test_label_onehot_list, axis=1))/len(test_predictions)
        print('Mirror {0}/{1} test accuracy: {2:.4f}'.format(mirror_idx,args.mirror_count,test_accuracy))
