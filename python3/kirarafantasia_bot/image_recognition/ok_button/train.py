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

NAME = model_setting.NAME

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

    # fast quit if summaryonly
    model = model_setting.create_model()
    model.summary()
    if args.summaryonly:
        quit()

    # load sample
    sample_fn   = 
    sample_list = clover.common.read_csv(os.path.join(
        'image_recognition','label',NAME
    ))
    
    # load img
    img_list, score_list = sample_list_to_data_set(sample_list)
    sample_list = list(zip(img_list,score_list))

    # randomize sample order
    random.shuffle(img_score_list)

    # clean dir
    clover.common.reset_dir(os.path.join('image_recognition','model',NAME))

    # write data
    j = {
        'mirror_count':    args.mirror_count
    }
    with open(os.path.join('image_recognition','model',NAME,'data.json'),'w') as fout:
        json.dump(j, fp=fout, indent=2, sort_keys=True)
        fout.write('\n')

    test_sample_count       = int(len(sample_list)/10)
    test_sample_list        = sample_list[:test_sample_count]
    train_valid_sample_list = sample_list[test_sample_count:]

    for mirror_idx in range(args.mirror_count):
        hdf5_fn = os.path.join('image_recognition','model',NAME,'weight.{}.hdf5'.format(mirror_idx))

        valid_start = int(len(train_valid_sample_list)*(mirror_idx+0)/float(args.mirror_count))
        valid_end   = int(len(train_valid_sample_list)*(mirror_idx+1)/float(args.mirror_count))
        train_sample_list = train_valid_sample_list[:valid_start]+train_valid_sample_list[valid_end:]
        valid_sample_list = train_valid_sample_list[valid_start:valid_end]

        # create model
        model = model_setting.create_model(label_count)
        model.compile(optimizer='adam', loss='mean_squared_error')
            
        # train_img_list, train_label_onehot_list = sample_list_to_data_set(train_sample_list,label_count)
        # valid_img_list, valid_label_onehot_list = sample_list_to_data_set(valid_sample_list,label_count)

        train_img_list, train_score_list = zip(*train_sample_list)
        train_img_list = list(train_img_list)
        train_score_list = list(train_score_list)
        
        valid_img_list, valid_score_list = zip(*valid_sample_list)
        valid_img_list = list(valid_img_list)
        valid_score_list = list(valid_score_list)
    
        checkpointer = ModelCheckpoint(filepath=hdf5_fn, verbose=1, save_best_only=True)
        
        epochs = args.epochs
        model.fit(train_img_list, train_score_list,
            validation_data=(valid_img_list, valid_score_list),
            epochs=epochs, batch_size=args.batch_size, callbacks=[checkpointer], verbose=1)
    
    model = model_setting.create_model(label_count)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    test_img_list,  test_score_list  = zip(*test_sample_list)
    test_img_list = list(test_img_list)
    test_score_list = list(test_score_list)
    for mirror_idx in range(args.mirror_count):
        hdf5_fn = os.path.join('image_recognition','model',NAME,'weight.{}.hdf5'.format(mirror_idx))

        model.load_weights(hdf5_fn)
        
        loss = model.test_on_batch(test_img_list)
        *** NOT FINISH
        test_predictions = [np.argmax(model.predict(np.expand_dims(img_list, axis=0))) for img_list in test_img_list]
        test_accuracy = np.sum(np.array(test_predictions)==np.argmax(test_label_onehot_list, axis=1))/len(test_predictions)
        print('Mirror {0}/{1} test accuracy: {2:.4f}'.format(mirror_idx,args.mirror_count,test_accuracy))
