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
import clover.common
import clover.image_recognition
from . import setting

NAME = setting.NAME

INPUT_WH = model_setting.WIDTH,model_setting.HEIGHT

BOUND_BOX_XYWH_LIST = setting.BOUND_BOX_XYWH_LIST

# return: [img], [one-hot]
def sample_list_to_data_set(sample_list, label_list):
    bound_box_list = BOUND_BOX_XYWH_LIST
    
    fn_idx_label_list_dict = {} # {fn:[(idx,label)]
    for sample in sample_list:
        fn    = sample['fn']
        idx   = int(sample['idx'])
        label = label_list.index(sample['label'])
        if fn not in fn_idx_label_list_dict:
            fn_idx_label_list_dict[fn] = []
        fn_idx_label_list_dict[fn].append((idx,label))

    img_list = []
    onehot_list = np.zeros((0,len(label_list)),np.float32)
    
    for fn, idx_label_list in fn_idx_label_list_dict.items():
        sample_img = clover.image_recognition.load_img(fn)
        sample_img_list = clover.image_recognition.create_bound_box_img_list(
            sample_img,bound_box_list,INPUT_WH
        )
        sample_img_list = [ sample_img_list[ idx_label[0] ] for idx_label in idx_label_list ]
        
        v_label_list = [i[1] for i in idx_label_list]
        v_onehot_list  = np_utils.to_categorical(v_label_list, len(label_list))
        
        assert(len(sample_img_list)==len(idx_label_list))
        #assert(len(sample_img_list)==10) # optional
        assert(sample_img_list[0].shape == INPUT_WH+(3,))
        assert(v_onehot_list.shape==(len(idx_label_list),len(label_list)))
        
        img_list   +=sample_img_list
        onehot_list=np.append(onehot_list,v_onehot_list,axis=0)

    assert(len(img_list)==len(sample_list))
    assert(onehot_list.shape==(len(sample_list),len(label_list)))
    
    return img_list, onehot_list

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='state classifier trainer')
    parser.add_argument('epochs', nargs='?', type=int, help="epochs")
    parser.add_argument('batch_size', nargs='?', type=int, help="batch_size")
    parser.add_argument('mirror_count', nargs='?', type=int, help="batch_size")
    parser.add_argument('--summaryonly', action='store_true', help="summary only")
    args = parser.parse_args()

    assert(((args.epochs!=None)and(args.batch_size!=None)and(args.mirror_count!=None))or(args.summaryonly))

    sample_list = clover.common.read_csv(os.path.join(
        'image_recognition','label',NAME,'sample_list.csv'
    ))

    label_name_list = [i['label'] for i in sample_list]
    label_name_list = list(sorted(list(set(label_name_list))))
    label_count = len(label_name_list)

    # fast quit if summaryonly
    model = model_setting.create_model(label_count)
    model.summary()
    if args.summaryonly:
        quit()

    # load img
    img_list, onehot_list = sample_list_to_data_set(sample_list,label_name_list)
    img_list = [ model_setting.preprocess_img(img) for img in img_list ]
    sample_list = list(zip(img_list,onehot_list))

    # randomize sample order
    random.shuffle(sample_list)

    # clean dir
    clover.common.reset_dir(os.path.join('image_recognition','model',NAME))

    # write data
    j = {
        'label_name_list': label_name_list,
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
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
            
        # train_img_list, train_label_onehot_list = sample_list_to_data_set(train_sample_list,label_count)
        # valid_img_list, valid_label_onehot_list = sample_list_to_data_set(valid_sample_list,label_count)

        train_img_list, train_onehot_list = zip(*train_sample_list)
        train_img_list    = np.asarray(list(train_img_list),np.float32)
        train_onehot_list = np.asarray(list(train_onehot_list),np.float32)
        
        valid_img_list, valid_onehot_list = zip(*valid_sample_list)
        valid_img_list    = np.asarray(list(valid_img_list),np.float32)
        valid_onehot_list = np.asarray(list(valid_onehot_list),np.float32)
    
        checkpointer = ModelCheckpoint(filepath=hdf5_fn, verbose=1, save_best_only=True)
        
        epochs = args.epochs
        model.fit(train_img_list, train_onehot_list,
            validation_data=(valid_img_list, valid_onehot_list),
            epochs=epochs, batch_size=args.batch_size, callbacks=[checkpointer], verbose=1)
    
    test_img_list,  test_onehot_list  = zip(*test_sample_list)
    test_img_list = np.asarray(list(test_img_list),np.float32)
    test_onehot_list = np.asarray(list(test_onehot_list),np.float32)
    for mirror_idx in range(args.mirror_count):
        model = model_setting.create_model(label_count)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        hdf5_fn = os.path.join('image_recognition','model',NAME,'weight.{}.hdf5'.format(mirror_idx))
        model.load_weights(hdf5_fn)
        
        test_predictions = [np.argmax(model.predict(np.expand_dims(test_img, axis=0))) for test_img in test_img_list]
        test_accuracy = np.sum(np.array(test_predictions)==np.argmax(test_onehot_list, axis=1))/len(test_predictions)
        print('Mirror {0}/{1} test accuracy: {2:.4f}'.format(mirror_idx,args.mirror_count,test_accuracy))
