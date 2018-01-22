import os
import json
import cv2
import numpy as np
from functools import lru_cache
import clover.image_recognition
from . import setting

MODEL_PATH = os.path.join('image_recognition','model',setting.NAME)
WEIGHT_FILENAME_FORMAT = 'weight.{}.hdf5'
DATA_FILENAME   = 'data.json'

from . import model as model_setting

WIDTH  = model_setting.WIDTH
HEIGHT = model_setting.HEIGHT

load_img = clover.image_recognition.load_img
preprocess_img = setting.preprocess_img
should_ignore = setting.should_ignore

class StateClassifier:

    def __init__(self, model_path):
        data_path   = os.path.join(model_path, DATA_FILENAME)
        with open(data_path,'r') as fin:
            self.data = json.load(fin)
        
        self.mirror_count = self.data['mirror_count']
        self.label_name_count = len(self.data['label_name_list'])
        
        self.model_list = []
        for mirror_idx in range(self.mirror_count):
            weight_path = os.path.join(model_path, WEIGHT_FILENAME_FORMAT.format(mirror_idx))
            model = model_setting.create_model(self.label_name_count)
            model.load_weights(weight_path)
            self.model_list.append(model)

    def get(self, img):
        assert(np.amax(img)<=1)
        assert(np.amin(img)>=-1)
    
        if should_ignore(img):
            return '_IGNORE', 1, True
        
        img = preprocess_img(img)
        score_list_list_list = np.zeros((self.mirror_count,1,self.label_name_count))
        for mirror_idx in range(self.mirror_count):
            p_list_list = self.model_list[mirror_idx].predict(np.expand_dims(img, axis=0))
            assert(p_list_list.shape==(1,self.label_name_count))
            score_list_list_list[mirror_idx] = p_list_list

        # cal disagree
        label_idx_list_list = np.argmax(score_list_list_list, axis=2)
        assert(label_idx_list_list.shape==(self.mirror_count,1))
        label_idx_ptp_list = np.ptp(label_idx_list_list,axis=0)
        assert(label_idx_ptp_list.shape==(1,))
        label_idx_ptp_max  = np.amax(label_idx_ptp_list)
        perfect = label_idx_ptp_max<=0
        
        # cal label
        score_list_list = np.average(score_list_list_list, axis=0)
        assert(score_list_list.shape==(1,self.label_name_count))
        label_idx_list  = np.argmax(score_list_list, axis=1)
        assert(label_idx_list.shape==(1,))
        label_list = [self.data['label_name_list'][i] for i in label_idx_list]
        label = label_list[0]
        
        # cal score
        score_list = np.max(score_list_list, axis=1)
        score = score_list[0]
        
        return label, score, perfect

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='state classifier')
    parser.add_argument('img_file', help="img_file")
    args = parser.parse_args()
    
    img = load_img(args.img_file)

    sc = StateClassifier(MODEL_PATH)

    label, score, perfect = sc.get(img)
    print('label={}, score={}, perfect={}'.format(label, score, perfect))
