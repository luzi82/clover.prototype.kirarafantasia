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

INPUT_WH = model_setting.WIDTH,model_setting.HEIGHT

SCAN_CROP_XYWH   = setting.SCAN_CROP_XYWH
SCAN_BOUND_LU_WH = setting.SCAN_BOUND_LU_WH
SCAN_STEP_XY     = setting.SCAN_STEP_XY

BOUND_BOX_LIST = setting.BOUND_BOX_LIST

class Classifier:

    def __init__(self, model_path):
        data_path   = os.path.join(model_path, DATA_FILENAME)
        with open(data_path,'r') as fin:
            self.data = json.load(fin)
        
        self.mirror_count = self.data['mirror_count']
        
        self.model_list = []
        for mirror_idx in range(self.mirror_count):
            weight_path = os.path.join(model_path, WEIGHT_FILENAME_FORMAT.format(mirror_idx))
            model = model_setting.create_model()
            model.load_weights(weight_path)
            self.model_list.append(model)

    def get(self, img):
        img_list = clover.image_recognition.create_bound_box_img_list(
            img,BOUND_BOX_LIST,INPUT_WH
        )
        img_list = [ model_setting.preprocess_img(img) for img in img_list ]
        img_list = np.asarray(img_list)
        score_list_list = np.zeros((self.mirror_count,len(BOUND_BOX_LIST)),np.float32)
        for mirror_idx in range(self.mirror_count):
            predict_list_list = self.model_list[mirror_idx].predict(img_list)
            #assert(predict_list_list.shape==(len(BOUND_BOX_LIST),1))
            predict_list = np.reshape(predict_list_list,(len(BOUND_BOX_LIST),))
            score_list_list[mirror_idx] = predict_list

        score_min_list = np.amin(score_list_list,axis=0)
        #assert(score_min_list.shape==(len(BOUND_BOX_LIST),))
        score_max_list = np.amax(score_list_list,axis=0)
        #assert(score_max_list.shape==(len(BOUND_BOX_LIST),))
        score_diff_list = score_max_list - score_min_list
        score_diff_max  = np.amax(score_diff_list)

        score_avg_list = np.average(score_list_list,axis=0)
        #assert(score_avg_list.shape==(len(BOUND_BOX_LIST),))
        score = np.amax(score_avg_list)
        bb_idx = np.argmax(score_avg_list)

        return BOUND_BOX_LIST[bb_idx], score, score_diff_max

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='state classifier')
    parser.add_argument('img_file', help="img_file")
    args = parser.parse_args()
    
    img = clover.image_recognition.load_img(args.img_file)

    sc = Classifier(MODEL_PATH)

    ret = sc.get(img)
    print('bound_box={}, score={}, diff_max={}'.format(*ret))
