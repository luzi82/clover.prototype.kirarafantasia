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

BOUND_BOX_LIST = setting.BOUND_BOX_XYWH_LIST

class Classifier:

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
        img_list = clover.image_recognition.create_bound_box_img_list(
            img,BOUND_BOX_LIST,INPUT_WH
        )
        img_list = [ model_setting.preprocess_img(img) for img in img_list ]
        img_list = np.asarray(img_list)
        score_list_list_list = np.zeros((self.mirror_count,len(BOUND_BOX_LIST),self.label_name_count),np.float32)
        for mirror_idx in range(self.mirror_count):
            predict_list_list = self.model_list[mirror_idx].predict(img_list)
            assert(predict_list_list.shape==(len(BOUND_BOX_LIST),self.label_name_count))
            score_list_list_list[mirror_idx] = predict_list_list

        # cal disagree
        label_idx_list_list = np.argmax(score_list_list_list, axis=2)
        assert(label_idx_list_list.shape==(self.mirror_count,len(BOUND_BOX_LIST)))
        label_idx_ptp_list = np.ptp(label_idx_list_list,axis=0)
        assert(label_idx_ptp_list.shape==(len(BOUND_BOX_LIST),))
        label_idx_ptp_max  = np.amax(label_idx_ptp_list)
        
        # cal label
        score_list_list = np.average(score_list_list_list, axis=0)
        assert(score_list_list.shape==(len(BOUND_BOX_LIST),self.label_name_count))
        label_idx_list  = np.argmax(score_list_list, axis=1)
        assert(label_idx_list.shape==(len(BOUND_BOX_LIST),))
        label_list = [self.data['label_name_list'][i] for i in label_idx_list]
        
        # cal score
        score_list = np.max(score_list_list, axis=1)

        return label_list, score_list, label_idx_ptp_max<=0, label_idx_ptp_list

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='state classifier')
    parser.add_argument('img_file', help="img_file")
    args = parser.parse_args()
    
    img = clover.image_recognition.load_img(args.img_file)

    sc = Classifier(MODEL_PATH)

    ret = sc.get(img)
    print('class={}, score_list={}, perfect={}, diff={}'.format(*ret))
