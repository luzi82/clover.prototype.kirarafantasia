import os
import json
import cv2
import numpy as np
from functools import lru_cache
import clover.image_recognition
from . import setting

MODEL_PATH = os.path.join('image_recognition','model','state')
WEIGHT_FILENAME_FORMAT = 'weight.{}.hdf5'
DATA_FILENAME   = 'data.json'

IGNORE_STDDEV_MAX = 0.2
IGNORE_DIFF_MAX   = 1

from . import model

WIDTH  = model.WIDTH
HEIGHT = model.HEIGHT

load_img = clover.image_recognition.load_img

def preprocess_img(img):
    img = cv2.resize(img,dsize=(WIDTH,HEIGHT),interpolation=cv2.INTER_AREA)
    img = np.append(img,clover.image_recognition.xy_layer(WIDTH,HEIGHT),axis=2)
    return img

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
            model = model.create_model(self.label_name_count)
            model.load_weights(weight_path)
            self.model_list.append(model)

    def get_state(self, img):
        if should_ignore(img):
            return '_IGNORE', True
        
        img = preprocess_img(img)
        score_list = [0] * self.label_name_count
        for mirror_idx in range(self.mirror_count):
            p_list_list = self.model_list[mirror_idx].predict(np.expand_dims(img, axis=0))
            label_idx = np.argmax(p_list_list,axis=1)[0]
            score_list[label_idx] += 1
        label_idx = np.argmax(score_list)

        label_name = self.data['label_name_list'][label_idx]
        perfect = (score_list[label_idx]==self.mirror_count)
        return label_name, perfect

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='state classifier')
    parser.add_argument('img_file', help="img_file")
    args = parser.parse_args()
    
    img = load_img(args.img_file)

    sc = StateClassifier(MODEL_PATH)

    label, perfect = sc.get_state(img)
    print('label={}, perfect={}'.format(label, perfect))
