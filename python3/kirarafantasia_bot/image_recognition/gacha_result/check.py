import os
import clover.common
import clover.image_recognition
from . import setting
import functools
import cv2

# output image label group setting
#   output/[label]/[img]

NAME = setting.NAME
BOUND_BOX_XYWH_LIST = setting.BOUND_BOX_XYWH_LIST

@functools.lru_cache()
def load_img(fn):
    return cv2.imread(fn)

if __name__ == '__main__':
    import argparse
    import shutil

    parser = argparse.ArgumentParser(description='state')
    parser.add_argument('label', nargs='?', help='label')
    args = parser.parse_args()

    csv_path = os.path.join('image_recognition','label','gacha_result','sample_list.csv')
    sample_list = clover.common.read_csv(csv_path)

    if args.label != None:
        label_list = [args.label]
    else:
        label_list = list(set([i['label'] for i in sample_list]))

    clover.common.reset_dir('output')

    for label in label_list:
        clover.common.makedirs(os.path.join('output',label))

    for sample in sample_list:
        if sample['label'] not in label_list:
            continue
        _, t = os.path.split(sample['fn'])
        t = t[:-4]
        
        idx = int(sample['idx'])
        x,y,w,h = BOUND_BOX_XYWH_LIST[idx]
        xw,yh = x+w,y+h

        img = load_img(sample['fn'])
        img = img[y:yh,x:xw,:]
        
        fn = os.path.join('output',sample['label'],'{0}-{1:02d}.png'.format(t,idx))
        cv2.imwrite(fn,img)
