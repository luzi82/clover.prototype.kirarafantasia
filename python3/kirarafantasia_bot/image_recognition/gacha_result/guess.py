import glob
import os
import sys
import shutil
from . import classifier
import clover.common
import clover.image_recognition
from . import setting
import cv2

# output image label group guess
#   output/[guess-label]/[img]

MODEL_PATH = classifier.MODEL_PATH
BOUND_BOX_XYWH_LIST = setting.BOUND_BOX_XYWH_LIST

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='state label util')
    parser.add_argument('--non_perfect', action='store_true', help="output only not perfect")
    args = parser.parse_args()
    
    clover.common.reset_dir('output')
    
    clr = classifier.Classifier(MODEL_PATH)

    fn_list = glob.glob(os.path.join('input','*.png'))

    for img_fn in fn_list:
        _, img_fn_t = os.path.split(img_fn)
        img_fn_t = img_fn_t[:-4]
        img_ori = cv2.imread(img_fn)
        img = clover.image_recognition.load_img(img_fn)
        label_list, _, _, non_perfect = clr.get(img)
        
        for i in range(len(BOUND_BOX_XYWH_LIST)):
            if(args.non_perfect) and (non_perfect[i]<=0):
                continue
            label = label_list[i]
            out_fn_dir = os.path.join('output',label)
            out_fn = os.path.join(out_fn_dir,'{0}-{1:02d}.png'.format(img_fn_t,i))
            clover.common.makedirs(out_fn_dir)
            x,y,xw,yh = clover.image_recognition.xywh_to_xyxy(BOUND_BOX_XYWH_LIST[i])
            img0 = img_ori[y:yh,x:xw,:]
            cv2.imwrite(out_fn,img0)
