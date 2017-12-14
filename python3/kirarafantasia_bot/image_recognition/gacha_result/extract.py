import cv2
import glob
import os
import shutil
from . import setting

BOUND_BOX_XYWH_LIST = setting.BOUND_BOX_XYWH_LIST

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='extract board animal')
    parser.add_argument('img_ts', help='img_ts')
    args = parser.parse_args()
    
    fn_list = glob.glob(os.path.join('image_recognition','screen_sample','*','*','{}.png'.format(args.img_ts)))
    assert(len(fn_list)==1)
    fn = fn_list[0]
    
    shutil.rmtree('output',ignore_errors=True)
    os.makedirs('output')

    img = cv2.imread(fn)
    
    for i in range(len(BOUND_BOX_XYWH_LIST)):
        x,y,w,h = BOUND_BOX_XYWH_LIST[i]
        xw = x+w
        yh = y+h
        fn = '%s-%02d.png'%(args.img_ts,i)
        fn = os.path.join('output',fn)
        img0 = img[y:yh,x:xw,:]
        cv2.imwrite(fn,img0)
