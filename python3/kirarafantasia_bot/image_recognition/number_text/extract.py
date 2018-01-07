import cv2
import glob
import os
import shutil
import clover.common
from kirarafantasia_bot.image_recognition.state import get_known_state

def get_state_num_box_data():
    state_num_box_data = os.path.dirname(__file__)
    state_num_box_data = os.path.join(state_num_box_data,'state_num_box.json')
    state_num_box_data = clover.common.read_json(state_num_box_data)
    return state_num_box_data

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='extract num img')
    parser.add_argument('img_ts', help='img_ts')
    parser.add_argument('--no_clear', action='store_true', help="no clear output dir")
    args = parser.parse_args()
    
    fn_list = glob.glob(os.path.join('image_recognition','screen_sample','*','*','{}.png'.format(args.img_ts)))
    assert(len(fn_list)==1)
    fn = fn_list[0]
    
    if not args.no_clear:
        shutil.rmtree('output',ignore_errors=True)
    clover.common.makedirs('output')
    
    state_num_box_data = get_state_num_box_data()
    state = get_known_state(fn)
    assert(state in state_num_box_data['state_data_dict'])
    state_data = state_num_box_data['state_data_dict'][state]

    img = cv2.imread(fn)
    
    for box_data in state_data['box_data_list']:
        x,y,w,h = box_data['x'],box_data['y'],box_data['w'],box_data['h']
        xw = x+w
        yh = y+h
        fn = '%s.%d.%d.%d.%d.png'%(args.img_ts,x,y,xw,yh)
        fn = os.path.join('output',fn)
        img0 = img[y:yh,x:xw,:]
        cv2.imwrite(fn,img0)
