import glob
import os
import sys
import json
import clover.common
import clover.image_recognition
from . import setting
import re

CSV_COL_LIST = ['fn','x','y','xw','yh','label']
NAME = setting.NAME

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='state label util')
    parser.add_argument('--reset', action='store_true', help='clear and reset')
    args = parser.parse_args()

    csv_dir  = os.path.join('image_recognition','label',NAME)
    clover.common.makedirs(csv_dir)

    csv_path = os.path.join(csv_dir,'sample_list.csv')
    if (not args.reset) and os.path.isfile(csv_path):
        tmp = clover.common.read_csv(csv_path)
        entry_dict = { (i['fn'],int(i['x']),int(i['y']),int(i['xw']),int(i['yh'])):i for i in tmp }
    else:
        entry_dict = {}

    fn_prog = re.compile('(\\d+)\.(\\d+)\.(\\d+)\.(\\d+)\.(\\d+)\.(.*)\.png')

    for fn in os.listdir('input'):
        mm = fn_prog.match(fn)
        if mm is None:
            print('pattern not match: {}'.format(fn))
        fidx = int(mm.group(1))
        x = int(mm.group(2))
        y = int(mm.group(3))
        xw = int(mm.group(4))
        yh = int(mm.group(5))
        label = mm.group(6)
        label = str.replace(label,'_','/')
        image_timestamp=clover.image_recognition.get_timestamp(fidx)
        image_ori_path = os.path.join('image_recognition','screen_sample',str(image_timestamp),str(int(fidx/100000)),'{}.png'.format(fidx))
        if not os.path.isfile(image_ori_path):
            print('{} not found'.format(image_ori_path),file=sys.stderr)
            continue
        key = (image_ori_path,x,y,xw,yh)
        entry_dict[key]={
            'fn':image_ori_path,
            'x':x,'y':y,'xw':xw,'yh':yh,
            'label':label
        }

    entry_key_list = sorted(entry_dict.keys())
    entry_list = [ entry_dict[i] for i in entry_key_list ]
    clover.common.write_csv(csv_path,entry_list,CSV_COL_LIST)
