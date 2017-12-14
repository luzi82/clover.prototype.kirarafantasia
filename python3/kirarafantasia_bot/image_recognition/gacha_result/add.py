import glob
import os
import sys
import json
import clover.common
import clover.image_recognition

CSV_COL_LIST = ['fn','idx','label']

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='state label util')
    parser.add_argument('--reset', action='store_true', help='clear and reset')
    args = parser.parse_args()

    csv_dir  = os.path.join('image_recognition','label','gacha_result')
    clover.common.makedirs(csv_dir)

    csv_path = os.path.join(csv_dir,'sample_list.csv')
    if (not args.reset) and os.path.isfile(csv_path):
        tmp = clover.common.read_csv(csv_path)
        entry_dict = { (i['fn'],int(i['idx'])):i for i in tmp }
    else:
        entry_dict = {}

    for label_name in os.listdir('input'):
        label_path = os.path.join('input',label_name)
        if not os.path.isdir(label_path):
            continue
        #data_dict['label_list'].append(label_name)
        for image_fn in os.listdir(label_path):
            if not image_fn.endswith('.png'):
                continue
            image_fn = image_fn[:-4]
            image_fn = image_fn.split('-')
            assert(len(image_fn)==2)
            image_idx = int(image_fn[1])
            image_fn  = int(image_fn[0])
            image_timestamp=clover.image_recognition.get_timestamp(image_fn)
            image_ori_path = os.path.join('image_recognition','screen_sample',str(image_timestamp),str(int(image_fn/100000)),'{}.png'.format(image_fn))
            if not os.path.isfile(image_ori_path):
                print('{} not found'.format(image_ori_path),file=sys.stderr)
                continue
            key = (image_ori_path,image_idx)
            entry_dict[key]={
                'fn':image_ori_path,
                'idx':'{0:02d}'.format(image_idx),
                'label':label_name
            }

    #data_dict['label_list'] = sorted(list(set(data_dict['label_list'])))

    entry_key_list = sorted(entry_dict.keys())
    entry_list = [ entry_dict[i] for i in entry_key_list ]
    clover.common.write_csv(csv_path,entry_list,CSV_COL_LIST)
