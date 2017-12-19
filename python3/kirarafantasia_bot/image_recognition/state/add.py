import glob
import os
import sys
import clover.common
import clover.image_recognition

# struct:
#   input/[label]/[img]

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='state label util')
    parser.add_argument('--reset', action='store_true', help='clear and reset')
    args = parser.parse_args()

    INPUT_DIR = 'input'

    # get label list
    label_state_path = os.path.join('image_recognition','label','state')
    label_name_list = os.listdir(label_state_path)
    label_name_list = filter(lambda v:os.path.isfile(os.path.join(label_state_path,v)),label_name_list)
    label_name_list = filter(lambda v:v.endswith('.txt'),label_name_list)
    label_name_list = [ i[:-4] for i in label_name_list]
    label_name_list = set(label_name_list)
    label_name_list = label_name_list | set([i for i in os.listdir(INPUT_DIR) if os.path.isdir(i)])
    label_name_list = sorted(list(label_name_list))
    
    # read old label
    fn_label_dict = {}
    if not args.reset:
        for label_name in label_name_list:
            label_txt_path = os.path.join('image_recognition','label','state','{}.txt'.format(label_name))
            if not os.path.isfile(label_txt_path):
                continue
            fn_list = clover.common.readlines(label_txt_path)
            for fn in fn_list:
                if not os.path.isfile(fn):
                    continue
                fn_label_dict[fn] = label_name

    # read dir
    for label_name in os.listdir(INPUT_DIR):
        label_path = os.path.join(INPUT_DIR,label_name)
        if not os.path.isdir(label_path):
            continue
        for image_fn in os.listdir(label_path):
            if not image_fn.endswith('.png'):
                continue
            image_idx = image_fn[:-4]
            image_idx = int(image_idx)
            image_timestamp=clover.image_recognition.get_timestamp(image_idx)
            image_ori_path = os.path.join('image_recognition','screen_sample',str(image_timestamp),str(int(image_idx/100000)),'{}.png'.format(image_idx))
            if not os.path.isfile(image_ori_path):
                print('{} not found'.format(image_ori_path),file=sys.stderr)
                continue
            fn_label_dict[image_ori_path] = label_name

    # output txt file
    for label_name in label_name_list:
        path_list = []
        for fn, label in fn_label_dict.items():
            if label != label_name:
                continue
            path_list.append(fn)
        path_list = sorted(path_list)

        label_txt_path = os.path.join('image_recognition','label','state','{}.txt'.format(label_name))
        clover.common.makedirs(os.path.join('image_recognition','label','state'))
        clover.common.writelines(label_txt_path,path_list)
