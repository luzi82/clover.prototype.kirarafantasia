import os
import clover.common

def get_label_list():
    label_state_path = os.path.join('image_recognition','label','state')
    label_name_list = os.listdir(label_state_path)
    label_name_list = filter(lambda v:os.path.isfile(os.path.join(label_state_path,v)),label_name_list)
    label_name_list = filter(lambda v:v.endswith('.txt'),label_name_list)
    label_name_list = [ i[:-4] for i in label_name_list]
    label_name_list = sorted(label_name_list)
    return label_name_list

def get_known_state(fn):
    for state in get_label_list():
        state_fn = os.path.join('image_recognition','label','state','{}.txt'.format(state))
        img_fn_list = clover.common.readlines(state_fn)
        if fn in img_fn_list:
            return state
    return None
