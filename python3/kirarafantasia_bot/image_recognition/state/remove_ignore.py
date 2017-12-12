import os
import clover.common
import clover.image_recognition
from . import setting

# remove all state input sample which should be ignored

def should_not_ignore(filename):
    img = clover.image_recognition.load_img(filename)
    return not setting.should_ignore(img)

if __name__ == '__main__':
    import argparse
    import shutil

    parser = argparse.ArgumentParser(description='state')
    parser.add_argument('state', nargs='?', help='state')
    args = parser.parse_args()

    if args.state != None:
        state_list = [args.state]
    else:
        state_list = clover.image_recognition.get_label_state_list()

    clover.common.reset_dir('output')
    for state in state_list:
        filename = os.path.join('image_recognition','label','state','{}.txt'.format(state))
        file_list = clover.common.readlines(filename)
        file_list = list(filter(should_not_ignore,file_list))
        clover.common.writelines(filename, file_list)
