import os
import clover.common
import clover.image_recognition

# output image label group setting
#   output/[label]/[img]

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
        with open(os.path.join('image_recognition','label','state','{}.txt'.format(state)),'r') as fin:
            file_list = fin.readlines()
        file_list = [ i.strip() for i in file_list ]
        for f in file_list:
            _, t = os.path.split(f)
            clover.common.makedirs(os.path.join('output',state))
            shutil.copyfile( f, os.path.join('output',state,t) )
