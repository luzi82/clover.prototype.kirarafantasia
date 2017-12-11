from clover.video_input import video_capture
import random
import os
import time
import sys
import cv2

FFMPEG_EXEC_PATH = os.path.join('dependency','FFmpeg','ffmpeg')
MAIN_WIDTH = 213
MAIN_HEIGHT = 120

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='video capture')
    parser.add_argument('src_name', help='src_name')
    #parser.add_argument('output_folder', help='output_folder')
    
    args = parser.parse_args()

    output_folder = os.path.join('resource_set','screen_sample_set')

    t = int(time.time()*1000)
    output_folder = os.path.join(output_folder,str(t))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vc = video_capture.VideoCapture(FFMPEG_EXEC_PATH,args.src_name,MAIN_WIDTH,MAIN_HEIGHT)
    vc.start()
    vc.wait_data_ready()
    while True:
        t = int(time.time()*1000)
        t0 = int(t/100000)
        ndata = vc.get_frame()
        write_ok = True
        if write_ok:
            fn_dir = os.path.join(output_folder,str(t0))
            fn = os.path.join(fn_dir,'{}.png'.format(t))
            if not os.path.isdir(fn_dir):
                os.makedirs(fn_dir)
            print(fn,file=sys.stderr)
            cv2.imwrite(fn,ndata)
        vc.release_frame()
        time.sleep(0.05+0.05*random.random())
    vc.close()
