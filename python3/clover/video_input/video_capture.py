import subprocess
import os
import re
import threading
import copy
import sys
import time
import numpy as np
import cv2
from clover.common import async_read_write_judge

BUFFER_COUNT = 3

class VideoCapture:

    def __init__(self,ffmpeg_exec_path,src_name,width,height):
        assert(os.path.isfile(ffmpeg_exec_path))
        self.src_name = src_name
        self.width = width
        self.height = height
        self.lock = threading.Lock()
        #self.read_lock = None
        #self.write_lock = None
        #self.next_frame_idx = 1
        #self.timestamp_list = [0] * BUFFER_COUNT
        self.arwj = async_read_write_judge.AsyncReadWriteJudge(self.lock)
        self.closing = False
        self.data_ready = False
        self.ffmpeg_exec_path = ffmpeg_exec_path
        self.buffer = [ bytearray(self.width*self.height*4) for _ in range(BUFFER_COUNT) ]
        nd_list = [ np.frombuffer(b, np.uint8) for b in self.buffer ]
        self.buffer_nd_list = [ np.reshape(nd,(self.height,self.width,4))[:,:,:3] for nd in nd_list ]
    
    def start(self):
        self.thread = threading.Thread( target=self._run )
        self.thread.start()

    def wait_data_ready(self):
        while(True):
            with self.lock:
                if self.data_ready:
                    return
            time.sleep(0.1)

    def close(self):
        self.closing = True
        if self.proc:
            self.proc.wait()

    def get_frame(self):
        #idx = self._get_read_buf_idx()
        idx = self.arwj.get_read_idx()
        return self.buffer_nd_list[idx]

    def release_frame(self):
        self.arwj.release_read_idx()
        #self._release_read_buf()

    def _run(self):
        self.proc = subprocess.Popen([
                self.ffmpeg_exec_path,
                '-nostdin',
                '-f','avfoundation',
                '-pixel_format','uyvy422',
                '-i','{}:none'.format(self.src_name),
                '-vsync','2',
                '-an',
                '-vf','scale={}:{}'.format(self.width,self.height),
                '-pix_fmt','bgr0',
                '-f','rawvideo',
                '-'
            ],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.PIPE
        )
        while (not self.closing):
            #buf = self._get_write_buf()
            buf_idx = self.arwj.get_write_idx()
            buf = self.buffer[buf_idx]
            llen = self.proc.stdout.readinto(buf)
            if llen!=len(buf):
                print('llen!=len(buf), llen={}'.format(llen),file=sys.stderr)
                assert(False)
            #self._release_write_buf()
            self.arwj.release_write_idx()
            with self.lock:
                self.data_ready = True
        #print('terminate',file=sys.stderr)
        self.proc.terminate()
        timeout = time.time() + 5
        while(time.time()<timeout):
            if self.proc.returncode != None:
                return
            time.sleep(0.1)
        #print('kill',file=sys.stderr)
        self.proc.kill()

#    def _get_write_buf(self):
#        with self.lock:
#            assert(self.write_lock == None)
#            tmp_timestamp_list = copy.copy(self.timestamp_list)
#            if self.read_lock != None:
#                tmp_timestamp_list[self.read_lock] = sys.float_info.max
#            idx = tmp_timestamp_list.index(min(tmp_timestamp_list))
#            self.write_lock = idx
#            self.timestamp_list[idx] = self.next_frame_idx
#            self.next_frame_idx += 1
#            return self.buffer[idx]
#
#    def _release_write_buf(self):
#        with self.lock:
#            assert(self.write_lock != None)
#            self.write_lock = None
#
#    def _get_read_buf_idx(self):
#        with self.lock:
#            assert(self.read_lock == None)
#            tmp_timestamp_list = copy.copy(self.timestamp_list)
#            if self.write_lock != None:
#                tmp_timestamp_list[self.write_lock] = 0
#            idx = tmp_timestamp_list.index(max(tmp_timestamp_list))
#            self.read_lock = idx
#            #print(str(idx),file=sys.stderr)
#            return idx
#
#    def _release_read_buf(self):
#        with self.lock:
#            assert(self.read_lock != None)
#            self.read_lock = None

#    def _ffmpeg_exec_path(self):
#        path = __file__
#        path = os.path.realpath(path)
#        path = os.path.dirname(path)
#        path = os.path.join(path,'external','ffmpeg','ffmpeg')
#        assert(os.path.isfile(path))
#        return path

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='video capture')
    parser.add_argument('ffmpeg_exec_path', help='ffmpeg_exec_path')
    parser.add_argument('src_name', help='src_name')
    parser.add_argument('width', type=int, help='width')
    parser.add_argument('height', type=int, help='height')
    args = parser.parse_args()

    vc = VideoCapture(args.ffmpeg_exec_path,args.src_name,args.width,args.height)
    vc.start()
    vc.wait_data_ready()
    print('data_ready')
    for i in range(10):
        #print('get_frame')
        ndata = vc.get_frame()
        fn = 'x{}.png'.format(i)
        print(fn,file=sys.stderr)
        cv2.imwrite(fn,ndata)
        vc.release_frame()
        time.sleep(0.2)
    vc.close()
