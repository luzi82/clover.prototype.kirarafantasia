import pygame
import sys
import cv2
import numpy as np
from clover.video_input import video_capture
import os
import time
import threading
import collections
import copy
import traceback
from clover.common import async_read_write_judge
import kirarafantasia_bot
import collections
import kirarafantasia_bot.image_recognition.state as ir_state

FFMPEG_EXEC_PATH = os.path.join('dependency','FFmpeg','ffmpeg')

SCREEN_SIZE = 1280, 720
VIDEO_SIZE = kirarafantasia_bot.VIDEO_SIZE
WHITE = 255,255,255
ARM_SPEED = 20000
PREVIEW_0_OFFSET = 0,0
PREVIEW_1_OFFSET   = 0,VIDEO_SIZE[1]
PREVIEW_1_DELAY    = 1

LABEL_DETECT_SAMPLE_WIDTH  = 71
LABEL_DETECT_SAMPLE_HEIGHT = 40

class Bot:

    def main(self, src_name):
        self.lock = threading.RLock()
        self.timestamp = int(time.time()*1000)

        self.run = True
    
        self.vc = video_capture.VideoCapture(FFMPEG_EXEC_PATH,src_name,VIDEO_SIZE[0],VIDEO_SIZE[1])
        self.vc.start()
        self.vc.wait_data_ready()

        pygame.init()
        screen = pygame.display.set_mode(SCREEN_SIZE)

        # self.frame_queue
        # locked by self.lock
        # unit: {'timestamp','img':np.array(h,w,3)}
        self.frame_queue = collections.deque()
        self.frame_queue_fill_thread = threading.Thread(target=self.frame_queue_fill_loop)
        self.frame_queue_fill_thread.start()

        # self.edit_list
        # unit: {'timestamp','img':np.array(h,w,3),'label_score_dict':{label:score}}
        self.edit_frame_list = None

        self.state = 'IDLE'
        self.state_operator_dict = {}
        self.state_operator_dict[None]     = BaseState()
        self.state_operator_dict['IDLE']   = IdleState()
        self.state_operator_dict['RECORD'] = RecordState()
        self.state_operator_dict['EDIT']   = EditState()

        while self.run:
            state_operator = self.get_state_operator()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.run = False
                elif event.type == pygame.KEYDOWN:
                    state_operator.on_key(self,event)
                elif event.type == pygame.MOUSEBUTTONUP:
                    state_operator.on_click(self,event)
                state_operator.on_event(self,event)
    
            if not self.run: break

            screen.fill(WHITE)
            state_operator.draw(self,screen)

            pygame.display.flip()

        print('wait frame_queue_fill_thread start')
        self.frame_queue_fill_thread.join()
        print('wait frame_queue_fill_thread done')

        print('close video input start')
        self.vc.close()
        print('close video input done')

    def get_state_operator(self):
        if self.state in self.state_operator_dict:
            return self.state_operator_dict[self.state]
        else:
            return self.state_operator_dict[None]

    def frame_queue_fill_loop(self):
        last_frame_idx = 0
        while self.run:
            tmp = self.vc.wait_next_frame(last_frame_idx)
            if tmp == 'TIMEOUT':
                continue
            if tmp == 'CLOSING':
                break

            ts = time.time()

            # pop old frame
            with self.lock:
                while True:
                    if len(self.frame_queue) <= 0:
                        break
                    if self.frame_queue[0]['timestamp'] > ts-PREVIEW_1_DELAY:
                        break
                    self.frame_queue.popleft()
            
            last_frame_idx, ts, img_nd = self.vc.get_frame()
            if last_frame_idx == 0:
                break
            
            tmp = {
                'timestamp': ts,
                'img': np.copy(img_nd)
            }
            
            state_operator = self.get_state_operator()
            
            with self.lock:
                self.frame_queue.append(tmp)
                state_operator.on_new_frame(self,tmp)
            
            self.vc.release_frame()


class LabelSetMgr:

    def __init__(self, bot):
        self._bot = bot
    
        # label: {'imgfn_list','mean_nd','std_nd','effect_sample_count'}
        self.label_data_dict = { label: {} for label in ir_state.get_label_list() }
        
        for label, data in label_data_dict.items():
            imgfn_list_fn = os.path.join('image_recognition','label','state','{}.txt'.format(label))
            imgfn_list = clover.common.readlines(imgfn_list_fn)
            data['imgfn_list'] = imgfn_list
        
        for label, data in label_data_dict.items():
            update_mean_std_nd(data)

    def get_score_dict(self, img):
        img = cv2.resize(img,dsize=(LABEL_DETECT_SAMPLE_WIDTH,LABEL_DETECT_SAMPLE_HEIGHT),interpolation=cv2.INTER_AREA)
        
        score_dict = {}
        for label, data in label_data_dict.items():
            if     (data is None)
                or ('effect_sample_count' not in data):
                score_dict[label] = float('inf')
                continue
            d = img-data['mean_nd']
            d /= data['std_nd']
            d = np.absolute(d)
            d = np.mean(d)
            score_dict[label] = d

        min_score = min(list(score_dict.values()))
        score_dict = { label: min_score/i for label, i in score_dict.items() }
        
        return score_dict

    def add_sample(self, label, sample_list):
        fn = os.path.join(
    
def update_mean_std_nd(label_data):
    effect_sample_count = cal_effect_sample_count(len(label_data['imgfn_list']))

    if effect_sample_count <= 0:
        return
    
    if 'effect_sample_count' in label_data:
        if label_data['effect_sample_count'] >= effect_sample_count:
            return
    
    # random pick some sample
    effect_sample = label_data['imgfn_list'][:]
    random.shuffle(effect_sample)
    effect_sample = effect_sample[:effect_sample_count]

    _effect_sample = effect_sample
    effect_sample = []
    for fn in _effect_sample:
        img = cv2.imread(fn)
        img = cv2.resize(img,dsize=(LABEL_DETECT_SAMPLE_WIDTH,LABEL_DETECT_SAMPLE_HEIGHT),interpolation=cv2.INTER_AREA)
        effect_sample.append(img)
    effect_sample = np.asarray(effect_sample)
    
    mean_nd = np.mean(effect_sample,axis=(0))
    std_nd  = np.std(effect_sample,axis=(0))
    std_nd  = np.maximum(std_nd,0.01)
    
    label_data['effect_sample_count'] = effect_sample_count
    label_data['mean_nd'] = mean_nd
    label_data['std_nd']  = std_nd

def cal_effect_sample_count(sample_count):
    return max(min(10,sample_count), math.ceil(sample_count**0.5))

class BaseState:

    def on_key(self, bot, event):
        pass

    def on_click(self, bot, event):
        pass
    
    def on_event(self, bot, event):
        pass

    def draw(self, bot, screen):
        pass

    def on_new_frame(self, bot, frame):
        pass


class IdleState(BaseState):

    def __init__(self):
        self.img_surf = pygame.pixelcopy.make_surface(np.zeros(VIDEO_SIZE+(3,),dtype=np.uint8))

    def draw(self, bot, screen):
        preview_0_nd = None
        preview_1_nd = None
        with bot.lock:
            if len(bot.frame_queue) > 0:
                preview_0_nd = np.copy(bot.frame_queue[0]['img'])
                preview_1_nd = np.copy(bot.frame_queue[-1]['img'])

        if (preview_0_nd is None) or (preview_1_nd is None):
            return

        blit_nd(screen, PREVIEW_0_OFFSET, self.img_surf, preview_0_nd)
        blit_nd(screen, PREVIEW_1_OFFSET, self.img_surf, preview_1_nd)

    def on_key(self, bot, event):
        if event.key == pygame.K_SPACE:
            with bot.lock:
                bot.edit_frame_list = []
                for frame in bot.frame_queue:
                    
            bot.state = 'RECORD'


class RecordState(BaseState):

    def __init__(self):
        self.img_surf = pygame.pixelcopy.make_surface(np.zeros(VIDEO_SIZE+(3,),dtype=np.uint8))

    def draw(self, bot, screen):
        preview_0_nd = None
        preview_1_nd = None
        with bot.lock:
            if len(bot.frame_queue) > 0:
                preview_0_nd = np.copy(bot.frame_queue[0]['img'])
                preview_1_nd = np.copy(bot.frame_queue[-1]['img'])

        if (preview_0_nd is None) or (preview_1_nd is None):
            return

        blit_nd(screen, PREVIEW_0_OFFSET, self.img_surf, preview_0_nd)
        blit_nd(screen, PREVIEW_1_OFFSET, self.img_surf, preview_1_nd)

    def on_key(self, bot, event):
        if event.key == pygame.K_ESC:
            bot.edit_frame_list = None
            bot.state = 'IDLE'
        elif event.key == pygame.K_SPACE:
            bot.state = 'EDIT'


def blit_nd(screen,pos,img_surf,nd):
    nd = cv2.cvtColor(nd, cv2.COLOR_BGR2RGB)
    nd = np.swapaxes(nd,0,1)
    pygame.pixelcopy.array_to_surface(img_surf,nd)
    screen.blit(img_surf,pos)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='video capture')
    parser.add_argument('--src_name', help='src_name')
    args = parser.parse_args()

    bot = Bot()
    bot.main(args.src_name)
