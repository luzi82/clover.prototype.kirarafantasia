import numpy as np
import pygame
import os
import json
import random
import time
import cv2

from clover.common import draw_util
import kirarafantasia_bot.image_recognition.state.classifier as state_classifier
#import clover.zookeeper.state_list as state_common
#from clover.zookeeper.state_list import _click
#from clover.zookeeper.state_list import battle
##from clover.zookeeper.state_list import battle_result
##from clover.zookeeper.state_list import boss_henchman_appeared
##from clover.zookeeper.state_list import limited_time_sale
#from clover.zookeeper.state_list import main_menu
##from clover.zookeeper.state_list import mission_boss_invasion
#from clover.zookeeper.state_list import ok_dialog
##from clover.zookeeper.state_list import title

from kirarafantasia_bot.state_list import z_pause
from . import bot
from clover import common

SCREEN_SIZE = bot.SCREEN_SIZE
VIDEO_SIZE  = bot.VIDEO_SIZE

class BotLogic:

    def __init__(self):
        self.state_op_dict = {}

        #self.state_op_dict['battle'] = battle
        ##self.state_op_dict['battle_result'] = _click.Click('battle_result',(350+(104/2), 946+(86/2)),3)
        #self.state_op_dict['battle_result'] = _click.Click('battle_result',(32+(80/2), 946+(86/2)),3)
        #self.state_op_dict['boss_henchman_appeared'] = _click.Click('boss_henchman_appeared',((40+(48/2))*640/120, (147+(18/2))*1136/213),3)
        #self.state_op_dict['full_zoo'] = _click.Click('level_up',btn_xy(104,57,12,9),3)
        #self.state_op_dict['level_up'] = _click.Click('level_up',btn_xy(320,568,0,0),3)
        #self.state_op_dict['limited_time_sale'] = _click.Click('limited_time_sale',(79+(104/2), 837+(52/2)),3)
        #self.state_op_dict['main_menu'] = main_menu
        #self.state_op_dict['mission_boss_invasion'] = _click.Click('mission_boss_invasion',((6+(17/2))*640/120, (144+(11/2))*1136/213),3)
        #self.state_op_dict['no_cp'] = _click.Click('no_cp',btn_xy(21,120,30,13),3)
        #self.state_op_dict['ok_dialog'] = ok_dialog
        #self.state_op_dict['power_bottle'] = _click.Click('power_bottle',btn_xy(21,124,31,14),3)
        #self.state_op_dict['title'] = _click.Click('mission_boss_invasion',(166+(308/2), 498+(106/2)),3)

        self.play = False
        self.cap_screen = False
        
        self.cap_screen_timeout = 0
        self.cap_screen_output_folder = os.path.join('image_recognition','screen_sample',str(int(time.time()*1000)))
        
        self.last_ticker = None
        
        self.v = {}

    def init(self):
        self.state_clr = state_classifier.StateClassifier(state_classifier.MODEL_PATH)
        for _, state_op in self.state_op_dict.items():
            state_op.init(self)
        z_pause.init(self)

    def on_event(self,event):
        if event.type == pygame.MOUSEBUTTONUP:
            pos = pygame.mouse.get_pos()
            if common.in_rect(pos,PLAY_BTN_RECT):
                self.play = not self.play
            if common.in_rect(pos,SCREENCAP_BTN_RECT):
                self.cap_screen = not self.cap_screen
        if (self.last_ticker != None) and (hasattr(self.last_ticker,'on_event')):
            self.last_ticker.on_event(self,event)

    def tick(self,img,arm_data,time_s):
        do_cap_screen = False
        do_cap_screen = do_cap_screen or self.cap_screen
        img_cap = img
        
        img = img.astype('float32')*2/255-1
        
        state, state_perfect = self.state_clr.get_state(img)
        ret = {
            'state': state,
            'play': self.play
        }
        
        if not state_perfect:
            print('HAVFWEEKPD not perfect: {}'.format(state))
        
        do_cap_screen = do_cap_screen or (not state_perfect)

        ticker = None
        if self.play:
            if state in self.state_op_dict:
                ticker = self.state_op_dict[state]
        else:
            ticker = z_pause

        if ticker != self.last_ticker:
            if (self.last_ticker!= None)and(hasattr(self.last_ticker,'end')):
                self.last_ticker.end(self,time_s)
            if (ticker!= None)and(hasattr(ticker,'start')):
                ticker.start(self,time_s)

        good = True
        if (ticker!= None)and(hasattr(ticker,'tick')):
            good = ticker.tick(self, img, arm_data, time_s, ret)

        self.last_ticker = ticker

        #print(json.dumps(ret))

        if do_cap_screen:
            if time_s >= self.cap_screen_timeout:
                t = int(time_s*1000)
                t0 = int(t/100000)
                fn_dir = os.path.join(self.cap_screen_output_folder,str(t0))
                common.makedirs(fn_dir)
                fn = os.path.join(fn_dir,'{}.png'.format(t))
                cv2.imwrite(fn,img_cap)
                self.cap_screen_timeout = time_s + 0.5 + 0.5*random.random()

        return ret if good else None

    
    def draw(self, screen, tick_result):
        if tick_result != None:
            state = tick_result['state']
            screen.blit(draw_util.text(state,(0,0,0)), (VIDEO_SIZE[0],0))
            if tick_result['play']:
                if state in self.state_op_dict:
                    self.state_op_dict[state].draw(screen, tick_result)

        if self.play:
            screen.blit(draw_util.text('P',(0,127,0)), PLAY_BTN_RECT[:2])
        else:
            screen.blit(draw_util.text('S',(127,0,0)), PLAY_BTN_RECT[:2])

        if self.cap_screen:
            screen.blit(draw_util.text('CAP',(127,0,0)), SCREENCAP_BTN_RECT[:2])
        else:
            screen.blit(draw_util.text('NoCAP',(0,127,0)), SCREENCAP_BTN_RECT[:2])

#    def render_state(self, screen, state):
#        if not hasattr(self, 'state_render_dict'):
#            self.state_render_dict = {}
#        if not hasattr(self, 'state_render_font'):
#            self.state_render_font = pygame.font.SysFont("monospace", 15)
#        if not state in self.state_render_dict:
#            self.state_render_dict[state] = self.state_render_font.render(state, 1, (0,0,0))
#        screen.blit(self.state_render_dict[state], (240,0))

def btn_xy(x,y,w,h):
    return ((x+(w/2))*640/120, (y+(h/2))*1136/213)

BTN_SIZE = 30
def btn_rect(idx):
     return (SCREEN_SIZE[0]-(BTN_SIZE*(idx+1)),SCREEN_SIZE[1]-BTN_SIZE,SCREEN_SIZE[0]-(BTN_SIZE*idx),SCREEN_SIZE[1])
PLAY_BTN_RECT = btn_rect(0)
SCREENCAP_BTN_RECT = btn_rect(1)
