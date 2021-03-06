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
#from clover.zookeeper.state_list import battle
##from clover.zookeeper.state_list import battle_result
##from clover.zookeeper.state_list import boss_henchman_appeared
##from clover.zookeeper.state_list import limited_time_sale
#from clover.zookeeper.state_list import main_menu
##from clover.zookeeper.state_list import mission_boss_invasion
#from clover.zookeeper.state_list import ok_dialog
##from clover.zookeeper.state_list import title

from kirarafantasia_bot.bot_set.start_gacha.state_list import z_pause
from kirarafantasia_bot.bot_set.start_gacha.state_list import ok_dialog
from kirarafantasia_bot.bot_set.start_gacha.state_list import click
from kirarafantasia_bot.bot_set.start_gacha.state_list import gacha_result
from . import bot
from clover import common
import shutil
import kirarafantasia_bot as kbot

SCREEN_SIZE = bot.SCREEN_SIZE
VIDEO_SIZE  = bot.VIDEO_SIZE
TOUCH_SIZE  = bot.TOUCH_SIZE

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
        #self.state_op_dict['power_bottle'] = _click.Click('power_bottle',btn_xy(21,124,31,14),3)
        #self.state_op_dict['title'] = _click.Click('mission_boss_invasion',(166+(308/2), 498+(106/2)),3)

        self.state_op_dict['center']       = click.Click('center',(TOUCH_SIZE[0]/2,TOUCH_SIZE[1]/2),3)
        self.state_op_dict['gacha_result'] = gacha_result
        self.state_op_dict['ok_dialog']    = ok_dialog
        #self.state_op_dict['top_right']    = click.Click('top_right',btn_xy(489,9,70,15),1)
        self.state_op_dict['top_right']    = click.Click('top_right',btn_xy(489,9+5,70,15),2,0.5)
        self.state_op_dict['gacha_retry_dialog'] = click.Click('gacha_retry_dialog',btn_xy(181,215,89,23),2)
        self.state_op_dict['ios_menu']    = click.Click('top_right',btn_xy(27,244,60,60),2)

        self.play = False
        self.cap_screen = False
        
        self.cap_screen_timeout = 0
        self.cap_screen_output_folder = os.path.join('image_recognition','screen_sample',str(int(time.time()*1000)))
        
        self.last_ticker = None
        
        self.v = {}
        self.d = {}
        
        self.draw_img_surf = pygame.pixelcopy.make_surface(np.zeros((VIDEO_SIZE[0],VIDEO_SIZE[1],3),dtype=np.uint8))
        self.draw_tick_result = None
        
        self.last_state_perfect = None
        
        self.init_done = False

    def init(self):
        self.state_clr = state_classifier.StateClassifier(state_classifier.MODEL_PATH)
        for _, state_op in self.state_op_dict.items():
            state_op.init(self)
        z_pause.init(self)
        self.init_done = True

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
        
        best_state, _, state_perfect = self.state_clr.get(img)
        state = best_state if state_perfect else None

        ret = {
            'state_perfect': state_perfect,
            'best_state': best_state,
            'state': state,
            'play': self.play,
            'add_sample_list': [],
        }
        
        # store img if non perfect for 5 sec
        STATE_PERFECT_TIMEOUT = 5
        if state_perfect:
            self.last_state_perfect = time_s
        elif (self.last_state_perfect is not None) and (time_s > self.last_state_perfect+STATE_PERFECT_TIMEOUT):
            print('HAVFWEEKPD not perfect: {}'.format(state_perfect))
            self.last_state_perfect = None
            ret['add_sample_list'].append(os.path.join('state',best_state))
        
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

        if len(ret['add_sample_list'])>0:
            do_cap_screen = True

        self.last_ticker = ticker

        #print(json.dumps(ret))

        if do_cap_screen:
            if time_s >= self.cap_screen_timeout:
                t = int(time_s*1000)
                t0 = int(t/100000)
                fn_dir = os.path.join(self.cap_screen_output_folder,str(t0))
                common.makedirs(fn_dir)
                fn0 = '{}.png'.format(t)
                fn = os.path.join(fn_dir,fn0)
                cv2.imwrite(fn,img_cap)
                self.cap_screen_timeout = time_s + 0.5 + 0.5*random.random()
                
                for add_sample in ret['add_sample_list']:
                    fnn_dir = os.path.join('add_sample',add_sample)
                    common.makedirs(fnn_dir)
                    fnn = os.path.join(fnn_dir,fn0)
                    shutil.copyfile(fn,fnn)

        return ret if good else None

    
    def draw(self, screen, img_surf, tick_result):
        if self.init_done:

            if tick_result is not None:
                state = tick_result['state']
                if state is not None:
                    screen.blit(draw_util.text(state,(0,0,0)), (VIDEO_SIZE[0],0))
                else:
                    screen.blit(draw_util.text('_NONE',(0,0,0)), (VIDEO_SIZE[0],0))
                if tick_result['play']:
                    if state in self.state_op_dict:
                        self.state_op_dict[state].draw(screen, tick_result)

            if (tick_result is not None) and (not tick_result['play']):
                z_pause.draw(screen, tick_result)
    
            if (tick_result is not None) and ('draw_screen' in tick_result) and (tick_result['draw_screen']):
                self.draw_tick_result = tick_result
                self.draw_img_surf.blit(img_surf,(0,0))
            
            screen.blit(self.draw_img_surf,(0,VIDEO_SIZE[1]))
            if self.draw_tick_result is not None:
                state = self.draw_tick_result['state']
                if state in self.state_op_dict:
                    self.state_op_dict[state].draw(screen, self.draw_tick_result)
    
            for _, ticker in self.state_op_dict.items():
                if not hasattr(ticker, 'force_draw'):
                    continue
                ticker.force_draw(self, screen, tick_result)

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

btn_xy = kbot.btn_xy

BTN_SIZE = 60
def btn_rect(idx):
     return (SCREEN_SIZE[0]-(BTN_SIZE*(idx+1)),SCREEN_SIZE[1]-BTN_SIZE,SCREEN_SIZE[0]-(BTN_SIZE*idx),SCREEN_SIZE[1])
PLAY_BTN_RECT = btn_rect(0)
SCREENCAP_BTN_RECT = btn_rect(1)
