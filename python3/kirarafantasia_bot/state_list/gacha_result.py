import os
import sys
import numpy as np
import pygame

from kirarafantasia_bot.image_recognition.gacha_result import classifier as gr_clr
from clover.common import draw_util
from kirarafantasia_bot import bot
import kirarafantasia_bot.state_list as state_common
import clover.image_recognition
from kirarafantasia_bot.image_recognition.gacha_result import setting as gr_setting

VIDEO_SIZE  = bot.VIDEO_SIZE
TOUCH_SIZE  = bot.TOUCH_SIZE
LOGIC_VIDEO_OFFSET = bot.LOGIC_VIDEO_OFFSET
BOUND_BOX_XYWH_LIST = gr_setting.BOUND_BOX_XYWH_LIST

def init(bot_logic):
    bot_logic.gacha_result_cooldown_0 = 0
    bot_logic.gacha_result_cooldown_1 = 0
    bot_logic.gacha_result_cooldown_2 = 0

    bot_logic.gr_clr = gr_clr.Classifier(gr_clr.MODEL_PATH)
    bot_logic.gr_clr.get(state_common.DUMMY_IMG)

def start(bot_logic, t):
    bot_logic.gacha_result_cooldown_0 = t+3
    bot_logic.gacha_result_cooldown_1 = t+6
    bot_logic.gacha_result_cooldown_2 = t+9

def tick(bot_logic, img, arm, t, ret):
    if t < bot_logic.gacha_result_cooldown_0:
        return False
    elif t < bot_logic.gacha_result_cooldown_1:
        label_list, score_list, perfect, ptp_list = bot_logic.gr_clr.get(img)

        add_sample = False
        if not perfect:
            add_sample = True
        if np.amin(score_list) < 0.5:
            add_sample = True
        if add_sample:
            print('VZDFTPLWUG add_sample_list gacha_result')
            ret['add_sample_list'].append('gacha_result')

        ret['gacha_result_label_list'] = label_list
        ret['perfect'] = perfect
        ret['ptp_list'] = ptp_list
        
        s5count = sum([ 1 if i == 's5' else 0 for i in label_list ])
        bad_count = sum([ 1 if i > 0 else 0 for i in ptp_list ])
        
        if (s5count+bad_count < 3) and (arm is not None):
            btn_xywh = (333,241,89,24)
            x = (btn_xywh[0]+btn_xywh[2]/2)*TOUCH_SIZE[0]/VIDEO_SIZE[0]
            y = (btn_xywh[1]+btn_xywh[3]/2)*TOUCH_SIZE[1]/VIDEO_SIZE[1]
            
            ret['arm_move_list'] = [
                (arm['last_pos'][:2])+(0,),
                (x,y,0),
                (x,y,1),
                (x,y,0)
            ]
            
            bot_logic.gacha_result_cooldown_0 = 0
            bot_logic.gacha_result_cooldown_1 = 0
            bot_logic.gacha_result_cooldown_2 = t+3
        
#        if (score >= 0) and (arm is not None):
#            x = (btn_xywh[0]+btn_xywh[2]/2)*TOUCH_SIZE[0]/VIDEO_SIZE[0]
#            y = (btn_xywh[1]+btn_xywh[3]/2)*TOUCH_SIZE[1]/VIDEO_SIZE[1]
#            
#            ret['arm_move_list'] = [
#                (arm['xyz'][:2])+(0,),
#                (x,y,0),
#                (x,y,1),
#                (x,y,0)
#            ]
#            
#            bot_logic.gacha_result_cooldown_0 = 0
#            bot_logic.gacha_result_cooldown_1 = 0
#            bot_logic.gacha_result_cooldown_2 = t+3

        return True
    elif t < bot_logic.gacha_result_cooldown_2:
        return False
    else:
        bot_logic.gacha_result_cooldown_0 = t+3
        bot_logic.gacha_result_cooldown_1 = t+6
        bot_logic.gacha_result_cooldown_2 = t+9
        return False

DRAW_SCREEN_XY = np.array([120,0])

def draw(screen, tick_result):
    if 'gacha_result_label_list' in tick_result:
        gacha_result_label_list = tick_result['gacha_result_label_list']
        screen.blit(draw_util.text('perfect={}'.format(tick_result['perfect']),(0,0,0)), (bot.VIDEO_SIZE[0],20))
        for i in range(len(BOUND_BOX_XYWH_LIST)):
            x,y,_,_ = BOUND_BOX_XYWH_LIST[i]
            gacha_result_label = gacha_result_label_list[i]
            screen.blit(draw_util.text(gacha_result_label,(0,0,0)), (x+2,bot.VIDEO_SIZE[1]+y))
            screen.blit(draw_util.text(gacha_result_label,(0,0,0)), (x-2,bot.VIDEO_SIZE[1]+y))
            screen.blit(draw_util.text(gacha_result_label,(0,0,0)), (x,bot.VIDEO_SIZE[1]+y+2))
            screen.blit(draw_util.text(gacha_result_label,(0,0,0)), (x,bot.VIDEO_SIZE[1]+y-2))
            color = (0,255,0) if (tick_result['ptp_list'][i] <= 0) else (255,0,0)
            screen.blit(draw_util.text(gacha_result_label,color), (x,bot.VIDEO_SIZE[1]+y))
