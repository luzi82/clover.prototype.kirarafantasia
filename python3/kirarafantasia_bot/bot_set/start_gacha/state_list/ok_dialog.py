import os
import sys
import numpy as np
import pygame

from kirarafantasia_bot.image_recognition.ok_button import classifier as ok_btn_clr
from clover.common import draw_util
from kirarafantasia_bot.bot_set.start_gacha import bot
import kirarafantasia_bot.bot_set.start_gacha.state_list as state_common
import clover.image_recognition

VIDEO_SIZE  = bot.VIDEO_SIZE
TOUCH_SIZE  = bot.TOUCH_SIZE
LOGIC_VIDEO_OFFSET = bot.LOGIC_VIDEO_OFFSET

def init(bot_logic):
    bot_logic.ok_dialog_cooldown_0 = 0
    bot_logic.ok_dialog_cooldown_1 = 0
    bot_logic.ok_dialog_cooldown_2 = 0

    bot_logic.ok_btn_clr = ok_btn_clr.Classifier(ok_btn_clr.MODEL_PATH)
    bot_logic.ok_btn_clr.get(state_common.DUMMY_IMG)

def start(bot_logic, t):
    bot_logic.ok_dialog_cooldown_0 = t+3
    bot_logic.ok_dialog_cooldown_1 = t+6
    bot_logic.ok_dialog_cooldown_2 = t+9

def tick(bot_logic, img, arm, t, ret):
    if t < bot_logic.ok_dialog_cooldown_0:
        return False
    elif t < bot_logic.ok_dialog_cooldown_1:
        btn_xywh, score, max_diff = bot_logic.ok_btn_clr.get(img)

        add_sample = False
        if score < 0:
            add_sample = True
        if max_diff > 1:
            add_sample = True
        if add_sample:
            ret['add_sample_list'].append('ok_dialog')

        ret['ok_dialog_xywh'] = btn_xywh
        ret['draw_screen'] = True
        
        if (score >= 0) and (arm is not None):
            x = (btn_xywh[0]+btn_xywh[2]/2)*TOUCH_SIZE[0]/VIDEO_SIZE[0]
            y = (btn_xywh[1]+btn_xywh[3]/2)*TOUCH_SIZE[1]/VIDEO_SIZE[1]
            
            ret['arm_move_list'] = [
                (arm['last_pos'][:2])+(0,),
                (x,y,0),
                (x,y,1),
                (x,y,0)
            ]
            
            bot_logic.ok_dialog_cooldown_0 = 0
            bot_logic.ok_dialog_cooldown_1 = 0
            bot_logic.ok_dialog_cooldown_2 = t+3
        return True
    elif t < bot_logic.ok_dialog_cooldown_2:
        return False
    else:
        bot_logic.ok_dialog_cooldown_0 = t+3
        bot_logic.ok_dialog_cooldown_1 = t+6
        bot_logic.ok_dialog_cooldown_2 = t+9
        return False

DRAW_SCREEN_XY = np.array([120,0])

def draw(screen, tick_result):
    if 'ok_dialog_xywh' in tick_result:
        btn_xywh = tick_result['ok_dialog_xywh']
        x0,y0,x1,y1 = clover.image_recognition.xywh_to_xyxy(btn_xywh)
        if btn_xywh != None:
            draw_xy00 = (x0+LOGIC_VIDEO_OFFSET[0],y0+LOGIC_VIDEO_OFFSET[1])
            draw_xy01 = (x0+LOGIC_VIDEO_OFFSET[0],y1+LOGIC_VIDEO_OFFSET[1])
            draw_xy10 = (x1+LOGIC_VIDEO_OFFSET[0],y0+LOGIC_VIDEO_OFFSET[1])
            draw_xy11 = (x1+LOGIC_VIDEO_OFFSET[0],y1+LOGIC_VIDEO_OFFSET[1])
            pygame.draw.line(screen, (255,0,0), tuple(draw_xy00), tuple(draw_xy01), 4)
            pygame.draw.line(screen, (255,0,0), tuple(draw_xy10), tuple(draw_xy11), 4)
            pygame.draw.line(screen, (255,0,0), tuple(draw_xy00), tuple(draw_xy10), 4)
            pygame.draw.line(screen, (255,0,0), tuple(draw_xy01), tuple(draw_xy11), 4)
        else:
            screen.blit(draw_util.text('y==None',(0,0,0)), (bot.VIDEO_SIZE[0],20))
