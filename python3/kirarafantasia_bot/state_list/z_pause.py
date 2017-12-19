import os
import sys
import pygame

from collections import deque
from clover.common import draw_util
from kirarafantasia_bot import bot_logic as b_logic
from kirarafantasia_bot import bot

VIDEO_SIZE  = bot.VIDEO_SIZE
TOUCH_SIZE  = bot.TOUCH_SIZE
MIN_TOUCH_SIDE = min(*TOUCH_SIZE)

MY_NAME = 'z_pause'
BUTTON_XYZ = -MIN_TOUCH_SIDE/4, bot.TOUCH_SIZE[1]/2, 0

def init(bot_logic):
    bot_logic.v[MY_NAME] = {}
    vv = bot_logic.v[MY_NAME]
    vv['cooldown_0'] = 0
    vv['move_queue'] = deque()

def on_event(bot_logic, event):
    vv = bot_logic.v[MY_NAME]
    if event.type == pygame.MOUSEBUTTONUP:
        pos = pygame.mouse.get_pos()
        if (pos[0] >= 0) and (pos[0] < VIDEO_SIZE[0]) and (pos[1] >= 0) and (pos[1] < VIDEO_SIZE[1]):
            pos = b_logic.btn_xy(pos[0],pos[1],0,0)
            vv['move_queue'].append(pos+(0,))
            vv['move_queue'].append(pos+(1,))
            vv['move_queue'].append(pos+(0,))

def start(bot_logic, t):
    vv = bot_logic.v[MY_NAME]
    vv['cooldown_0'] = t+3
    vv['move_queue'].clear()

def tick(bot_logic, img, arm, t, ret):
    if arm is None:
        return True
    vv = bot_logic.v[MY_NAME]
    if len(vv['move_queue']) > 0:
        ret['arm_move_list'] = []
        while (len(vv['move_queue']) > 0):
            ret['arm_move_list'].append(vv['move_queue'].popleft())
        vv['cooldown_0'] = t+3
    if t < vv['cooldown_0']:
        return True
    else:
        ret['arm_move_list'] = [
            (arm['last_pos'][:2])+(0,),
            BUTTON_XYZ
        ]
        vv['cooldown_0'] = t+3
        return True

def draw(screen, tick_result):
    pass
