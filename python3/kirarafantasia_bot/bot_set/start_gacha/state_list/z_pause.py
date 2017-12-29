import os
import sys
import pygame

from collections import deque
from clover.common import draw_util
import kirarafantasia_bot as kbot
import kirarafantasia_bot.bot_set.start_gacha.bot_logic as b_logic
import clover.common

VIDEO_SIZE  = kbot.VIDEO_SIZE
TOUCH_SIZE  = kbot.TOUCH_SIZE
MIN_TOUCH_SIDE = min(*TOUCH_SIZE)

MY_NAME = 'z_pause'
BUTTON_XYZ = -MIN_TOUCH_SIDE/4, kbot.TOUCH_SIZE[1]/2, 0

DRAG_NORTH_BTN_RECT = b_logic.btn_rect(7)
DRAG_SOUTH_BTN_RECT = b_logic.btn_rect(6)
DRAG_EAST_BTN_RECT = b_logic.btn_rect(5)
DRAG_WEST_BTN_RECT = b_logic.btn_rect(4)
HOME_1_BTN_RECT = b_logic.btn_rect(3)
HOME_2_BTN_RECT = b_logic.btn_rect(2)

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
            pos = kbot.btn_xy(pos[0],pos[1],0,0)
            vv['move_queue'].append(pos+(0,))
            vv['move_queue'].append(pos+(1,))
            vv['move_queue'].append(pos+(0,))
        elif clover.common.in_rect(pos,DRAG_NORTH_BTN_RECT):
            vv['move_queue'].append((1136*1/2,640*7/8,0))
            vv['move_queue'].append((1136*1/2,640*7/8,1))
            vv['move_queue'].append((1136*1/2,640*1/8,1))
            vv['move_queue'].append((1136*1/2,640*1/8,0))
        elif clover.common.in_rect(pos,DRAG_SOUTH_BTN_RECT):
            vv['move_queue'].append((1136*1/2,640*1/8,0))
            vv['move_queue'].append((1136*1/2,640*1/8,1))
            vv['move_queue'].append((1136*1/2,640*7/8,1))
            vv['move_queue'].append((1136*1/2,640*7/8,0))
        elif clover.common.in_rect(pos,DRAG_EAST_BTN_RECT):
            vv['move_queue'].append((1136*2/8,640*4/8,0))
            vv['move_queue'].append((1136*2/8,640*4/8,1))
            vv['move_queue'].append((1136*7/8,640*4/8,1))
            vv['move_queue'].append((1136*7/8,640*4/8,0))
        elif clover.common.in_rect(pos,DRAG_WEST_BTN_RECT):
            vv['move_queue'].append((1136*6/8,640*4/8,0))
            vv['move_queue'].append((1136*6/8,640*4/8,1))
            vv['move_queue'].append((1136*1/8,640*4/8,1))
            vv['move_queue'].append((1136*1/8,640*4/8,0))
        elif clover.common.in_rect(pos,HOME_1_BTN_RECT):
            vv['move_queue'].append((1136+130,640*4/8,0))
            vv['move_queue'].append((1136+130,640*4/8,2.2))
            vv['move_queue'].append((1136+130,640*4/8,0))
        elif clover.common.in_rect(pos,HOME_2_BTN_RECT):
            vv['move_queue'].append((1136+130,640*4/8,0))
            vv['move_queue'].append((1136+130,640*4/8,2.2))
            vv['move_queue'].append((1136+130,640*4/8,1.9))
            vv['move_queue'].append((1136+130,640*4/8,2.2))
            vv['move_queue'].append((1136+130,640*4/8,0))

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
    screen.blit(draw_util.text('N',(0,0,0)), DRAG_NORTH_BTN_RECT[:2])
    screen.blit(draw_util.text('S',(0,0,0)), DRAG_SOUTH_BTN_RECT[:2])
    screen.blit(draw_util.text('E',(0,0,0)), DRAG_EAST_BTN_RECT[:2])
    screen.blit(draw_util.text('W',(0,0,0)), DRAG_WEST_BTN_RECT[:2])
    screen.blit(draw_util.text('H1',(0,0,0)), HOME_1_BTN_RECT[:2])
    screen.blit(draw_util.text('H2',(0,0,0)), HOME_2_BTN_RECT[:2])
