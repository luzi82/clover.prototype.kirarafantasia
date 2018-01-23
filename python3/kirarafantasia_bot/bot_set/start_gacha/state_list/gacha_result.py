import os
import sys
import numpy as np
import pygame
import math
import json
import time

from kirarafantasia_bot.image_recognition.gacha_result import classifier as gr_clr
from clover.common import draw_util
import clover.common
from kirarafantasia_bot.bot_set.start_gacha import bot
import kirarafantasia_bot.bot_set.start_gacha.state_list as state_common
import clover.image_recognition
from kirarafantasia_bot.image_recognition.gacha_result import setting as gr_setting
import copy

NAME = 'gacha_result'
HISTORY_FILE = 'gacha_result_history.txt'

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
    
    bot_logic.v[NAME] = {}
    bot_logic.v[NAME]['last_img_box_list'] = cal_box_list(state_common.DUMMY_IMG)
    bot_logic.v[NAME]['hi'] = -1.1244443326651827
    bot_logic.v[NAME]['lo'] = -4.414975405913796

    #gacha_result_stat = clover.common.read_json('gacha_result_stat.json')
    gacha_result_stat = {
        'single':[0]*3,
        'bad_single':0,
        'ten_s5':[0]*6,
        'bad_ten':0,
    }
    if os.path.isfile(HISTORY_FILE):
        for line in clover.common.readlines(HISTORY_FILE):
            gacha_result = json.loads(line)
            update_stat(gacha_result_stat, gacha_result)
    bot_logic.v[NAME]['gacha_result_stat'] = gacha_result_stat

    bot_logic.d[NAME] = {}
    bot_logic.d[NAME]['gacha_result_stat'] = copy.copy(gacha_result_stat)
    bot_logic.d[NAME]['alarm_cooldown'] = 0

def start(bot_logic, t):
    bot_logic.gacha_result_cooldown_0 = t+3
    bot_logic.gacha_result_cooldown_1 = t+99
    bot_logic.gacha_result_cooldown_2 = t+999

def tick(bot_logic, img, arm, t, ret):
    if t < bot_logic.gacha_result_cooldown_0:
        return False
    elif t < bot_logic.gacha_result_cooldown_1:
        label_list, score_list, perfect, ptp_list = bot_logic.gr_clr.get(img)
        predict_good = [ (1 if (ptp_list[i]<=0)and(score_list[i]>=0.5) else 0) for i in range(len(label_list))]
        
        now_box_list = cal_box_list(img)
        img_diff = np.average(np.absolute(now_box_list-bot_logic.v[NAME]['last_img_box_list']))
        log_img_diff = math.log(img_diff)
        if log_img_diff > -2.5:
            bot_logic.v[NAME]['hi'] = min(bot_logic.v[NAME]['hi'],log_img_diff)
        else:
            bot_logic.v[NAME]['lo'] = max(bot_logic.v[NAME]['lo'],log_img_diff)
        diff_mean = (bot_logic.v[NAME]['hi']+bot_logic.v[NAME]['lo'])/2

        print('{}, {}, {}, {}, {}'.format(img_diff,log_img_diff,bot_logic.v[NAME]['hi'],bot_logic.v[NAME]['lo'],diff_mean))
        bot_logic.v[NAME]['last_img_box_list'] = np.copy(now_box_list)

        gacha_result_stat = bot_logic.v[NAME]['gacha_result_stat']
        if log_img_diff > -2.75:
            gacha_result = {
                'label_list': label_list,
                'predict_good': predict_good,
                'time':int(t*1000),
            }
            clover.common.appendlines(HISTORY_FILE,[json.dumps(gacha_result)])
            update_stat(gacha_result_stat, gacha_result)
        
        ret['gacha_result_stat'] = copy.copy(gacha_result_stat)

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
        ret['predict_good'] = predict_good
        ret['draw_screen'] = True
        
        s5count = sum([ 1 if i == 's5' else 0 for i in label_list ])
        bad_count  = sum([ 1 if i > 0 else 0 for i in ptp_list ])
        bad_count += sum([ 1 if i == 'bad' else 0 for i in label_list ])
        ret['bingo'] = (s5count+bad_count >= 4)
        
        if (not ret['bingo']) and (arm is not None):
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
        else:
            bot_logic.gacha_result_cooldown_0 = t+3
            bot_logic.gacha_result_cooldown_1 = t+99
            bot_logic.gacha_result_cooldown_2 = t+999
        
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
            color = (0,255,0) if (tick_result['predict_good'][i]>0) else (255,0,0)
            screen.blit(draw_util.text(gacha_result_label,color), (x,bot.VIDEO_SIZE[1]+y))


def force_draw(bot_logic, screen, tick_result):
    if 'sound_effect' not in bot_logic.d[NAME]:
        bot_logic.d[NAME]['sound_effect'] = pygame.mixer.Sound(os.path.join('resource_set','tadUPD04.wav'))

    if (tick_result is not None) and ('gacha_result_stat' in tick_result):
        bot_logic.d[NAME]['gacha_result_stat'] = tick_result['gacha_result_stat']

    gacha_result_stat = bot_logic.d[NAME]['gacha_result_stat']
    if gacha_result_stat is not None:
        y=60
        single_sum = sum(gacha_result_stat['single'])
        for i in range(len(gacha_result_stat['single'])):
            if single_sum > 0:
                star = 3+i
                count = gacha_result_stat['single'][i]
                ratio = count*100/single_sum
                screen.blit(draw_util.text('s{0}: {1}, {2:.2f}%'.format(star,count,ratio),(0,0,0)), (bot.VIDEO_SIZE[0], y))
            y+=20
    
        screen.blit(draw_util.text('X: {0}'.format(gacha_result_stat['bad_single']),(0,0,0)), (bot.VIDEO_SIZE[0], y))
        y+=20
        y+=20
        
        ten_sum = sum(gacha_result_stat['ten_s5'])
        for i in range(len(gacha_result_stat['ten_s5'])):
            count = gacha_result_stat['ten_s5'][i]
            if count > 0:
                ratio = ten_sum/count
                ratio = '{0:.02f}'.format(ratio)
            else:
                ratio = 'inf'
            screen.blit(draw_util.text('s5 x{0}: {1}, 1/{2}'.format(i,count,ratio),(0,0,0)), (bot.VIDEO_SIZE[0], y))
            y+=20
        screen.blit(draw_util.text('X: {0}'.format(gacha_result_stat['bad_ten']),(0,0,0)), (bot.VIDEO_SIZE[0], y))
        y+=20
        y+=20
        
    if     (tick_result is not None) \
       and ('bingo' in tick_result) \
       and (tick_result['bingo']) \
       and (time.time()>bot_logic.d[NAME]['alarm_cooldown']):
           bot_logic.d[NAME]['sound_effect'].play()
           bot_logic.d[NAME]['alarm_cooldown'] = time.time()+15

# check for hardcode
for _,_,w,h in BOUND_BOX_XYWH_LIST:
    assert(w==55)
    assert(h==55)

def cal_box_list(img):
    img_list = np.empty((len(BOUND_BOX_XYWH_LIST),55,55,3),np.float32)
    for i in range(len(BOUND_BOX_XYWH_LIST)):
        x,y,w,h = BOUND_BOX_XYWH_LIST[i]
        xw, yh = x+w, y+h
        img_list[i] = img[y:yh,x:xw,:]
    return img_list

def update_stat(gacha_result_stat, gacha_result):
    label_list = gacha_result['label_list']
    predict_good = gacha_result['predict_good']
    perfect = True

    ten_s5 = 0
    for i in range(len(label_list)):
        if (predict_good[i]>0) and (label_list[i]!='bad'):
            label_lv = 2 if label_list[i]=='s5' else \
                       2 if label_list[i]=='d5' else \
                       1 if label_list[i]=='s4' else \
                       1 if label_list[i]=='d4' else \
                       0
            gacha_result_stat['single'][label_lv] += 1
            if label_lv == 2:
                ten_s5 += 1
        else:
            gacha_result_stat['bad_single'] += 1
            perfect = False

    ten_s5 = min(ten_s5,5)
    if perfect:
        gacha_result_stat['ten_s5'][ten_s5] += 1
    else:
        gacha_result_stat['bad_ten'] += 1
