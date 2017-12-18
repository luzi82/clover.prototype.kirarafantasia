import os
import sys

class Click:

    def __init__(self,name,button_xy,delay,period=None):
        self.name = name
        self.button_xy = button_xy
        self.delay = delay
        self.period = period if period is not None else delay

    def init(self,bot_logic):
        bot_logic.v[self.name] = {}
        vv = bot_logic.v[self.name]
        vv['cooldown_0'] = 0
    
    def start(self,bot_logic, t):
        vv = bot_logic.v[self.name]
        vv['cooldown_0'] = t+self.delay
    
    def tick(self,bot_logic, img, arm, t, ret):
        vv = bot_logic.v[self.name]
        if t < vv['cooldown_0']:
            return True
        else:
            if arm is not None:
                ret['arm_move_list'] = [
                    (arm['last_pos'][:2])+(0,),
                    self.button_xy+(0,),
                    self.button_xy+(1,),
                    self.button_xy+(0,)
                ]
                vv['cooldown_0'] = t+self.period
            return True
    
    def draw(self,screen, tick_result):
        pass
