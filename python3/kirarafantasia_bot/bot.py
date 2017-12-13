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
from . import bot_logic
import traceback
from clover.common import async_read_write_judge
from clover.uarmswiftpro import uarm_screen

FFMPEG_EXEC_PATH = os.path.join('dependency','FFmpeg','ffmpeg')

SCREEN_SIZE = 1280, 720
VIDEO_SIZE = 568, 320
WHITE = 255,255,255
#ARM_SPEED = 10000

class Bot:

    def __init__(self):
        self.uarm_screen_last_cmd = None
        self.last_pos = None
    
    def main(self, src_name, uarm_calibration_filename):
        self.lock = threading.RLock()

        self.run = True
    
        self.vc = video_capture.VideoCapture(FFMPEG_EXEC_PATH,src_name,VIDEO_SIZE[0],VIDEO_SIZE[1])
        self.vc.start()
        self.vc.wait_data_ready()

        self.uarm_screen = None
        if uarm_calibration_filename is not None:
            self.uarm_screen = uarm_screen.UArmScreen(uarm_calibration_filename)
            self.uarm_screen.connect()
            self.uarm_screen.wait_ready()
            self.uarm_screen.set_report_position(True).wait()
            self.uarm_screen.wait_report_position_ready()
            #self.uarm_screen.set_acceleration(300,300).wait()

        pygame.init()
        img_surf = pygame.pixelcopy.make_surface(np.zeros((VIDEO_SIZE[0],VIDEO_SIZE[1],3),dtype=np.uint8))
        screen = pygame.display.set_mode(SCREEN_SIZE)

        self.logic_img_buf = [np.zeros((VIDEO_SIZE[1],VIDEO_SIZE[0],3),dtype=np.uint8) for _ in range(async_read_write_judge.BUFFER_COUNT)]
        self.logic_result_buf = [None] * async_read_write_judge.BUFFER_COUNT
        self.logic_result_arwj = async_read_write_judge.AsyncReadWriteJudge(self.lock)
        self.logic = bot_logic.BotLogic()
        self.logic_thread = threading.Thread(target=self.logic_run)
        self.logic_thread.start()

        img = np.zeros((VIDEO_SIZE[1],VIDEO_SIZE[0],3),dtype=np.uint8)

        while self.run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.run = False
                    break
                self.logic.on_event(event)
    
            if not self.run: break

            screen.fill(WHITE)
    
            with self.lock:
                tmp_img = self.vc.get_frame()
                np.copyto(img, tmp_img)
                self.vc.release_frame()
            tmp_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tmp_img = np.swapaxes(tmp_img,0,1)
            pygame.pixelcopy.array_to_surface(img_surf,tmp_img)
            screen.blit(img_surf,(0,0))

            logic_read_idx = self.logic_result_arwj.get_read_idx()
            if self.logic_result_arwj.get_order_idx(logic_read_idx) != 0:
                assert(self.logic_result_buf[logic_read_idx] != None)
                np.copyto(img, self.logic_img_buf[logic_read_idx])
                tmp_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tmp_img = np.swapaxes(tmp_img,0,1)
                pygame.pixelcopy.array_to_surface(img_surf,tmp_img)
                screen.blit(img_surf,(0,VIDEO_SIZE[1]))
                self.logic.draw(screen,self.logic_result_buf[logic_read_idx])
            else:
                self.logic.draw(screen,None)
            self.logic_result_arwj.release_read_idx()
            
            pygame.display.flip()
    
        self.logic_thread.join()
        self.vc.close()
        if self.uarm_screen is not None:
            self.uarm_screen.close()

    def logic_run(self):
        try:
            self.logic.init()
            img = np.zeros((VIDEO_SIZE[1],VIDEO_SIZE[0],3),dtype=np.uint8)
            while self.run:
                with self.lock:
                    tmp_img = self.vc.get_frame()
                    np.copyto(img, tmp_img)
                    self.vc.release_frame()
                arm_data = self.get_arm_data()
                logic_result = self.logic.tick(img, arm_data, time.time())
                if logic_result:
                    write_idx = self.logic_result_arwj.get_write_idx()
                    np.copyto(self.logic_img_buf[write_idx], img)
                    self.logic_result_buf[write_idx] = logic_result
                    self.logic_result_arwj.release_write_idx()
                if logic_result and 'arm_move_list' in logic_result:
                    for pos in logic_result['arm_move_list']:
                        self.uarm_screen_last_cmd = self.uarm_screen.set_position(pos,ARM_SPEED)
                        self.last_pos = list(pos)
        except:
            traceback.print_exc()

    def get_arm_data(self):
        if self.uarm_screen is None:
            return None
        ret = {}
        ret['is_busy'] = ( self.uarm_screen_last_cmd != None ) and ( self.uarm_screen_last_cmd.is_busy() )
        ret['xyz'] = self.uarm_screen.get_last_report_position()
        ret['last_pos'] = self.last_pos if (self.last_pos!=None) else ret['xyz']
        return ret

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='video capture')
    parser.add_argument('--src_name', help='src_name')
    parser.add_argument('--uarm_calibration', nargs='?', help='uarm_calibration')
    args = parser.parse_args()

    bot = Bot()
    bot.main(args.src_name, args.uarm_calibration)
