import pygame
import sys
import cv2
import numpy
from . import video_capture

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='video capture')
    parser.add_argument('ffmpeg_exec_path', help='ffmpeg_exec_path')
    parser.add_argument('src_name', help='src_name')
    parser.add_argument('width', type=int, help='width')
    parser.add_argument('height', type=int, help='height')
    args = parser.parse_args()

    vc = video_capture.VideoCapture(args.ffmpeg_exec_path,args.src_name,args.width,args.height)
    vc.start()
    vc.wait_data_ready()

    size = args.width, args.height
    black = 0,0,0

    img_surf = None
    screen = pygame.display.set_mode(size)
    
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

        if not run: break

        img = vc.get_frame()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        vc.release_frame()
        img = numpy.swapaxes(img,0,1)
        if img_surf == None:
            img_surf = pygame.pixelcopy.make_surface(img)
        else:
            pygame.pixelcopy.array_to_surface(img_surf,img)
            
        #print(img.shape)
        
        screen.fill(black)
        screen.blit(img_surf,(0,0))
        pygame.display.flip()

    vc.close()
