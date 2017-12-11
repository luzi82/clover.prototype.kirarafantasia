import pygame
import functools

@functools.lru_cache()
def _font():
    return pygame.font.SysFont("monospace", 15)

@functools.lru_cache(maxsize=1024)
def text(ttext, color):
    return _font().render(ttext,1,color)
