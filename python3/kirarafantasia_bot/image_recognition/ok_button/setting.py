# avoid tensorflow dependency
import numpy as np

NAME = 'ok_button'

ZOOM_FACTOR_XY = 3,3

SCAN_CROP_XYWH = 239,162,90,158
SCAN_BOUND_LU_WH = ((90,90),(24,24))
SCAN_STEP_XY = 3,3

TRAIN_CROP_XYWH = 237,160,92,160
TRAIN_BOUND_LU_WH = ((89,91),(23,25))
TRAIN_STEP_XY = 1,1
