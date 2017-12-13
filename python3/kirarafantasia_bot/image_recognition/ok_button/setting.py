# avoid tensorflow dependency
import numpy as np

ZOOM_FACTOR_XY = 3,3

SCAN_CROP_XYWH = 239,162,90,158
SCAN_BOUND_WH_LU = ((90,90),(24,24))
SCAN_STEP_XY = 3,3

TRAIN_CROP_XYWH = 237,160,92,160
TRAIN_BOUND_WH_LU = ((89,91),(23,25))
TRAIN_STEP_XY = 1,1

NAME = 'ok_button'

def should_ignore(img):
    IGNORE_STDDEV_MAX = 0.2
    IGNORE_DIFF_MAX   = 1

    img_diff = np.max(np.max(img,axis=(0,1))-np.min(img,axis=(0,1)))
    if img_diff > IGNORE_DIFF_MAX:
        return False
    img_stddev = np.max(np.std(img,axis=(0,1)))
    if img_stddev > IGNORE_STDDEV_MAX:
        return False
    return True
