# avoid tensorflow dependency
import numpy as np

ZOOM_FACTOR_XY = 3,3

SCAN_CROP_XYWH = 239,158,90,162
SCAN_BOUND_WH_LU = ((30,30),(9,12))
SCAN_STEP_XY = 3,3

TRAIN_CROP_XYWH = 230,158,108,162
TRAIN_BOUND_WH_LU = ((30,33),(9,12))
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
