# avoid tensorflow dependency
import numpy as np

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
