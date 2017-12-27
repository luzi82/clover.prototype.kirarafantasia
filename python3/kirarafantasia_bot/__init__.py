VIDEO_SIZE = 568, 320
TOUCH_SIZE = 1136, 640

def btn_xy(x,y,w,h):
    return ((x+(w/2))*TOUCH_SIZE[0]/VIDEO_SIZE[0], (y+(h/2))*TOUCH_SIZE[1]/VIDEO_SIZE[1])
