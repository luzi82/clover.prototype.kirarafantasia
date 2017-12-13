import os
import numpy as np
import cv2
from functools import lru_cache
import csv
import shutil

def get_label_state_list():
    label_state_path = os.path.join('image_recognition','label','state')
    label_name_list = os.listdir(label_state_path)
    label_name_list = filter(lambda v:os.path.isfile(os.path.join(label_state_path,v)),label_name_list)
    label_name_list = filter(lambda v:v.endswith('.txt'),label_name_list)
    label_name_list = [ i[:-4] for i in label_name_list]
    label_name_list = sorted(label_name_list)
    return label_name_list

def load_img(fn):
    img = cv2.imread(fn).astype('float32')*2/255-1
    return img

def xy_layer(w,h):
    xx = np.array(list(range(w))).astype('float32')*2/(w-1)-1
    xx = np.tile(xx,h)
    xx = np.reshape(xx,(h,w,1))
    yy = np.array(list(range(h))).astype('float32')*2/(h-1)-1
    yy = np.repeat(yy,w)
    yy = np.reshape(yy,(h,w,1))
    xxyy = np.append(xx,yy,axis=2)
    return xxyy

def xy1_layer(w,h):
    ret = xy_layer(w,h)

    oo = np.ones(shape=(h,w,1),dtype=np.float)
    ret = np.append(ret,oo,axis=2)

    return ret

@lru_cache(maxsize=4)
def get_screen_sample_timestamp_list():
    screen_sample_timestamp_list = os.listdir(os.path.join('image_recognition','screen_sample'))
    screen_sample_timestamp_list = filter(lambda v:os.path.isdir(os.path.join('image_recognition','screen_sample',v)),screen_sample_timestamp_list)
    screen_sample_timestamp_list = [ int(i) for i in screen_sample_timestamp_list ]
    return screen_sample_timestamp_list

def get_timestamp(v):
    return max(filter(lambda i:i<v,get_screen_sample_timestamp_list()))

def create_bound_box_img_list(img,bound_box_list,dest_size):
    img_list = [img_bound_box_scale(img,bound_box,dest_size) for bound_box in bound_box_list]
    return img_list

# src_area:   area of bound box, (x,y,w,h), w/h would be factor of step
# size_range: size range of bound box, ((w0,w1),(h0,h1)), should be factor of step
# step:       base factor, shift step, resize step, (x,y)
def cal_bound_box_list(src_xywh,size_range_lu_wh,step_xy):
    ax,ay,aw,ah = src_xywh
    ((w0,w1),(h0,h1)) = size_range_lu_wh
    step_x,step_y = step_xy

    output_list = []
    for w,h in [(w,h) for w in range(w0,w1+1,step_x) for h in range(h0,h1+1,step_y)]:
        output_list += [(x,y,w,h) for x in range(ax,ax+aw+1-w,step_x) for y in range(ay,ay+ah+1-h,step_y)]
    
    return output_list

def img_bound_box_scale(img,bound_box,dest_size):
    bx,by,bw,bh = bound_box
    dw,dh = dest_size
    img = img[by:(by+bh),bx:(bx+bw),:]
    img = cv2.resize(img,dsize=(dw,dh),interpolation=cv2.INTER_AREA)
    return img

# (intercept_area/union_area)*2-1, min -1, max 1
def cal_bound_box_score(bound_box, answer_box):
    bxl,byl,bxu,byu = xywh_to_xyxy(bound_box)
    axl,ayl,axu,ayu = xywh_to_xyxy(answer_box)
    
    xl = max(bxl,axl)
    xu = min(bxu,axu)
    yl = max(byl,ayl)
    yu = min(byu,ayu)
    if(xl>=xu):
        return 0
    if(yl>=yu):
        return 0
    
    inner_size = (xu-xl)*(yu-yl)
    a_size = (axu-axl)*(ayu-ayl)
    b_size = (bxu-bxl)*(byu-byl)
    
    ratio = inner_size / (a_size+b_size-inner_size)
    
    return ratio*2 - 1

def xywh_to_xyxy(xywh):
    return xywh[0],xywh[1],xywh[0]+xywh[2],xywh[1]+xywh[3]

#def crop_img(img, xywh):
#    x0,y0,x1,y1 = xywh_to_xyxy(xywh)
#    return img[y0:y1][x0:x1][]
