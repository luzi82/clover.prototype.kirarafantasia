import cv2

def imgfile_to_touchtracedata(fn):
    img = cv2.imread(fn)
    return img_to_touchtracedata(img)

def img_to_touchtracedata(img):
    imgc   = img[:round(img.shape[1]*9/16),:,:]
    imgcg  = cv2.cvtColor(imgc, cv2.COLOR_BGR2GRAY)
    imgcgs = cv2.resize(imgcg,dsize=(16,9),interpolation = cv2.INTER_AREA)
    time64 = ( imgline_to_int(imgcgs[5]) <<  0 ) | \
             ( imgline_to_int(imgcgs[6]) << 16 ) | \
             ( imgline_to_int(imgcgs[7]) << 32 ) | \
             ( imgline_to_int(imgcgs[8]) << 48 )
    ret = {
        'width':imgline_to_int(imgcgs[0]),
        'height':imgline_to_int(imgcgs[1]),
        'x':imgline_to_int(imgcgs[2]),
        'y':imgline_to_int(imgcgs[3]),
        'down':(imgline_to_int(imgcgs[4])>0),
        'time':time64
    }
    return ret
    
def imgline_to_int(imgline):
    ret = 0
    i = 1
    for p in imgline:
        if p >=127:
            ret+=i
        i<<=1
    return ret

if __name__ == '__main__':
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='touch trace screenshot to data')
    parser.add_argument('filename', help='filename')
    args = parser.parse_args()

    touchtracedata = imgfile_to_touchtracedata(args.filename)
    print(json.dumps(touchtracedata))
