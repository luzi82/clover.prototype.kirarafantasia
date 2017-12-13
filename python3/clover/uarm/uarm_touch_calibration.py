import touch_trace
import video_capture
import uarm
import time
import json
import screen_arm_position

def get_touchtracedata(windowid,filename):
    screen_capture.screencapture(windowid,filename)
    data = touch_trace.imgfile_to_touchtracedata(filename)
    return data

class _UArmTouchCalibrationUtil:

    def __init__(self):
        self.arm = None
        self.up_z = None
        self.down_z = None
        self.windowid = None
        self.tmp_file = None
    
    def jot(self,arm_x,arm_y,delay=0.3):
        self.arm.set_position(arm_x,arm_y,self.up_z,1000).wait()
        self.arm.set_position(arm_x,arm_y,self.down_z,1000).wait()
        self.arm.wait_stop()
        time.sleep(delay)
        ttd = get_touchtracedata(self.windowid,self.tmp_file)
        self.arm.set_position(arm_x,arm_y,self.up_z,1000).wait()
        if not ttd['down']:
            return None
        point_data = {
            'screen_x':ttd['x'],'screen_y':ttd['y'],
            'arm_x':arm_x,'arm_y':arm_y
        }
        return point_data

    def jot_screen(self,screen_x,screen_y,sap,delay=0.2):
        arm_x, arm_y = sap.screen_to_arm(screen_x,screen_y)
        return self.jot(arm_x,arm_y,delay)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='get calibrate data')
    parser.add_argument('x_count', type=int, help='x_count')
    parser.add_argument('y_count', type=int, help='y_count')

    parser.add_argument('init_step', type=int, help='init_step')

    parser.add_argument('down', type=float, help='down')
    parser.add_argument('up', type=float, help='up')

    parser.add_argument('repeat', type=int, help='repeat')
    parser.add_argument('border_in', type=int, help='border_in')
    parser.add_argument('border_out', type=int, help='border_out')

    parser.add_argument('tmp_file', help='tmp_file')
    parser.add_argument('output_file', help='output_file')
    args = parser.parse_args()
    
    assert(args.x_count>=2)
    assert(args.y_count>=2)
    
    windowid = screen_capture.get_windowid()
    ttd = get_touchtracedata(windowid,args.tmp_file)
    screen_width = ttd['width']
    screen_height = ttd['height']

    arm = uarm.UArm()
    arm.connect()
    arm.wait_ready()
    
    for i in range(3):
        arm.detach_servo(i).wait()
    
    input('Prepare')

    for i in range(3):
        arm.attach_servo(i).wait()
    arm.set_user_mode(0).wait()

    time.sleep(3)
    ori_position = arm.get_position().wait()
    #print(ori_position)
    
    down_z = ori_position[2] + args.down
    up_z   = ori_position[2] + args.up

    arm.set_position(ori_position[0],ori_position[1],up_z,1000).wait()

    uatcu = _UArmTouchCalibrationUtil()
    uatcu.arm = arm
    uatcu.up_z = up_z
    uatcu.down_z = down_z
    uatcu.windowid = windowid
    uatcu.tmp_file = args.tmp_file

    # stage0
    stage0 = {}
    stage0['point_list'] = []

    point_data = uatcu.jot(ori_position[0]-args.init_step,ori_position[1]-args.init_step)
    assert(point_data!=None)
    stage0['point_list'].append(point_data)
    point_data = uatcu.jot(ori_position[0]-args.init_step,ori_position[1]+args.init_step)
    assert(point_data!=None)
    stage0['point_list'].append(point_data)
    point_data = uatcu.jot(ori_position[0]+args.init_step,ori_position[1])
    assert(point_data!=None)
    stage0['point_list'].append(point_data)
    
    sap0 = screen_arm_position.ScreenArmPosition(filename=None,jjson=stage0)

    # stage1

    stage1 = {}
    stage1['point_list'] = []
    
    point_data = uatcu.jot_screen(args.border_in,args.border_in,sap0)
    if point_data:
        stage1['point_list'].append(point_data)
    point_data = uatcu.jot_screen(args.border_in,screen_height-args.border_in,sap0)
    if point_data:
        stage1['point_list'].append(point_data)
    point_data = uatcu.jot_screen(screen_width-args.border_in,args.border_in,sap0)
    if point_data:
        stage1['point_list'].append(point_data)
    point_data = uatcu.jot_screen(screen_width-args.border_in,screen_height-args.border_in,sap0)
    if point_data:
        stage1['point_list'].append(point_data)

    assert(len(stage1['point_list'])>=3)
    
    sap1 = screen_arm_position.ScreenArmPosition(filename=None,jjson=stage1)

    output = {}
    output['point_list'] = []

    for test_y in range(args.y_count):
        for test_x in range(args.x_count):
            sx0 = args.border_in + (screen_width -args.border_in-args.border_in)*test_x/(args.x_count-1)
            sy0 = args.border_in + (screen_height-args.border_in-args.border_in)*test_y/(args.y_count-1)
            arm_x,arm_y = sap1.screen_to_arm(sx0,sy0)
            
            point_data = uatcu.jot(arm_x,arm_y,0)
            if point_data == None:
                continue
            
            good = True
            screen_x = 0
            screen_y = 0
            for _ in range(args.repeat):
                point_data = uatcu.jot(arm_x,arm_y)
                good = point_data != None
                if not good:
                    break
                screen_x += point_data['screen_x']
                screen_y += point_data['screen_y']
            screen_x /= args.repeat
            screen_y /= args.repeat
            if screen_x < args.border_out:
                good = False
            if screen_y < args.border_out:
                good = False
            if screen_x >= screen_width - args.border_out:
                good = False
            if screen_y >= screen_height - args.border_out:
                good = False
            if good:
                #print('{},{}'.format(screen_x/args.repeat,screen_y/args.repeat))
                point_data = {
                    'screen_x':screen_x,'screen_y':screen_y,
                    'arm_x':arm_x,'arm_y':arm_y
                }
                output['point_list'].append(point_data)
    arm.set_position(ori_position[0],ori_position[1],up_z,1000).wait()
    arm.set_position(ori_position[0],ori_position[1],down_z,1000).wait()
    arm.wait_stop()
    arm.close()
    
    with open(args.output_file, 'w') as outfile:
        json.dump(output, outfile)
