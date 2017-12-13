import re
import time

from . import uarm_queue

class UArm:

    def __init__(self):
        self._on_report_position = None
        self.last_report_position = None

        self.uaa = uarm_queue.UArmQueue()
        self.uaa.set_on_msg(self._uaa_on_msg)

    def connect(self,port=None):
        self.uaa.connect(port)

    def close(self):
        self.uaa.close()

    def wait_ready(self):
        return self.uaa.wait_ready()

    def set_position(self,x,y,z,f):
        cmd = 'G0 X{:.2f} Y{:.2f} Z{:.2f} F{}'.format(x,y,z,f)
        uaacf = self.uaa.send_cmd(cmd)
        return _UArmFuture(uaacf,lambda line:line=='ok')

    def get_position(self):
        cmd = 'P2220'
        uaacf = self.uaa.send_cmd(cmd)
        return _UArmFuture(uaacf,get_position_func)

    def set_acceleration(self,p,t):
        cmd = 'M204 P{} T{}'.format(p,t)
        uaacf = self.uaa.send_cmd(cmd)
        return _UArmFuture(uaacf,lambda line:line=='ok')

    def attach_servo(self, servo_id):
        cmd = 'M2201 N{}'.format(servo_id)
        uaacf = self.uaa.send_cmd(cmd)
        return _UArmFuture(uaacf,lambda line:line=='ok')

    def detach_servo(self, servo_id):
        cmd = 'M2202 N{}'.format(servo_id)
        uaacf = self.uaa.send_cmd(cmd)
        return _UArmFuture(uaacf,lambda line:line=='ok')

    def set_user_mode(self,mode_id):
        cmd = 'M2400 S{}'.format(mode_id)
        uaacf = self.uaa.send_cmd(cmd)
        return _UArmFuture(uaacf,lambda line:line=='ok')

    def get_moving(self):
        cmd = 'M2200'
        uaacf = self.uaa.send_cmd(cmd)
        return _UArmFuture(uaacf,get_moving_func)

    def set_report_position(self, enable):
        cmd = 'M2120 V{}'.format(1 if enable else 0)
        uaacf = self.uaa.send_cmd(cmd)
        return _UArmFuture(uaacf,lambda line:line=='ok')

    def set_on_report_position(self, on_report_position):
        self._on_report_position = on_report_position

    def wait_stop(self):
        while(self.get_moving().wait()):
            time.sleep(0.01)

    def _uaa_on_msg(self, line):
        m = re.fullmatch('@3 X(\\S+) Y(\\S+) Z(\\S+) R(\\S+)',line)
        if m:
            self.last_report_position = (float(m.group(1)),float(m.group(2)),float(m.group(3)),float(m.group(4)))
            if self._on_report_position:
                self._on_report_position(self.last_report_position)

def get_position_func(line):
    m = re.fullmatch('ok X(\\S+) Y(\\S+) Z(\\S+)',line)
    if m:
        return float(m.group(1)),float(m.group(2)),float(m.group(3))
    return None

def get_moving_func(line):
    m = re.fullmatch('ok V(\\S+)',line)
    if m:
        return m.group(1) == '1'
    return None

class _UArmFuture:

    def __init__(self, uaacf, func):
        self.uaacf = uaacf
        self.func = func
    
    def wait(self):
        ret = self.uaacf.wait()
        return self.func(ret)

    def is_busy(self):
        return self.uaacf.is_busy()
