import queue
import threading
import time
import traceback
import sys
from collections import deque

from . import uarm_async

CMD_IDX_MIN = 1
CMD_IDX_MAX = 10000

SLOT_COUNT = 2

class UArmQueue:

    def __init__(self, lock=None):
        if lock == None:
            lock = threading.RLock()
        
        self.lock = lock
        self.uaa = uarm_async.UArmAsync()
        self.cmd_queue = queue.Queue()
        self.next_cmd_idx = CMD_IDX_MIN
        self.loop_thread = None

    def connect(self,port=None):
        self.uaa.connect(port)
        self.loop_thread = threading.Thread(target=self._loop)
        self.loop_thread.start()

    def close(self):
        self.cmd_queue.put({'type':'close'})
        if self.loop_thread != None:
            self.loop_thread.join()
        self.uaa.close()

    def wait_ready(self):
        return self.uaa.wait_ready()

    def set_on_msg(self,on_msg):
        self.uaa.on_msg = on_msg

    def _get_next_cmd_idx(self):
        cmd_idx = self.next_cmd_idx
        self.next_cmd_idx += 1
        if self.next_cmd_idx > CMD_IDX_MAX:
            self.next_cmd_idx = CMD_IDX_MIN
        return cmd_idx

    def send_cmd(self,cmd):
        cmd_idx = self._get_next_cmd_idx()
        cmd_unit = {
            'type': 'cmd',
            'idx': cmd_idx,
            'cmd': cmd,
            'future': None,
            'uaa_future': None
        }
        future = _Future(self, cmd_unit)
        cmd_unit['future'] = future
        self.cmd_queue.put(cmd_unit)
        return future

    def _loop(self):
        try:
            busy_queue = deque()
            next_cmd_time = 0 # uarm easy die if cmd go too fast
            while(True):
                unit = self.cmd_queue.get(block=True)
                if unit['type'] == 'close':
                    return
                if unit['type'] == 'cmd':
                    while len(busy_queue) >= 2:
                        wait_unit = busy_queue.popleft()
                        wait_unit['future'].wait()
                    busy_queue.append(unit)
                    now_time = time.time()
                    #printf('URTSGFSDLS now_time:{:.3f}, next_cmd_time:{:.3f}'.format(now_time,next_cmd_time),file=sys.stderr)
                    if now_time < next_cmd_time:
                        time.sleep(next_cmd_time-now_time)
                        now_time = next_cmd_time
                    with unit['future'].cond:
                        unit['uaa_future'] = self.uaa.send_cmd(unit['cmd'])
                        unit['future'].cond.notify_all()
                    next_cmd_time = now_time + 0.1
        except:
            traceback.print_exc()

    def wait_ready(self):
        self.uaa.wait_ready()
    
class _Future:
    
    def __init__(self,uq,unit):
        self.uq = uq
        self.unit = unit
        self.cond = threading.Condition()
    
    def wait(self):
        with self.cond:
            self.cond.wait_for(lambda: self.unit['uaa_future'] != None)
        return self.unit['uaa_future'].wait()
    
    def is_busy(self):
        if self.unit['uaa_future'] == None:
            return True
        return self.unit['uaa_future'].is_busy()
