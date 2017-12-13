import serial
import threading
import sys
import re
import time

from . import list_uarms
from . import uarmterminal

CMD_IDX_MIN = 1
CMD_IDX_MAX = 100

class UArmAsync:

    def __init__(self):
        self.next_cmd_idx = CMD_IDX_MIN
        self.cmd_future_dict = {}
        self.lock = threading.Lock()
        self.uat = uarmterminal.UArmTerminal()
        self.ready = False
        self.on_msg = None
    
    def connect(self,port=None):
        self.uat.connect(port)
        self.uat.start_listen(self._on_msg)

    def close(self):
        self.uat.close()

    def send_cmd(self,cmd):
        cmd_idx = None
        with self.lock:
            while True:
                cmd_idx = self.next_cmd_idx
                self.next_cmd_idx += 1
                if self.next_cmd_idx > CMD_IDX_MAX:
                    self.next_cmd_idx = CMD_IDX_MIN
                if cmd_idx not in self.cmd_future_dict:
                    break
            cmd_future = self.cmd_future_dict[cmd_idx] = _UArmAsyncCmdFuture(self, cmd_idx)
            cmd_future.state = 'busy'
        ccmd = "#{} {}\n".format(cmd_idx,cmd)
        print('<<{:.3f}: {}'.format(time.time(),ccmd.strip()),file=sys.stderr)
        self.uat.send_cmd(ccmd)
        return cmd_future

    def _on_msg(self,line):
        print('>>{:.3f}: {}'.format(time.time(),line),file=sys.stderr)
        if line == '@5 V1':
            with self.lock:
                self.ready = True
        m = re.fullmatch('^\\$(\\d+) (.+)$',line)
        if m:
            cmd_idx = int(m.group(1))
            with self.lock:
                with self.cmd_future_dict[cmd_idx].cond:
                    self.cmd_future_dict[cmd_idx].state = 'done'
                    self.cmd_future_dict[cmd_idx].result = m.group(2)
                    self.cmd_future_dict[cmd_idx].cond.notify_all()
                    self.cmd_future_dict.pop(cmd_idx)
        if self.on_msg:
            self.on_msg(line)
    
    def wait_ready(self):
        while True:
            with self.lock:
                if self.ready:
                    break
            time.sleep(0.01)

class _UArmAsyncCmdFuture:

    def __init__(self, uarm_async, cmd_id):
        self.uarm_async = uarm_async
        self.cmd_id = cmd_id
        self.state = None
        self.result = None
        self.cond = threading.Condition()
    
    def wait(self):
        with self.cond:
            self.cond.wait_for(lambda: not self.is_busy())
        return self.result

    def is_busy(self):
        return self.state == 'busy'

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='uArm signal terminal')
    parser.add_argument('port', nargs='?', help='port')
    args = parser.parse_args()
    
    port = args.port

    uat = UArmAsync()
    uat.connect(port)
    try:
        for line in sys.stdin:
            line = line.strip()
            future = uat.send_cmd(line)
            print(future.wait())
    except KeyboardInterrupt:
        pass
    uat.close()

