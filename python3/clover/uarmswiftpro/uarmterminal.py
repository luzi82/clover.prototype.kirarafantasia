import serial
import threading
import sys

from . import list_uarms

class UArmTerminal:

    def __init__(self):
        self.on_close = False
    
    def connect(self,port=None):
        try:
            if port == None:
                uarm_port_list = list_uarms.uarm_ports()
                assert(len(uarm_port_list)==1)
                port = uarm_port_list[0]
            self.port = port
            self.ser = serial.Serial(self.port, 115200, timeout=1)
        except Exception as e:
            print("Error URBEULCICT {}".format(str(e)));

    def _listen_loop(self):
        try:
            while self.enable_listen_loop:
                line = str(self.ser.readline().strip().decode('ascii'))
                if(len(line)>0):
                    self.listen_handler(line)
        except Exception as e:
            if self.on_close:
                return
            raise e

    def start_listen(self,handler):
        self.listen_handler = handler
        self.enable_listen_loop = True
        self.listen_thread = threading.Thread( target=self._listen_loop )
        self.listen_thread.start()

    def stop_listen(self):
        self.enable_listen_loop = False
        self.listen_thread.join()

    def send_cmd(self,cmd):
        self.ser.write(cmd.encode('ascii'))
    
    def close(self):
        self.on_close = True
        self.ser.close()
        self.stop_listen()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='uArm signal terminal')
    parser.add_argument('port', nargs='?', help='port')
    args = parser.parse_args()
    
    port = args.port

    uat = UArmTerminal()
    uat.connect(port)
    uat.start_listen(lambda line:print(line))
    try:
        for line in sys.stdin:
            uat.send_cmd(line)
    except KeyboardInterrupt:
        pass
    uat.close()

