import threading
import sys
import copy

BUFFER_COUNT = 3

class AsyncReadWriteJudge:

    def __init__(self,lock=None):
        if lock == None:
            lock = threading.Lock()
        self._lock = lock
        self._read_lock_idx = None
        self._write_lock_idx = None
        self._next_order_idx = 1
        self._order_idx_list = [0] * BUFFER_COUNT

    def get_write_idx(self):
        with self._lock:
            assert(self._write_lock_idx == None)
            tmp_order_idx_list = copy.copy(self._order_idx_list)
            if self._read_lock_idx != None:
                tmp_order_idx_list[self._read_lock_idx] = sys.float_info.max
            idx = tmp_order_idx_list.index(min(tmp_order_idx_list))
            self._write_lock_idx = idx
            self._order_idx_list[idx] = self._next_order_idx
            self._next_order_idx += 1
            return idx

    def release_write_idx(self):
        with self._lock:
            assert(self._write_lock_idx != None)
            self._write_lock_idx = None

    def get_read_idx(self):
        with self._lock:
            assert(self._read_lock_idx == None)
            tmp_order_idx_list = copy.copy(self._order_idx_list)
            if self._write_lock_idx != None:
                tmp_order_idx_list[self._write_lock_idx] = -1
            max_order_idx = max(tmp_order_idx_list)
            idx = tmp_order_idx_list.index(max_order_idx)
            self._read_lock_idx = idx
            return idx

    def release_read_idx(self):
        with self._lock:
            assert(self._read_lock_idx != None)
            self._read_lock_idx = None

    def get_order_idx(self, idx):
        return self._order_idx_list[idx]
