import threading
import ctypes
import os
import sys


class StopThreadException(BaseException):
    def __init__(self):
        BaseException.__init__(self)


class TimedOutException(Exception):
    def __init__(self):
        Exception.__init__(self)


class StoppableThread(threading.Thread):
    def stop(self):
        if not self.is_alive():
            return

        self._stderr = open(os.devnull, 'w')
        self._Thread__stderr  = self._stderr
        exception = type("StopThreadException", StopThreadException.__bases__, dict(StopThreadException.__dict__))
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(self.ident), ctypes.py_object(exception))


def run_with_timeout(func, args, timeout):
    """ Runs a function and returns None if it takes loner that `timeout` (in seconds) """
    result = [None]
    exception = []

    def worker():
        try:
            result[0] = func(*args)
        except StopThreadException:
            pass
        #except Exception as e:
        #    exc_info = sys.exc_info()
        #    # Assemble the alternate traceback, excluding this function from the trace (by going to next frame)
        #    e.__traceback__ = exc_info[2].tb_next
        #       exception.append(e)

    thread = StoppableThread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        thread.stop()
        print("Killed")
        thread.join(0.1)
        if thread.is_alive():
            print("HILFE")
            thread.join()
        try:
            thread._stderr.close()
        except:
            pass
    return result[0]
