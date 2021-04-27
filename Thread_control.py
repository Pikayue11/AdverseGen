# import threading
#
# class OurThread (threading.Thread):
#     def __init__(self, threadID, name, func, args):
#         threading.Thread.__init__(self)
#         self.threadID = threadID
#         self.name = name
#         self.func = func
#         self.args = args
#         self.retValue = 0
#     def run(self):
#         # print ("Thread Start: " + self.name)
#         self.func(*self.args)
#         # print ("Thread Start: " + self.name)
#
#

import ctypes
import inspect
def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)
