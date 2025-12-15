import threading
import time

def set_interval(func, interval):
    """
    类似 JS 的 setInterval
    :param func: 要执行的方法
    :param interval: 秒
    :return: stop 函数
    """
    stopped = threading.Event()

    def loop():
        while not stopped.wait(interval):
            func()

    threading.Thread(target=loop, daemon=True).start()

    def stop():
        stopped.set()

    return stop
