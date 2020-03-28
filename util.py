import threading


def run_with_timeout(func, args, timeout):
    """ Runs a function and returns None if it takes loner that `timeout` (in seconds) """
    result = [None]
    def worker():
        result[0] = func(*args)
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout)
    return result[0]

