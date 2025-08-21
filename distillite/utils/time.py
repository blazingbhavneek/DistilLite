import time

from distillite.utils import log


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        log(f"Function {func.__name__} executed in {elapsed_time:.4f} seconds")
        return result

    return wrapper
