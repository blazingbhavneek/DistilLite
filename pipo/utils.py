import inspect
import logging
import time


def log(message, type="info"):
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename
    func_name = frame.f_code.co_name

    filename = filename.split("/")[-1]

    source = f"{filename}: {func_name}"

    if type == "info":
        logging.info(f"{source} >> {message}")
    elif type == "warning":
        logging.warning(f"{source} >> {message}")
    elif type == "error":
        logging.error(f"{source} >> {message}")
    else:
        logging.debug(f"{source} >> {message}")


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        log(f"Function {func.__name__} executed in {elapsed_time:.4f} seconds")
        return result

    return wrapper
