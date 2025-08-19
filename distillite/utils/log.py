import inspect
import logging


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
