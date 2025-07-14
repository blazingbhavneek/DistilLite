import logging
import time

from pipo.utils import log, measure_time


def main():
    log("Hello from distillite!")
    measure_time(time.sleep)(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
