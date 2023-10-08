import logging
import os
from glob import glob

def get_logger(model_name, path='./mypath/{}.log'):
    # os.makedirs('./logging', exist_ok=True)
    # logging
    logger = logging.getLogger('mylogger')
    fomatter = logging.Formatter(fmt='%(asctime)s > %(message)s', datefmt='[%d-%m-%Y %H:%M:%S]')
    # open handler
    find_log = glob(path.format('*'))
    log_path = find_log[0] if len(find_log) else  path.format(model_name)

    fileHandler = logging.FileHandler(log_path)
    streamHandler = logging.StreamHandler()
    # set formatter
    fileHandler.setFormatter(fomatter)
    streamHandler.setFormatter(fomatter)
    # add to logger
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    # logging
    logger.setLevel(logging.DEBUG)
    logger.info("Logger initialized !")
    return logger