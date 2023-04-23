from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger

OUTPUT_DIR = './'


def init_logger(log_file=OUTPUT_DIR + "xgboost.log"):

    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


logger = init_logger()
