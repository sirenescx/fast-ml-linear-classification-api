import logging

from fast_ml_linear_regression_models.services.utils.file_utils import get_filepath


def get_logger(log_file_directory: str) -> logging.Logger:
    file_handler = logging.FileHandler(get_filepath(log_file_directory, "log"), "a")
    formatter = logging.Formatter("%(levelname)s %(asctime)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger
