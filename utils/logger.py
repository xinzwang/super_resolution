import logging


def build_logger(log_path):
	logger = logging.getLogger(__name__)
	logger.setLevel(level = logging.INFO)
	handler = logging.FileHandler(log_path)
	handler.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s [%(levelname)s]  %(message)s')
	handler.setFormatter(formatter)
	logger.addHandler(handler)
	return logger