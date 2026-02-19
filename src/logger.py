from datetime import datetime
from logging.handlers import RotatingFileHandler
import sys
import os
import logging

LOG_DIR = "logs"
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
MAX_LOG_SIZE  = 5*1024*1024
BACKUP_COUNT = 3

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
log_dir_path = os.path.join(root_dir, LOG_DIR)
os.makedirs(log_dir_path, exist_ok = True)
log_file_path = os.path.join(log_dir_path, LOG_FILE)

def configure_logger():
    """"Configure logging with rotating file and console handler"""

    # create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Define Formatter
    formatter = logging.Formatter("[%(asctime)s] - %(name)s - %(levelname)s - %(message)s")

    # File Handler with rotation
    file_handler = RotatingFileHandler(log_file_path, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT, encoding = 'utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Adding handlers to logging 
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Configure with logger
configure_logger()