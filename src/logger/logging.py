import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"  # file name formating
log_directory = os.path.join("src/logger", "logs")  # path for logs folder
os.makedirs(log_directory, exist_ok=True)  # creates directory if not exist

LOG_FILE_PATH = os.path.join(log_directory, LOG_FILE)  # joins the path with the file name

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
