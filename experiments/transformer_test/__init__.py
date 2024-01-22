import logging
import os
import sys
from datetime import datetime


log_dir = "logs"
log_file_name = f'transformer_exploration_{datetime.now().strftime("%Y%m%d%H%M%S")}.log'


fh = logging.FileHandler(f"logs/{log_file_name}")

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler(os.path.join(log_dir, log_file_name)),  # Log to file
    ],
)
