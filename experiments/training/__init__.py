import logging
import os
import sys
from datetime import datetime

# def parse_arguments():
#     parser = argparse.ArgumentParser(description="Your description here")
#     parser.add_argument("--DATASET_NAME", type=str, default="default_value", help="Log parameter description")
#     parser.add_argument("--MODEL_NAME", type=str, default="default_value", help="Log parameter description")
#     # Add other argparse parameters as needed
#     return parser.parse_args()

# args = parse_arguments()


log_dir = "logs"
log_file_name = f'final_training_{datetime.now().strftime("%Y%m%d%H%M%S")}.log'
# log_file_name = f'{args.DATASET_NAME}_{args.MODEL_NAME}_final_training_log.log'


fh = logging.FileHandler(f"logs/{log_file_name}")

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler(os.path.join(log_dir, log_file_name)),  # Log to file
    ],
)
