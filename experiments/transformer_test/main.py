import argparse
import logging
import os
from datetime import datetime

import torch
import yaml
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from experiments.tools import data_to_list, train_and_save
from src.MLtools import get_CIFAR_data, get_mnist_data
from src.models import FNN, AlexNet
from src.optuna_tools import study_properties

logger = logging.getLogger(__name__)

# DEFAULT PARAMS
MAX_EPOCHS: int = 40
LOSS_FN = CrossEntropyLoss
MODELS = {"FNN": FNN, "AlexNet": AlexNet}
LR = 0.0001
BATCH_SIZE = 32
OPTIMIZER = "Adam"
DROPOUT_PROB = 0.25
DATASET_LOADERS = {"MNIST": get_mnist_data, "CIFAR100": get_CIFAR_data}
logger.info(
    f"Running {__file__} with default parameters: MAX_EPOCHS=%s, LOSS_FN=%s, LR=%s, BATCH_SIZE=%s,OPTIMIZER=%s",
    MAX_EPOCHS,
    LOSS_FN,
    LR,
    BATCH_SIZE,
    OPTIMIZER,
)

# IMPORT REPO PATHS
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

models_path = config["PATHS"]["models_path"]

# CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device {device}")
if device.type == "cuda":
    logger.info(torch.cuda.get_device_name(0))





if __name__ == "__main__":

    train_dataset, val_dataset, test_dataset=get_CIFAR_data()
