import argparse
import logging
import os
import sys
from datetime import datetime

import torch
import yaml
from torch.nn import CrossEntropyLoss

from experiments.tools import tune_model
from src.MLtools import get_CIFAR_data, get_mnist_data
from src.models import FNN, AlexNet, Standard_VIT

logger = logging.getLogger(__name__)

# DEFAULT PARAMS TUNING
MAX_EPOCHS: int = 20
N_TRIALS: int = 20
HB_MIN_RESOURCE: int = 3
HB_MAX_RESOURCE: int = 20
HB_REDUCTION_FACTOR: int = 3
LOSS_FN = CrossEntropyLoss
MODELS = {"FNN": FNN, "AlexNet": AlexNet, "VIT": Standard_VIT}
DATASET_LOADERS = {"MNIST": get_mnist_data, "CIFAR100": get_CIFAR_data}
logger.info(sys.argv)
logger.info(
    f"Running {__file__} with default parameters:  MAX_EPOCHS=%s, N_TRIALS=%s, HB_MIN_RESOURCE=%s, HB_MAX_RESOURCE=%s,HB_REDUCTION_FACTOR=%s,  LOSS_FN=%s",
    MAX_EPOCHS,
    N_TRIALS,
    HB_MIN_RESOURCE,
    HB_MAX_RESOURCE,
    HB_REDUCTION_FACTOR,
    LOSS_FN,
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
    parser = argparse.ArgumentParser(description="Description of your experiment.")
    parser.add_argument(
        "--MODEL_NAME", type=str, help=f"Pick one model out of {MODELS.items()}"
    )
    parser.add_argument(
        "--DATASET_NAME",
        type=str,
        help=f"Pick one model out of {DATASET_LOADERS.items()}",
    )
    parser.add_argument(
        "--MAX_EPOCHS", type=int, default=MAX_EPOCHS, help="Description of MAX_EPOCHS"
    )
    parser.add_argument(
        "--N_TRIALS", type=int, default=N_TRIALS, help="Description of N_TRIALS"
    )
    parser.add_argument(
        "--HB_MIN_RESOURCE",
        type=int,
        default=HB_MIN_RESOURCE,
        help="Description of PATIENCE",
    )
    parser.add_argument(
        "--HB_MAX_RESOURCE",
        type=float,
        default=HB_MAX_RESOURCE,
        help="Description of MIN_DELTA",
    )
    parser.add_argument(
        "--HB_REDUCTION_FACTOR",
        type=float,
        default=HB_REDUCTION_FACTOR,
        help="Description of MIN_DELTA",
    )

    args = parser.parse_args()
    train_dataset, val_dataset, test_dataset = DATASET_LOADERS[args.DATASET_NAME]()

    input_model = MODELS[args.MODEL_NAME]

    tune_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_name=args.MODEL_NAME,
        input_model=input_model,
        loss_fn=LOSS_FN,
        max_epochs=args.MAX_EPOCHS,
        n_trials=args.N_TRIALS,
        hyperband_min_resource=args.HB_MIN_RESOURCE,
        hyperband_max_resource=args.HB_MAX_RESOURCE,
        hyperband_reduction_factor=args.HB_REDUCTION_FACTOR,
        model_path=f"{models_path}/{args.DATASET_NAME}",
    )
