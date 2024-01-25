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
from src.models import FNN, AlexNet, Standard_VIT
from src.optuna_tools import study_properties

logger = logging.getLogger(__name__)

# DEFAULT PARAMS
MAX_EPOCHS: int = 40
LOSS_FN = CrossEntropyLoss
MODELS = {"FNN": FNN, "AlexNet": AlexNet, "VIT": Standard_VIT }
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
    parser = argparse.ArgumentParser(description="Description of your experiment.")
    parser.add_argument(
        "--MODEL_NAME", type=str, help=f"Pick one model out of {MODELS.items()}"
    )
    parser.add_argument(
        "--DATASET_NAME",
        type=str,
        help=f"Pick one model out of {DATASET_LOADERS.items()}",
    )
    parser.add_argument("--READ_HP", action="store_true")
    parser.add_argument(
        "--LR",
        type=float,
        default=LR,
        help="If true optimal HP are read from the tuning step, if false provided HPs are read",
    )
    parser.add_argument(
        "--BATCH_SIZE",
        type=int,
        default=BATCH_SIZE,
        help="If true optimal HP are read from the tuning step, if false provided HPs are read",
    )
    parser.add_argument(
        "--OPTIMIZER",
        type=str,
        default=OPTIMIZER,
        help="If true optimal HP are read from the tuning step, if false provided HPs are read",
    )
    parser.add_argument(
        "--MAX_EPOCHS", type=int, default=MAX_EPOCHS, help="Description of MAX_EPOCHS"
    )

    args = parser.parse_args()
    train_dataset, val_dataset, test_dataset = DATASET_LOADERS[args.DATASET_NAME]()
    num_channels = train_dataset.dataset[0][0].shape[0]
    input_size = train_dataset.dataset[0][0].shape[1]
    num_classes = len(set(data_to_list(train_dataset.dataset.targets)))
    logger.info(f"Number of classes:{num_classes}")

    if args.READ_HP is True:
        logger.info("Reading best hyperparams from earlier Optuna study")
        trial = study_properties(
            f"{models_path}/{args.DATASET_NAME}/{args.MODEL_NAME}_model/study.pickle"
        )
        lr = trial.params["lr"]
        dropout_prob = trial.params["dropout_prob"]
        optimizer_name = trial.params["optimizer"]
        batch_size = trial.params["batch_size"]

    elif args.READ_HP is False:
        logger.info("Using hyperparams passed to the program")
        lr = args.LR
        batch_size = args.BATCH_SIZE
        optimizer_name = args.OPTIMIZER
        dropout_prob = args.DROPOUT_PROB

    model = MODELS[args.MODEL_NAME](
        input_size=input_size,
        num_channels=num_channels,
        num_classes=num_classes,
        dropout_prob=dropout_prob,
    ).to(device)
    optimizer = getattr(optim, optimizer_name)(
        model.parameters(), lr=lr, weight_decay=0.001
    )
    train_loader = DataLoader(train_dataset, batch_size)
    val_loader = DataLoader(val_dataset, batch_size)
    test_loader = DataLoader(test_dataset, batch_size)

    train_and_save(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_name=args.MODEL_NAME,
        dataset_name=args.DATASET_NAME,
        input_model=model,
        loss_fn=LOSS_FN,
        optimizer=optimizer,
        max_epochs=args.MAX_EPOCHS,
        models_path=f"{models_path}/{args.DATASET_NAME}",
    )
