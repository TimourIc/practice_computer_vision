import argparse
import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import yaml
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from experiments.tools import data_to_list, train_and_save
from src.MLtools import get_CIFAR_data, get_mnist_data, train_full, test, model_params
from src.models import FNN, AlexNet, VisionTransformer
from src.optuna_tools import study_properties

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("VIT_CIFAR")

logger = logging.getLogger(__name__)

# DEFAULT PARAMS
MAX_EPOCHS: int = 40
LOSS_FN = CrossEntropyLoss
MODELS = {"FNN": FNN, "AlexNet": AlexNet}
LR = 0.005
BATCH_SIZE = 128
OPTIMIZER = "Adam"
DROPOUT_PROB = 0.1
WEIGHT_DECAY=0.001

#TRANSFORMER PARAMS
EMBED_DIM=256
HIDDEN_DIM=512
NUM_HEADS=8
NUM_LAYERS=6

logger.info(
    f"Running {__file__} with default parameters: MAX_EPOCHS=%s, LOSS_FN=%s, LR=%s, BATCH_SIZE=%s,OPTIMIZER=%s",
    MAX_EPOCHS,
    LOSS_FN,
    LR,
    BATCH_SIZE,
    OPTIMIZER,
)

#HYPERPARAM DICT FOR TRACKING
params={
        "max_epochs": MAX_EPOCHS,
        "lr" : LR,
        "batch_size" : BATCH_SIZE,
        "optimizer" : OPTIMIZER,
        "dropout": DROPOUT_PROB,
        "weight_decay": WEIGHT_DECAY,
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_heads":NUM_HEADS,
        "num_layers": NUM_LAYERS,
        }

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

    with mlflow.start_run(run_name="VIT_baseline"):

        mlflow.set_tag("model_name", "VisionTransformer")
        mlflow.log_params(params)

        model = VisionTransformer(
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            patch_size=4,
            num_channels=3,
            num_patches=64,
            num_classes=100,
            dropout=DROPOUT_PROB,
        ).to(device)

        logging.info(f"Created Vision Transformer Model with {model_params(model)} trainable parameters")


        optimizer = getattr(optim, OPTIMIZER)(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )

        train_dataset, val_dataset, test_dataset = get_CIFAR_data()
        train_loader = DataLoader(train_dataset, BATCH_SIZE)
        val_loader = DataLoader(val_dataset, BATCH_SIZE)
        test_loader = DataLoader(test_dataset, BATCH_SIZE)
        train_full(
            model=model,
            optimizer=optimizer,
            loss_fn=LOSS_FN(),
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=MAX_EPOCHS,
            log_mlflow=True
        )
        test_accuracy, test_loss = test(
        model=model, loss_fn=LOSS_FN(), test_loader=test_loader
        )
        
        mlflow.log_metric("test_loss",test_loss)
        mlflow.log_metric("test_accuracy",test_accuracy)    
 
