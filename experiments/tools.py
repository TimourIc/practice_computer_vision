import argparse
from typing import Type

import joblib
import numpy as np
import optuna
import torch
import yaml
from torch import nn, optim
from torch.nn import CrossEntropyLoss, Module
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import logging
import os
import pickle

import matplotlib.pyplot as plt

from src.MLtools import get_mnist_data, model_params, test, train, train_full
from src.models import FNN
from src.optuna_tools import objective, study_properties


def plot_nist(data_loader: DataLoader):
    examples = enumerate(data_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i in range(9):
        axes[i // 3, i % 3].imshow(example_data[i][0], cmap="gray_r")
        axes[i // 3, i % 3].set_title("Ground Truth: {}".format(example_targets[i]))
    plt.tight_layout()
    plt.show()


def data_to_list(x):
    if type(x) == torch.Tensor:
        return x.tolist()
    elif type(x) == list:
        return x


def tune_model(
    train_dataset,
    val_dataset,
    model_name: str,
    input_model: Type[Module],
    loss_fn: Type[Module],
    max_epochs: int,
    n_trials: int,
    hyperband_min_resource: int,
    hyperband_max_resource: int,
    hyperband_reduction_factor: int,
    model_path: str,
):
    num_channels = train_dataset.dataset[0][0].shape[0]
    input_size = train_dataset.dataset[0][0].shape[1]
    num_classes = len(set(data_to_list(train_dataset.dataset.targets)))
    num_params = model_params(
        input_model(
            input_size=input_size, num_channels=num_channels, num_classes=num_classes
        )
    )
    logging.info(
        f"Starting Tune and Save func for {model_name} with {num_params} trainable weights"
    )

    logging.info(
        f"Starting Hyperparameter Search in Optuna for {model_name} model with validation set."
    )

    ##OPTUNA STUDY
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=hyperband_min_resource,
            max_resource=hyperband_max_resource,
            reduction_factor=hyperband_reduction_factor,
        ),
    )

    objective_optuna = lambda trial: objective(
        trial=trial,
        input_model=input_model,
        input_size=input_size,
        num_channels=num_channels,
        num_classes=num_classes,
        loss_fn=loss_fn(),
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        max_epochs=max_epochs,
    )

    save_path = f"{model_path}/{model_name}_model"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    study.optimize(objective_optuna, n_trials=n_trials)
    joblib.dump(study, f"{save_path}/study.pickle")


def train_and_save(
    train_loader,
    val_loader,
    test_loader,
    model_name: str,
    dataset_name: str,
    input_model: Type[Module],
    loss_fn: Type[Module],
    optimizer: Type[Module],
    max_epochs: int,
    models_path: str,
):
    logging.info(f"Starting model training for {model_name} model")
    training_loss, validation_loss, training_accuracy, validation_accuracy, _ = train_full(
        model=input_model,
        optimizer=optimizer,
        loss_fn=loss_fn(),
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=max_epochs,
    )

    model_path = f"{models_path}/{model_name}_model"
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    logging.info(f"Saving best model in:{model_path}")
    torch.save(input_model.state_dict(), f"{model_path}/{model_name}.pt")

    loaded_model = input_model.to(device)
    loaded_model.load_state_dict(torch.load(f"{model_path}/{model_name}.pt"))
    test_accuracy, test_loss = test(
        model=loaded_model, loss_fn=loss_fn(), test_loader=test_loader
    )
    logging.info(
        f"Final {model_name} model yields {test_accuracy} accuracy on test set"
    )

    logging.info(f"Saving train/val/test info in:{model_path}")
    with open(f"{model_path}/training_loss.pickle", "wb") as f:
        pickle.dump(
            {
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "test_loss": test_loss,
                "training_accuracy": training_accuracy,
                "validation_accuracy": validation_accuracy,
                "test_accuracy": test_accuracy
            },
            f,
        )
