import logging
from typing import Type

import joblib
import numpy as np
import optuna
import torch
from optuna.trial import Trial, TrialState
from torch import nn, optim
from torch.nn import CrossEntropyLoss, Module
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor

from src.MLtools import EarlyStopper, test, train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def objective(
    trial: Trial,
    input_model: Type[nn.Module],
    input_size: int,
    num_channels: int,
    num_classes: int,
    loss_fn: Type[nn.Module],
    train_dataset,
    val_dataset,
    max_epochs: int = 10,
):
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    lr = trial.suggest_categorical("lr", [0.01, 0.001, 0.0001, 0.00001])
    batch_size = trial.suggest_categorical("batch_size", [16, 64, 256])
    dropout_rate = trial.suggest_categorical("dropout_prob", [0.1, 0.25, 0.5])

    model = input_model(
        input_size=input_size,
        num_channels=num_channels,
        num_classes=num_classes,
        dropout_prob=dropout_rate,
    )
    model = model.to(device)

    optimizer = getattr(optim, optimizer_name)(
        model.parameters(), lr=lr, weight_decay=0.01
    )

    train_loader = DataLoader(train_dataset, batch_size)
    val_loader = DataLoader(val_dataset, batch_size)

    logging.info(
        f" *** Starting new Optuna trial: *** Optimizer: {trial.params['optimizer']}, Learning Rate: {trial.params['lr']}, Batch Size: {trial.params['batch_size']}, Dropout: {dropout_rate}"
    )

    # trial.set_user_attr("pruned_epoch", max_epochs)

    for epoch in range(0, max_epochs):
        _ = train(model, optimizer, loss_fn, epoch, train_loader)
        accuracy, val_loss = test(model, loss_fn, val_loader)
        trial.report(accuracy, epoch)

        if trial.should_prune():
            trial.set_user_attr("pruned_epoch", epoch)
            raise optuna.TrialPruned()

    return accuracy


def study_properties(
    study_path: str,
) -> Trial:
    "Reads in Optuna Study and returns best trial"
    "stud_path: path to completed optuna study"

    # READ BEST TRIAL
    study = joblib.load(study_path)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    logging.info("Study statistics: ")
    logging.info(f"Number of finished trials: {len(study.trials)}")
    logging.info(f"Number of pruned trials: {len(pruned_trials)}")
    logging.info(f"Number of complete trials: {len(complete_trials)}")

    logging.info("Best trial:")
    trial = study.best_trial

    logging.info(f"Value: {trial.value}")

    logging.info(f"Params: ")
    for key, value in trial.params.items():
        logging.info("    {}: {}".format(key, value))

    return trial
