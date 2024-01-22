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
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from tqdm import tqdm

import mlflow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





def model_params(input_model: Type[nn.Module]):
    num_params = 0
    for x in input_model.parameters():
        num_params += len(torch.flatten(x))

    return num_params


transform_CIFAR = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

transform_MNIST = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
)


def get_mnist_data(seed: int = 42, fashion: bool = True):
    logging.info("Downloading Data")

    torch.manual_seed(seed)

    if fashion is True:
        train_dataset = datasets.FashionMNIST(
            root="data", download=True, train=True, transform=transform_MNIST
        )
        test_dataset = datasets.FashionMNIST(
            root="data", download=True, train=False, transform=transform_MNIST
        )
    else:
        train_dataset = datasets.MNIST(
            root="data", download=True, train=True, transform=transform_MNIST
        )
        test_dataset = datasets.MNIST(
            root="data", download=True, train=False, transform=transform_MNIST
        )
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    return train_dataset, val_dataset, test_dataset


def get_CIFAR_data(seed: int = 42):
    logging.info("Downloading Data")

    torch.manual_seed(seed)

    train_dataset = datasets.CIFAR100(
        root="data", download=True, train=True, transform=transform_CIFAR
    )
    test_dataset = datasets.CIFAR100(
        root="data", download=True, train=False, transform=transform_CIFAR
    )

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    return train_dataset, val_dataset, test_dataset


def get_cifar_data(seed: int = 42):
    pass


def train(
    model: Type[nn.Module],
    optimizer: Type[Optimizer],
    loss_fn: Type[Module],
    epoch: int,
    train_loader: DataLoader,
) -> tuple[float, float]:
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    logging.info(f"Train Epoch: {epoch}, Train Average Loss: {loss.item():.4f}")

    average_train_loss = train_loss / len(train_loader)
    average_train_accuracy = correct / len(train_loader.dataset)
    return average_train_loss, average_train_accuracy


def train_full(
    model: Type[nn.Module],
    optimizer: Type[Optimizer],
    loss_fn: Type[Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int,
    log_mlflow: bool= False
):
    training_loss = []
    validation_loss = []
    training_accuracy = []
    validation_accuracy = []
    early_stopper = EarlyStopper()

    for epoch in range(0, max_epochs):
        train_loss, train_accuracy = train(
            model, optimizer, loss_fn, epoch, train_loader
        )
        val_accuracy, val_loss = test(model, loss_fn, val_loader)
        training_loss.append(train_loss)
        validation_loss.append(val_loss)
        training_accuracy.append(train_accuracy)
        validation_accuracy.append(val_accuracy)

        if log_mlflow:
            mlflow.log_metric("train_loss",train_loss, step=epoch)
            mlflow.log_metric("val_loss",val_loss, step=epoch)
            mlflow.log_metric("train_accuracy",train_accuracy , step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

        if early_stopper.early_stop(model, val_loss):
            logging.info(
                f"Stopped Early at epoch {epoch} because the validation loss was no longer improving"
            )
            final_epoch = epoch
            break

    final_epoch = epoch

    return (
        training_loss,
        validation_loss,
        training_accuracy,
        validation_accuracy,
        final_epoch,
    )


def test(
    model: Type[nn.Module], loss_fn: Type[Module], test_loader: DataLoader
) -> float:
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for _, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss = test_loss / len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    logging.info(
        f"Test/Val Average loss: {test_loss:.4f}, Test/Val Accuracy {accuracy}"
    )

    return accuracy, test_loss


class EarlyStopper:
    def __init__(self, patience=6, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")
        self.best_model_state = None

    def early_stop(self, model, validation_loss):
        logging.info(f"Early stop counter: {self.counter}")
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best_model_state = model.state_dict()
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                if self.best_model_state is not None:
                    model.load_state_dict(self.best_model_state)
                    return True
        return False


def img_to_patch(x: torch.Tensor, patch_size: int, flatten_channels: bool = True):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    # print(f"batch_size: {B}, channels: {C}, height:{H}, width: {W}")
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
    return x