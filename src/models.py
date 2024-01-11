from typing import Type

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import Module
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FNN(nn.Module):
    def __init__(
        self, input_size=28, num_channels=1, num_classes=10, dropout_prob=0.25
    ):
        super().__init__()

        self.net_input_size = input_size**2 * num_channels
        self.intermediate_size = round(self.net_input_size / 784) * 200

        self.fc1 = nn.Linear(self.net_input_size, self.intermediate_size)
        self.fc2 = nn.Linear(self.intermediate_size, self.intermediate_size)
        self.fc3 = nn.Linear(self.intermediate_size, self.intermediate_size)
        self.fcN = nn.Linear(self.intermediate_size, num_classes)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.norm = nn.BatchNorm1d(self.intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.norm(x)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.norm(x)
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fcN(x)

        return x


class AlexNet(nn.Module):
    "not really AlexNet, but I guess kind of similar structure"

    def __init__(self, input_size=28, num_channels=1, num_classes=10, dropout_prob=0.3):
        super().__init__()

        self.num_channels = num_channels
        self.input_size = input_size
        self.num_classes = num_classes

        ###CONVOLUTIONAL PART
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.num_channels, out_channels=48, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.conv2 = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Conv2d(in_channels=48, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.conv3 = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(192),
            nn.Dropout(p=dropout_prob),
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        self.flatten = nn.Flatten()
        self.dense_input_size = self.calculate_fc_input_size()
        self.dense_intermediate_size = self.dense_input_size

        ###CLASSIFIER PART
        self.dense = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(self.dense_input_size, self.dense_intermediate_size),
            nn.ReLU(),
            nn.BatchNorm1d(self.dense_intermediate_size),
            nn.Linear(self.dense_intermediate_size, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)

        return x

    def calculate_fc_input_size(self):
        # Define a dummy input tensor with the minimum required dimensions
        x = torch.zeros(
            1, self.num_channels, self.input_size, self.input_size, dtype=torch.float32
        )

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)

        return x.size(1)
