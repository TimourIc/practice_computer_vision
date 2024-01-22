from typing import Type

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import Module
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from src.MLtools import img_to_patch

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



class AttentionBlock(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )


    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_channels,
        num_heads,
        num_layers,
        num_classes,
        patch_size,
        num_patches,
        dropout=0.0,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.patch_size = patch_size

        # Layers/Networks
        self.input_layer = nn.Linear(num_channels * (patch_size**2), embed_dim)
        self.transformer = nn.Sequential(
            *[
                AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))

    def forward(self, x):
        # Preprocess input
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, : T + 1]

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out