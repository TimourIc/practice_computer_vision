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
from src.models import FNN, AlexNet
from src.optuna_tools import study_properties

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("VIT_CIFAR")

logger = logging.getLogger(__name__)

# DEFAULT PARAMS
MAX_EPOCHS: int = 40
LOSS_FN = CrossEntropyLoss
MODELS = {"FNN": FNN, "AlexNet": AlexNet}
LR = 0.001
BATCH_SIZE = 64
OPTIMIZER = "Adam"
DROPOUT_PROB = 0.25
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


def show_img():
    NUM_IMAGES = 4
    CIFAR_images = torch.stack(
        [val_dataset[idx][0] for idx in range(NUM_IMAGES)], dim=0
    )
    img_grid = torchvision.utils.make_grid(
        CIFAR_images, nrow=4, normalize=True, pad_value=0.9
    )
    img_grid = img_grid.permute(1, 2, 0)

    plt.figure(figsize=(8, 8))
    plt.title("Image examples of the CIFAR10 dataset")
    plt.imshow(img_grid)
    plt.axis("off")
    plt.show()
    plt.close()


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

        logging.info(f"Created Vision Transformer Model with {model_params} trainable parameters")


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

        # attn=AttentionBlock(embed_dim=48,hidden_dim=)
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
