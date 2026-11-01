"""Small CNN classifiers for GTSRB-style 32x32 sign crops, used for CPU replications.

Two architectures from the open-source literature are reproduced here from their public
descriptions (no source code copied):

``lenet5_gray``
    The LeNet-5 variant used by the Udacity Self-Driving Car Nanodegree "Traffic Sign
    Classifier" project (2016 to 2018): grayscale 32x32 input, conv5x5(6) -> pool ->
    conv5x5(16) -> pool -> FC120 -> FC84 -> FC43. Most public repositories report 91 to 97
    percent on the official test set with this network, depending on preprocessing and
    augmentation.

``stn_cnn_poojahira``
    The network of github.com/poojahira/gtsrb-pytorch (NYU CV course Kaggle in-class
    competition, 2018): a spatial transformer (conv7x7(8) -> pool -> conv5x5(10) -> pool ->
    FC32 -> affine 2x3) in front of conv5x5(100) -> conv3x3(150) -> conv3x3(250) with
    BatchNorm, max-pooling, Dropout2d and leaky ReLU, then FC350 -> FC43. The README reports
    99.809 percent on the private leaderboard (which scores the official GTSRB test set) after
    40 epochs on a 10x offline-augmented training set with test-time augmentation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 43


class LeNet5Gray(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 3, 32, 32) in [0, 1]; convert to grayscale and standardise like the Udacity recipe
        x = x.mean(dim=1, keepdim=True)
        x = (x - 0.5) / 0.5
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        return self.fc3(x)


class STNCNNPoojahira(nn.Module):
    """Architecture of poojahira/gtsrb-pytorch (re-implemented from its README and model.py)."""

    MEAN = (0.3337, 0.3064, 0.3171)
    STD = (0.2672, 0.2564, 0.2629)

    def __init__(self, num_classes: int = NUM_CLASSES, align_corners: bool = False):
        super().__init__()
        # The original repository trained in 2018 with PyTorch 0.4 / 1.0, whose affine_grid and
        # grid_sample defaulted to align_corners=True (changed to False in PyTorch 1.3). Use True when
        # loading the released model_40.pth, False for models trained here.
        self.align_corners = align_corners
        self.conv1 = nn.Conv2d(3, 100, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(100)
        self.conv2 = nn.Conv2d(100, 150, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(150)
        self.conv3 = nn.Conv2d(150, 250, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(250)
        self.conv_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(250 * 2 * 2, 350)
        self.fc2 = nn.Linear(350, num_classes)
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )
        self.fc_loc = nn.Sequential(nn.Linear(10 * 4 * 4, 32), nn.ReLU(True), nn.Linear(32, 3 * 2))
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.register_buffer("mean", torch.tensor(self.MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.STD).view(1, 3, 1, 1))

    def stn(self, x: torch.Tensor) -> torch.Tensor:
        xs = self.localization(x).flatten(1)
        theta = self.fc_loc(xs).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=self.align_corners)
        return F.grid_sample(x, grid, align_corners=self.align_corners)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        x = self.stn(x)
        x = self.bn1(F.max_pool2d(F.leaky_relu(self.conv1(x)), 2))
        x = self.conv_drop(x)
        x = self.bn2(F.max_pool2d(F.leaky_relu(self.conv2(x)), 2))
        x = self.conv_drop(x)
        x = self.bn3(F.max_pool2d(F.leaky_relu(self.conv3(x)), 2))
        x = self.conv_drop(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return self.fc2(x)


MODELS = {
    "lenet5_gray": LeNet5Gray,
    "stn_cnn_poojahira": STNCNNPoojahira,
}


def build_model(name: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](num_classes=num_classes)


def random_affine_batch(
    x: torch.Tensor,
    degrees: float = 15.0,
    translate: float = 0.1,
    scale: tuple[float, float] = (0.9, 1.1),
    shear_deg: float = 2.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Per-sample random affine warp of a batch (N, C, H, W) via affine_grid, border replicated.

    This is the online counterpart of the torchvision RandomAffine / RandomRotation transforms
    used by the GTSRB recipes, done on the batch so that CPU training is not bound by PIL.
    """
    n = x.shape[0]
    rad = torch.deg2rad((torch.rand(n, generator=generator) * 2 - 1) * degrees)
    sh = torch.deg2rad((torch.rand(n, generator=generator) * 2 - 1) * shear_deg)
    s = scale[0] + torch.rand(n, generator=generator) * (scale[1] - scale[0])
    tx = (torch.rand(n, generator=generator) * 2 - 1) * translate * 2  # normalised coords span 2
    ty = (torch.rand(n, generator=generator) * 2 - 1) * translate * 2
    cos, sin = torch.cos(rad), torch.sin(rad)
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0] = cos / s
    theta[:, 0, 1] = (-sin + torch.tan(sh)) / s
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin / s
    theta[:, 1, 1] = cos / s
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, padding_mode="border", align_corners=False)


def random_photometric_batch(
    x: torch.Tensor, brightness: float = 0.4, contrast: float = 0.4, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Per-sample brightness and contrast jitter on a batch in [0, 1]."""
    n = x.shape[0]
    b = 1 + (torch.rand(n, 1, 1, 1, generator=generator) * 2 - 1) * brightness
    c = 1 + (torch.rand(n, 1, 1, 1, generator=generator) * 2 - 1) * contrast
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * c + mean
    x = x * b
    return x.clamp_(0, 1)
