import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.model = resnet18(weights=None)

        # CIFAR-10 images are 32x32, so replace the
        # ImageNet-style first convolution and max pooling.
        self.model.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.model.maxpool = nn.Identity()

        # CIFAR-10 has 10 classes
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes
        )

    def forward(self, x):
        return self.model(x)