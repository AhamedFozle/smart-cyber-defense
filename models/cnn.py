import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # Convolution layers
        self.conv1 = nn.Conv2d(
            3, 32,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            32, 64,
            kernel_size=3,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            64, 128,
            kernel_size=3,
            padding=1
        )

        # Pooling
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Fully connected layers
        # 128 × 4 × 4 = 2048
        self.fc1 = nn.Linear(
            128 * 4 * 4,
            256
        )

        self.fc2 = nn.Linear(
            256,
            num_classes
        )

        self.dropout = nn.Dropout(0.5)


    def extract_features(self, x):
        """
        Extract the learned 256-dimensional feature vector.
        """

        # Block 1: 32 × 32 → 16 × 16
        x = self.pool(
            F.relu(self.conv1(x))
        )

        # Block 2: 16 × 16 → 8 × 8
        x = self.pool(
            F.relu(self.conv2(x))
        )

        # Block 3: 8 × 8 → 4 × 4
        x = self.pool(
            F.relu(self.conv3(x))
        )

        # Flatten: 128 × 4 × 4 = 2048
        x = x.view(x.size(0), -1)

        # Learned feature representation
        x = F.relu(self.fc1(x))

        return x


    def forward(self, x):

        features = self.extract_features(x)

        features = self.dropout(features)

        output = self.fc2(features)

        return output