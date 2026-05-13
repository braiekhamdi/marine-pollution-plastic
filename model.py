import torch.nn as nn
import torch.nn.functional as F

# Define a CNN model for classification
class MarineDebrisClassifier(nn.Module):
    def __init__(self):
        super(MarineDebrisClassifier, self).__init__()

        # First convolutional layer: 3 input channels (RGB), 6 output channels
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding="same")  # Input: 224x224x3, Output: 224x224x6
        self.pool = nn.MaxPool2d(2, 2)  # Downsample by 2x, Output: 112x112x6

        # Second convolutional layer: 6 input channels, 16 output channels
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding="same")  # Input: 112x112x6, Output: 112x112x16
        self.pool = nn.MaxPool2d(2, 2)  # Downsample by 2x, Output: 56x56x16

        # Third convolutional layer: 16 input channels, 64 output channels
        self.conv3 = nn.Conv2d(16, 64, kernel_size=3, padding="valid")  # Input: 56x56x16, Output: 54x54x64
        self.pool = nn.MaxPool2d(2, 2)  # Downsample by 2x, Output: 27x27x64

        # Fourth convolutional layer: 64 input channels, 32 output channels
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding="same")  # Input: 27x27x64, Output: 27x27x32
        self.pool = nn.MaxPool2d(2, 2)  # Downsample by 2x, Output: 13x13x32

        # Fully connected layers
        self.fc1 = nn.Linear(13 * 13 * 32, 128)  # Adjusted for additional conv layer
        self.fc2 = nn.Linear(128, 2)  # Output: 2 classes (no-plastic, plastic)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Apply ReLU activation and pooling
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(x.shape[0], -1)  # Flatten before FC layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No softmax (handled in loss function)
        return x


import torch.nn as nn
from torchvision import models

class MarineDebrisClassifierV2(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained MobileNetV2
        self.base = models.mobilenet_v2(
            weights='IMAGENET1K_V1')
        # Replace classifier head
        self.base.classifier[1] = nn.Linear(
            1280, 2)

    def forward(self, x):
        return self.base(x)

