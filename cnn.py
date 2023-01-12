import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, size_in=28, channels_in=1, classes_out=62):
        super().__init__()
        # 1 input image channel, 64 output channels, 3x3 convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels_in, 64, 3),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.drop = nn.Dropout(0.5)
        i = ((size_in - 4) // 2 - 2) // 2
        self.fc1 = nn.Sequential(nn.Linear(128 * i * i, 256), nn.LeakyReLU())
        self.fc2 = nn.Sequential(nn.Linear(256, 128), nn.LeakyReLU())
        self.fc3 = nn.Linear(128, classes_out)

    def forward(self, x):
        # (start-window) / stride + 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.drop(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
