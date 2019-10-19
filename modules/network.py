import torch.nn as nn
import torch
import numpy as np


class FeatureDetector(nn.Module):

    def __init__(self):
        super(FeatureDetector, self).__init__()
        # Convolution 1
        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


    def forward(self, x):
        x = torch.Tensor(x).unsqueeze(0)
        # Convolution
        x = self.cnn_layers(x)
        # flattens
        x = x.view(x.size(0), -1)

        return x


