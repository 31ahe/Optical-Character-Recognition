from torch import nn
import torch

class feature_extractor(nn.Module):
    def __init__(self):
        super(feature_extractor, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2), 

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2), 

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=7, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Flatten(2),
        )

    def forward(self, x):
        x = torch.permute(self.feature_extractor(x), [0,2,1])
        return x