"""
    Custom model for training on cifar10
"""

import torch.nn.functional as F
import torch.nn as nn

class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        """
            input   - (3, 32, 32)
            block 1 - (32, 32, 32)
            maxpool - (32, 16, 16)
            block 2 - (64, 16, 16)
            maxpool - (64, 8, 8)
            block 3 - (128, 8, 8)
            maxpool - (128, 4, 4)
            block 4 - (128, 4, 4)
            avgpool - (128, 1, 1), reshpe to (128,)
            fc      - (128,) -> (10,)

        """
        # block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # block 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # block 3
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # block 4
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):

        # block 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.bn1(self.conv2(x)))

        # maxpool
        x = F.max_pool2d(x, 2)

        # block 2
        x = F.relu(self.conv3(x))
        x = F.relu(self.bn2(self.conv4(x)))

        # maxpool
        x = F.max_pool2d(x, 2)

        # block 3
        x = F.relu(self.conv5(x))
        x = F.relu(self.bn3(self.conv6(x)))

        # maxpool
        x = F.max_pool2d(x, 2)

        # block 4
        x = F.relu(self.conv7(x))
        x = F.relu(self.bn4(self.conv8(x)))

        # avgpool and reshape to 1D
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # fc
        x = self.fc(x)

        return x
