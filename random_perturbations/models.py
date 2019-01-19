"""
    Custom models for training on cifar10 and mnist

    BasicCNN and BasicNN
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

class BasicNN(nn.Module):
    def __init__(self):
        super(BasicNN, self).__init__()

        self.fc1 = nn.Linear(28*28, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)


        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)

        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)

        self.fc6 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))

        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))

        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))

        x = self.fc6(x)

        return x
