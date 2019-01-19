import torch.nn as nn
import torch
import torch.nn.functional as F


class Model_A(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Model_A, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5)

        self.dropout1 = nn.Dropout2d(p=0.25)

        self.fc1 = nn.Linear(8*8*64, 128)

        self.dropout2 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(128, self.num_classes)


    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = self.dropout1(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return x



class Model_B(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Model_B, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=8) # 21

        self.dropout1 = nn.Dropout2d(p=0.2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=6) # 16
        self.conv3 = nn.Conv2d(128, 128, kernel_size=5) # 12

        self.dropout2 = nn.Dropout2d(p=0.5)

        self.fc = nn.Linear(12*12*128, self.num_classes)


    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.dropout1(x)

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        x = self.dropout2(x)

        x = self.fc(x)

        return x



class Model_C(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Model_C, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv1_1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)


        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(7*7*64, 200)
        self.fc2 = nn.Linear(200, self.num_classes)


    def forward(self, x):

        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))

        x = self.maxpool1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))

        x = self.maxpool2(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
