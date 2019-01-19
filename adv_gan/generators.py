import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = x

        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))

        out = out + residual

        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()

        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(mode='nearest', scale_factor=upsample)

        padding = kernel_size // 2

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)

    def forward(self, x):

        if self.upsample:
            x = self.upsample_layer(x)

        x = self.conv2d(x)

        return x



class Generator_MNIST(nn.Module):
    def __init__(self):
        super(Generator_MNIST, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.in1 = nn.InstanceNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.in2 = nn.InstanceNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.in3 = nn.InstanceNorm2d(32)

        self.resblock1 = ResidualBlock(32)
        self.resblock2 = ResidualBlock(32)
        self.resblock3 = ResidualBlock(32)
        self.resblock4 = ResidualBlock(32)


        self.up1 = UpsampleConvLayer(32, 16, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(16)
        self.up2 = UpsampleConvLayer(16, 8, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(8)


        self.conv4 = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)
        self.in6 = nn.InstanceNorm2d(8)


    def forward(self, x):

        x = F.relu(self.in1(self.conv1(x)))
        x = F.relu(self.in2(self.conv2(x)))
        x = F.relu(self.in3(self.conv3(x)))

        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)

        x = F.relu(self.in4(self.up1(x)))
        x = F.relu(self.in5(self.up2(x)))

        x = self.in6(self.conv4(x)) # remove relu for better performance and when input is [-1 1]

        return x



if __name__ == '__main__':

    from tensorboardX import SummaryWriter
    from torch.autograd import Variable
    from torchvision import models

    X = Variable(torch.rand(13, 1, 28, 28))

    model = Generator_MNIST()

    with SummaryWriter(log_dir="visualization/Generator_MNIST", comment='Generator_MNIST') as w:
        w.add_graph(model, (X, ))
