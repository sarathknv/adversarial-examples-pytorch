import torch.nn as nn
import torch
import torch.nn.functional as F


class Discriminator_MNIST(nn.Module):
	def __init__(self):
		super(Discriminator_MNIST, self).__init__()

		self.conv1 = nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=1)
		#self.in1 = nn.InstanceNorm2d(8)
		# "We do not use instanceNorm for the first C8 layer."

		self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1)
		self.in2 = nn.InstanceNorm2d(16)

		self.conv3 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
		self.in3 = nn.InstanceNorm2d(32)

		self.fc = nn.Linear(3*3*32, 1)

	def forward(self, x):

		x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
		x = F.leaky_relu(self.in2(self.conv2(x)), negative_slope=0.2)

		x = F.leaky_relu(self.in3(self.conv3(x)), negative_slope=0.2)

		x = x.view(x.size(0), -1)

		x = self.fc(x)

		return x


if __name__ == '__main__':

	from tensorboardX import SummaryWriter
	from torch.autograd import Variable
	from torchvision import models

	X = Variable(torch.rand(13, 1, 28, 28))

	model = Discriminator_MNIST()
	model(X)

	with SummaryWriter(log_dir="visualization/Discriminator_MNIST", comment='Discriminator_MNIST') as w:
		w.add_graph(model, (X, ), verbose=True)
