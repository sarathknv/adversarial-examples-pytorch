""" Generative Adversarial Trainer

Paper: https://arxiv.org/pdf/1705.03387


Generates adversarial examples using GANs, not exactly.

The Discriminator is just a classifier, and Generator outputs a perturbation by taking
gradient image as input..You can read the paper.

But the Generator doesn't seem to converge.
"""

import torch.nn as nn
from torchvision import datasets, transforms
import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import argparse
import torch.optim as optim
from torchvision.transforms import ToTensor, RandomHorizontalFlip, ToPILImage, RandomRotation, RandomResizedCrop, Normalize
import time

batch_size = 32
epochs = 32
transform = transforms.Compose([RandomRotation(20), RandomResizedCrop(size=32, scale=(0.9, 1.0)), ToTensor()])
train_loader = DataLoader(datasets.CIFAR10('../data', train=True, download=True,
						transform=transform),
						batch_size=batch_size, shuffle=True)

test_loader = DataLoader(datasets.CIFAR10('../data', train=False,
						transform=transforms.Compose([ToTensor()])),
						batch_size=batch_size, shuffle=True)



class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()

		self.network = nn.Sequential(
			nn.Conv2d(3, 48, kernel_size=3, padding=1),
			nn.BatchNorm2d(48),
			nn.ReLU(inplace=True),

			nn.Conv2d(48, 48, kernel_size=3, padding=1),
			nn.BatchNorm2d(48),
			nn.ReLU(inplace=True),

			nn.Conv2d(48, 48, kernel_size=3, padding=1),
			nn.BatchNorm2d(48),
			nn.ReLU(inplace=True),

			nn.Conv2d(48, 48, kernel_size=3, padding=1),
			nn.BatchNorm2d(48),
			nn.ReLU(inplace=True),

			nn.Conv2d(48, 48, kernel_size=3, padding=1),
			nn.BatchNorm2d(48),
			nn.ReLU(inplace=True),

			nn.Conv2d(48, 48, kernel_size=3, padding=1),
			nn.BatchNorm2d(48),
			nn.ReLU(inplace=True),

			nn.Conv2d(48, 48, kernel_size=1, padding=0),
			nn.BatchNorm2d(48),
			nn.ReLU(inplace=True),

			nn.Conv2d(48, 3, kernel_size=1, padding=0),
			nn.Tanh()
				)
	def forward(self, x):
		return self.network(x)

class Classifier(nn.Module):
	def __init__(self):
		super(Classifier, self).__init__()

		self.network = nn.Sequential(

				nn.Conv2d(3, 48, kernel_size=3, padding=1),
				nn.BatchNorm2d(48),
				nn.ReLU(inplace=True),

				nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1),
				nn.BatchNorm2d(48),
				nn.ReLU(inplace=True),

				nn.Conv2d(48, 96, kernel_size=3, padding=1),
				nn.BatchNorm2d(96),
				nn.ReLU(inplace=True),

				nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1),
				nn.BatchNorm2d(96),
				nn.ReLU(inplace=True),

				nn.Conv2d(96, 96, kernel_size=3, padding=1),
				nn.BatchNorm2d(96),
				nn.ReLU(inplace=True),

				nn.Conv2d(96, 96, kernel_size=1, padding=0),
				nn.BatchNorm2d(96),
				nn.ReLU(inplace=True),

				nn.Conv2d(96, 10, kernel_size=1, padding=0),
				nn.BatchNorm2d(10),
				nn.ReLU(inplace=True),

				nn.AdaptiveAvgPool2d(1) # nn.AdaptiveAvgPool2d((1, 1))

				)

		self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
		x = self.network(x)
		x = x.view(x.size(0), -1)
		x = self.softmax(x)
		return x

def weights_init(m):
	if isinstance(m, nn.Conv2d):
		nn.init.xavier_normal(m.weight.data)
		m.bias.data.zero_()

    


netG = Generator().cuda()
netF = Classifier().cuda()

netG.apply(weights_init)
netF.apply(weights_init)


optimizerF = optim.Adam(netF.parameters(), lr=1e-3)
optimizerG = optim.Adam(netG.parameters(), lr=1e-6)

def perturbationNorm(perturbation):
	return torch.mean(torch.pow(torch.norm(perturbation, p=2), exponent=2.0))

def classProbability(probs, labels):
	mean = 0
	for i in range(probs.size(0)):
		mean += probs[i][labels[i]]

	return mean/probs.size(0)

def crossEntropy(probs, labels):
	mean = 0
	for i in range(probs.size(0)):
		mean += -torch.log(probs[i][labels[i]])
	return mean/probs.size(0)

import pdb


lg = np.zeros((epochs, 1), dtype=np.float32)
lf = np.zeros((epochs, 1), dtype=np.float32)

for epoch in range(epochs):
	t = time.time()
	n = 0
	nf = 0.05
	for i, data in enumerate(train_loader, 0):

		
		x = Variable(data[0].float().cuda()*255.0, requires_grad=True)
		y = Variable(data[1].long().cuda())

		Fy = netF(x)
		Fy.backward(torch.ones(x.size(0), 10).cuda())

		###################################################
		delta = Variable(x.grad.data, requires_grad=True)
		
		#delta.volatile = False
		
		netG.zero_grad()
		netF.zero_grad()
		# deriviative wrt which variables?
		# detach - necessary?
		lossG = 1*perturbationNorm(delta) + classProbability(netF(x+netG(delta)), y)
		lossG.backward()

		optimizerG.step()

		netG.zero_grad()
		netF.zero_grad()
		if i%10 == 0:

			# these variables need not be defined again.
			# there's redundancy clear it
			x = Variable(data[0].float().cuda()*255.0, requires_grad=True)
			y = Variable(data[1].long().cuda())

			Fy = netF(x)
			Fy.backward(torch.ones(x.size(0), 10).cuda())

			delta = Variable(x.grad.data, requires_grad=True)
			x.grad.data.zero_()
			netF.zero_grad() # is it necessary?
			lossF = 0.5*crossEntropy(netF(x), y) + 0.5*crossEntropy(netF(x+delta), y)

			
			x.grad.data.zero_()
			
			lossF.backward()
			optimizerF.step()
			lf[epoch] += lossF.data[0]*x.size(0)
			nf += x.size(0)
			netG.zero_grad()
			netF.zero_grad()
		
		n += x.size(0)
		# why multiply by n? unnecessary.
		lg[epoch] += lossG.data[0]*x.size(0)

		time_spent = time.time() - t

		print('Epoch [%2d/%2d]: [%4d/%4d]\tLossG: %3.5f\tLossF: %3.5f'%(epoch+1, epochs, i+1, len(train_loader), lg[epoch]/n, lf[epoch]/nf), end='\r')

	print('Epoch [%2d/%2d]'%(epoch+1, epochs))
	print('-'*20)
	print('Total Time: %0.2f'%(time_spent))
	print('Time per sample : %.6f'%(time_spent/n))
	print('LossG: %.8f' %(lg[epoch]/n))
	print('LossF: %.8f' %(lf[epoch]/nf))
	print()

pdb.set_trace()