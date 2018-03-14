import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.transforms import ToTensor, Normalize
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import Dataset, DataLoader
import time
import shutil
import numpy as np
from model import BasicCNN, BasicNN

from torchvision.transforms import ToTensor, RandomHorizontalFlip, ToPILImage, RandomRotation, RandomResizedCrop, Normalize


numb = 0

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'])
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=50)
args = parser.parse_args()


batch_size = args.batch_size
epochs = args.epochs
lr = 0.01
weight_decay = 1e-5

print('dataset:', args.dataset)
print('epochs:', epochs)
print('batch size:', batch_size)


transform = transforms.Compose([RandomRotation(20), RandomResizedCrop(size=32, scale=(0.8, 1.1)), ToTensor()])
train_loader = DataLoader(datasets.CIFAR10('../data', train=True, download=True,
						transform=transform),
						batch_size=batch_size, shuffle=True)

test_loader = DataLoader(datasets.CIFAR10('../data', train=False,
						transform=transforms.Compose([ToTensor()])),
						batch_size=batch_size, shuffle=True)


model = BasicCNN()
model.cuda()


optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)	
#optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss().cuda()
scheduler = StepLR(optimizer, step_size=10, gamma=0.3)



train_loss = np.zeros((epochs, 1), dtype=np.float32)
train_acc = np.zeros((epochs, 1), dtype=np.float32)
val_acc = np.zeros(shape=(epochs,1), dtype=np.float32)
val_loss = np.zeros(shape=(epochs, 1), dtype=np.float32)



def save_checkpoint(state, filename='saved/cifar10_checkpoint_%s.pth.tar'%(numb)):
	torch.save(state, filename)
	if state['is_best']==True:
		shutil.copyfile(filename, 'saved/cifar10_model_best_%s.pth.tar'%(numb))


def validate(epoch):
	N_val = 0.0
	for data_val in test_loader:
		inputs = Variable(data[0].float().cuda())
		labels = Variable(data[1].long().cuda())

		optimizer.zero_grad()


		outputs = model(inputs)
		_, preds = torch.max(outputs.data, 1)
		loss = criterion(outputs, labels)
		curr_batch_size = inputs.size(0)
		N_val += curr_batch_size
		val_loss[epoch] += loss.data[0]*curr_batch_size
		val_acc[epoch] += torch.sum(preds == labels.data)

	

	val_loss[epoch] = val_loss[epoch]/N_val
	val_acc[epoch] = val_acc[epoch]/N_val
	return (val_loss[epoch],val_acc[epoch])


best_acc = 0.0
for epoch_cnt in range(epochs):
	tic = time.time()
	n_train = 0.0
	for i, data in enumerate(train_loader):
		
		inputs = Variable(data[0].float().cuda())
		labels = Variable(data[1].long().cuda())
		arr = data[0].numpy()
		#print(arr[(arr<1.0) & (arr > 0.0)])
		optimizer.zero_grad()
		outputs = model(inputs)
		_, preds = torch.max(outputs.data, 1)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		curr_batch_size = inputs.size(0)
		n_train += curr_batch_size
		train_loss[epoch_cnt] += loss.data[0]*curr_batch_size
		train_acc[epoch_cnt] += torch.sum(preds == labels.data)


		print('Epoch [%2d/%2d]: [%3d/%3d]\tLoss: %1.8f\tAccuracy: %1.8f'
			%(epoch_cnt + 1, epochs, i+1, len(train_loader), train_loss[epoch_cnt]/n_train, train_acc[epoch_cnt]/n_train), end='\r')

	train_loss[epoch_cnt] = train_loss[epoch_cnt]/n_train
	train_acc[epoch_cnt] = train_acc[epoch_cnt]/n_train
	scheduler.step()
	time_spent = time.time() - tic

	model.eval()
	curr_val_loss, curr_val_acc = validate(epoch_cnt)
	model.train()

	print('Epoch [%2d/%2d]'%(epoch_cnt+1, epochs))
	print('-'*20)
	print('Total Time: %0.2f'%time_spent)
	print('Time per sample : %.6f'%(time_spent/n_train))
	print('Train loss: %.8f' %(train_loss[epoch_cnt]))
	print('Train Acc: %.8f' %(train_acc[epoch_cnt]))
	print('Val loss: %.8f' %curr_val_loss)
	print('Val Acc: %.8f' %curr_val_acc)
	print()

	is_best = curr_val_acc > best_acc
	best_acc = max(best_acc, curr_val_acc)
	save_checkpoint({'epoch': epoch_cnt,'state_dict': model.state_dict(),'best_acc': best_acc,'optimizer' : optimizer.state_dict(), 'is_best' : is_best})

address = 'saved/'
"""
np.save(address+'mnist_train_loss.npy', train_loss)
np.save(address+'mnist_train_acc.npy', train_acc)
np.save(address+'mnist_val_loss.npy', val_loss)
np.save(address+'mnist_val_acc.npy', val_acc)
"""
print('Finished training.')
print('#'*20)