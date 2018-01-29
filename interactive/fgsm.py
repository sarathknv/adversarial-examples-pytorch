""" Fast Gradient Sign Method
	Paper link: https://arxiv.org/abs/1607.02533

	Controls: 
		'esc' - exit
		 's'  - save adversarial image
"""
import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
from torchvision import transforms

import numpy as np
import cv2
import argparse
from imagenet_labels import classes

parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, default='aeroplane.jpg', help='path to image')
parser.add_argument('--model', type=str, default='resnet18',
					 choices=['resnet18', 'inception_v3', 'resnet50'],
					 required=False, help="Which network?")
parser.add_argument('--y', type=int, required=False, help='Label')
parser.add_argument('--y_target', type=int, default=None, required=False, help='Label to target')
#parser.add_argument('--eps', type=float, default=0.1, required=False, help="epsilon value")

args = parser.parse_args()
image_path = 'images/' + args.img
model_name = args.model
y_true = args.y
y_target = args.y_target
#eps = args.eps

def nothing(x):
	pass

window_adv = 'adversarial image'
cv2.namedWindow(window_adv)
cv2.createTrackbar('eps', window_adv, 1, 100, nothing)

# load image and reshape to (3, 224, 224) and RGB (not BGR)
# preprocess as described here: http://pytorch.org/docs/master/torchvision/models.html
orig = cv2.imread(image_path)[..., ::-1]
orig = cv2.resize(orig, (224, 224))
img = orig.copy().astype(np.float32)
perturbation = np.empty_like(orig)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img /= 255.0
img = (img - mean)/std
img = img.transpose(2, 0, 1)

# load model
model = getattr(models, model_name)(pretrained=True)
model.eval().cuda()

criterion = nn.CrossEntropyLoss().cuda()


while True:
	# get trackbar position
	eps = cv2.getTrackbarPos('eps', window_adv) * 0.01

	inp = Variable(torch.from_numpy(img).cuda().float().unsqueeze(0), requires_grad=True)
	
	# forward pass, predict, compute loss
	outs = model(inp)
	label = np.argmax(outs.data.cpu().numpy())
	if y_target is not None:
		label = y_target
	loss = criterion(outs, Variable(torch.Tensor([float(label)]).cuda().long()))

	# compute gradients
	loss.backward()
	
	# this is it, this is the method
	inp.data = inp.data + (eps * torch.sign(inp.grad.data))
	
	inp.grad.data.zero_() # this is just a compulsion, unnecessary here

	# predict on the adversarial image
	pred_adv = np.argmax(model(inp).data.cpu().numpy())
	print('Prediction after attack: %s \t\t'%(classes[pred_adv]))#, end='\r')#'eps:', eps, end='\r')
	
	# deprocess image
	adv = inp.data.cpu().numpy()[0]
	perturbation = cv2.normalize((adv - img).transpose(1, 2, 0), perturbation, 0, 255, cv2.NORM_MINMAX, 0)
	adv = adv.transpose(1, 2, 0)
	adv = (adv * std) + mean
	adv = adv * 255.0
	adv = adv[..., ::-1] # RGB to BGR
	adv = np.clip(adv, 0, 255).astype(np.uint8)	
	
	cv2.imshow(window_adv, adv)
	cv2.imshow('perturbation', perturbation)
	key = cv2.waitKey(500) & 0xFF
	if key == 27:
		break
	elif key == ord('s'):
		cv2.imwrite('img_adv.png', arr)
cv2.destroyAllWindows()
