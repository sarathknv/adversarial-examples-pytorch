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
parser.add_argument('--img', type=str, default='images/goldfish.jpg', help='path to image')
parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50'], required=False, help="Which network?")
parser.add_argument('--y', type=int, required=False, help='Label')
parser.add_argument('--gpu', action="store_true", default=False)

args = parser.parse_args()
image_path = args.img
model_name = args.model
y_true = args.y
gpu = args.gpu

IMG_SIZE = 224

print('Fast Gradient Sign Method')
print('Model: %s' %(model_name))
print()


def nothing(x):
    pass

window_adv = 'perturbation'
cv2.namedWindow(window_adv)
cv2.createTrackbar('eps', window_adv, 1, 255, nothing)


# load image and reshape to (3, 224, 224) and RGB (not BGR)
# preprocess as described here: http://pytorch.org/docs/master/torchvision/models.html
orig = cv2.imread(image_path)[..., ::-1]
orig = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))
img = orig.copy().astype(np.float32)
perturbation = np.empty_like(orig)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img /= 255.0
img = (img - mean)/std
img = img.transpose(2, 0, 1)


# load model
model = getattr(models, model_name)(pretrained=True)
model.eval()
criterion = nn.CrossEntropyLoss()

device = 'cuda' if gpu else 'cpu'


# prediction before attack
inp = Variable(torch.from_numpy(img).to(device).float().unsqueeze(0), requires_grad=True)

out = model(inp)
pred = np.argmax(out.data.cpu().numpy())
print('Prediction before attack: %s' %(classes[pred].split(',')[0]))


while True:
    # get trackbar position
    eps = cv2.getTrackbarPos('eps', window_adv)

    inp = Variable(torch.from_numpy(img).to(device).float().unsqueeze(0), requires_grad=True)


    out = model(inp)
    loss = criterion(out, Variable(torch.Tensor([float(pred)]).to(device).long()))

    # compute gradients
    loss.backward()


    # this is it, this is the method
    inp.data = inp.data + ((eps/255.0) * torch.sign(inp.grad.data))
    inp.grad.data.zero_() # unnecessary


    # predict on the adversarial image
    pred_adv = np.argmax(model(inp).data.cpu().numpy())
    print(" "*60, end='\r') # to clear previous line, not an elegant way
    print("After attack: eps [%f] \t%s"
            %(eps, classes[pred_adv].split(',')[0]), end="\r")#, end='\r')#'eps:', eps, end='\r')


    # deprocess image
    adv = inp.data.cpu().numpy()[0]
    perturbation = (adv - img).transpose(1, 2, 0) #cv2.normalize((adv - img).transpose(1, 2, 0), perturbation, 0, 255, cv2.NORM_MINMAX, 0)
    adv = adv.transpose(1, 2, 0)
    adv = (adv * std) + mean
    adv = adv * 255.0
    adv = adv[..., ::-1] # RGB to BGR
    adv = np.clip(adv, 0, 255).astype(np.uint8)
    perturbation = perturbation * 255
    perturbation = np.clip(perturbation, 0, 255).astype(np.uint8)


    # display images
    cv2.imshow(window_adv, perturbation)
    cv2.imshow('adversarial image', adv)
    key = cv2.waitKey(500) & 0xFF
    if key == 27:
        break
    elif key == ord('s'):
        cv2.imwrite('img_adv.png', adv)
        cv2.imwrite('perturbation.png', perturbation)
print()
cv2.destroyAllWindows()
