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
from model_mnist import Basic_CNN


parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, default='images/img_3.jpg', help='path to image')
parser.add_argument('--y', type=int, required=False, help='Label')
parser.add_argument('--gpu', action="store_true", default=False)

args = parser.parse_args()
image_path = args.img
y_true = args.y
gpu = args.gpu

IMG_SIZE = 28

print('Fast Gradient Sign Method')
print()


def nothing(x):
    pass

window_adv = 'adversarial image'
cv2.namedWindow(window_adv)
cv2.createTrackbar('eps', window_adv, 1, 255, nothing)


orig = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#orig = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))
img = orig.copy().astype(np.float32)
perturbation = np.empty_like(orig)

mean = [0.5]
std = [0.5]
img /= 255.0
img = (img - mean)/std


# load model
model = Basic_CNN(1, 10)
saved = torch.load('9920.pth.tar', map_location='cpu')
model.load_state_dict(saved['state_dict'])
model.eval()
criterion = nn.CrossEntropyLoss()

device = 'cuda' if gpu else 'cpu'


# prediction before attack
inp = Variable(torch.from_numpy(img).to(device).float().unsqueeze(0).unsqueeze(0), requires_grad=True)

out = model(inp)
pred = np.argmax(out.data.cpu().numpy())
print('Prediction before attack: %s' %(pred))


while True:
    # get trackbar position
    eps = cv2.getTrackbarPos('eps', window_adv)

    inp = Variable(torch.from_numpy(img).to(device).float().unsqueeze(0).unsqueeze(0), requires_grad=True)


    out = model(inp)
    loss = criterion(out, Variable(torch.Tensor([float(pred)]).to(device).long()))

    # compute gradients
    loss.backward()


    # this is it, this is the method
    inp.data = inp.data + ((eps/255.0) * torch.sign(inp.grad.data))
    inp.data = inp.data.clamp(min=-1, max=1)
    inp.grad.data.zero_() # unnecessary


    # predict on the adversarial image
    pred_adv = np.argmax(model(inp).data.cpu().numpy())
    print(" "*60, end='\r') # to clear previous line, not an elegant way
    print("After attack: eps [%f] \t%s"
            %(eps, pred_adv), end="\r")#, end='\r')#'eps:', eps, end='\r')


    # deprocess image
    adv = inp.data.cpu().numpy()[0][0]
    perturbation = adv-img
    adv = (adv * std) + mean
    adv = adv * 255.0
    adv = np.clip(adv, 0, 255).astype(np.uint8)
    perturbation = perturbation*255
    perturbation = np.clip(perturbation, 0, 255).astype(np.uint8)


    # display images
    cv2.imshow(window_adv, perturbation)
    cv2.imshow('perturbation', adv)
    key = cv2.waitKey(500) & 0xFF
    if key == 27:
        break
    elif key == ord('s'):
        cv2.imwrite('img_adv.png', adv)
        cv2.imwrite('perturbation.png', perturbation)
print()
cv2.destroyAllWindows()
