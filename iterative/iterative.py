""" Basic Iterative Method (Targeted and Non-targeted)
    Paper link: https://arxiv.org/abs/1607.02533

    Controls:
        'esc' - exit
         's'  - save adversarial image
      'space' - pause
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
parser.add_argument('--target', type=int, required=False, default=None, help='target label')

args = parser.parse_args()
image_path = args.img
model_name = args.model
y_true = args.y
target = args.target
gpu = args.gpu

IMG_SIZE = 224

print('Iterative Method')
print('Model: %s' %(model_name))
print()


# break loop when parameters are changed
break_loop = False

def nothing(x):
    global break_loop
    break_loop = True

window_adv = 'perturbation'
cv2.namedWindow(window_adv)
cv2.createTrackbar('eps', window_adv, 1, 255, nothing)
#cv2.createTrackbar('alpha', window_adv, 1, 255, nothing)
cv2.createTrackbar('iter', window_adv, 10, 1000, nothing)


# load image and reshape to (3, 224, 224) and RGB (not BGR)
# preprocess as described here: http://pytorch.org/docs/master/torchvision/models.html
orig = cv2.imread(image_path)[..., ::-1]
orig = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))
img = orig.copy().astype(np.float32)
#perturbation = np.empty_like(orig)


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
orig = torch.from_numpy(img).float().to(device).unsqueeze(0)

out = model(inp)
pred = np.argmax(out.data.cpu().numpy())

if target is not None:
    pred = target

print('Prediction before attack: %s' %(classes[pred].split(',')[0]))


while True:
    inp = Variable(torch.from_numpy(img).to(device).float().unsqueeze(0), requires_grad=True)

    eps = cv2.getTrackbarPos('eps', window_adv)
    alpha = 10 #alpha = cv2.getTrackbarPos('alpha', window_adv)
    num_iter = cv2.getTrackbarPos('iter', window_adv)

    print('eps [%d]' %(eps))
    print('Iter [%d]' %(num_iter))
    print('alpha [1]')
    print('-'*20)

    break_loop = False

    for i in range(num_iter):

        if break_loop == False:

            ##############################################################
            out = model(inp)
            loss = criterion(out, Variable(torch.Tensor([float(pred)]).to(device).long()))

            loss.backward()

            # this is the method
            perturbation = (alpha/255.0) * torch.sign(inp.grad.data)
            perturbation = torch.clamp((inp.data + perturbation) - orig, min=-eps/255.0, max=eps/255.0)
            inp.data = orig + perturbation

            inp.grad.data.zero_()
            ################################################################

            pred_adv = np.argmax(model(inp).data.cpu().numpy())

            print("Iter [%3d/%3d]:  Prediction: %s"
                    %(i, num_iter, classes[pred_adv].split(',')[0]))


            # deprocess image
            adv = inp.data.cpu().numpy()[0]
            pert = (adv-img).transpose(1,2,0)
            adv = adv.transpose(1, 2, 0)
            adv = (adv * std) + mean
            adv = adv * 255.0
            adv = adv[..., ::-1] # RGB to BGR
            adv = np.clip(adv, 0, 255).astype(np.uint8)
            pert = pert * 255
            pert = np.clip(pert, 0, 255).astype(np.uint8)

            cv2.imshow(window_adv, pert)
            cv2.imshow('adversarial image', adv)

            key = cv2.waitKey(250) & 0xFF
            if key == 32:
                while True:
                    key2 = cv2.waitKey(1) & 0xFF
                    if key2 == 27:
                        key = 27
                        break
                    elif key2 == 32:
                        break
            elif key == 27:
                break

    print()
    while True:
        print("Press 'space' for another...", end="\r")

        cv2.imshow(window_adv, pert)
        cv2.imshow('adversarial image', adv)

        key = cv2.waitKey(250) & 0xFF

        if key == 27:
            break
        elif key == 32:
            break
        elif key == ord('s'):
            cv2.imwrite('img_adv.png', adv)
            cv2.imwrite('perturbation.png', pert)
    print(' '*50)
    if key == 27:
        break
print()
cv2.destroyAllWindows()
