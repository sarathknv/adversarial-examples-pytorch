"""One Pixel Attack

    Paper: https://arxiv.org/abs/1710.08864

    d - number of pixels to change (L0 norm)
    iters - number of iterations
    popsize - population size

"""

from torchvision import models
import torch
import cv2
import numpy as np
from scipy.optimize import differential_evolution
import torch.nn as nn
from torch.autograd import Variable
from model import BasicCNN

import argparse

cifar10_class_names = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, default='airplane.png', help='path to image')
parser.add_argument('--d', type=int, default=3, help='number of pixels to change')
parser.add_argument('--iters', type=int, default=600, help='number of iterations')
parser.add_argument('--popsize', type=int, default=10, help='population size')
parser.add_argument('--model_path', type=str, default='cifar10_basiccnn.pth.tar', help='path to trained model')

args = parser.parse_args()
image_path = args.img
d = args.d
iters = args.iters
popsize = args.popsize
model_path = args.model_path

def nothing(x):
    pass


# load image and reshape to (3, 224, 224) and RGB (not BGR)
# preprocess as described here: http://pytorch.org/docs/master/torchvision/models.html
orig = cv2.imread(image_path)[..., ::-1]
orig = cv2.resize(orig, (32, 32))
img = orig.copy()
shape = orig.shape

def preprocess(img):
    img = img.astype(np.float32)
    img /= 255.0
    img = img.transpose(2, 0, 1)
    return img

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


model = BasicCNN()
saved = torch.load(model_path, map_location='cpu')
model.load_state_dict(saved['state_dict'])
model.eval()

inp = Variable(torch.from_numpy(preprocess(img)).float().unsqueeze(0))
prob_orig = softmax(model(inp).data.numpy()[0])
pred_orig = np.argmax(prob_orig)
print('Prediction before attack: %s' %(cifar10_class_names[pred_orig]))
print('Probability: %f' %(prob_orig[pred_orig]))
print()


def perturb(x):
    adv_img = img.copy()

    # calculate pixel locations and values
    pixs = np.array(np.split(x, len(x)/5)).astype(int)
    loc = (pixs[:, 0], pixs[:,1])
    val = pixs[:, 2:]
    adv_img[loc] = val

    return adv_img

def optimize(x):
    adv_img = perturb(x)

    inp = Variable(torch.from_numpy(preprocess(adv_img)).float().unsqueeze(0))
    out = model(inp)
    prob = softmax(out.data.numpy()[0])

    return prob[pred_orig]

pred_adv = 0
prob_adv = 0
def callback(x, convergence):
    global pred_adv, prob_adv
    adv_img = perturb(x)

    inp = Variable(torch.from_numpy(preprocess(adv_img)).float().unsqueeze(0))
    out = model(inp)
    prob = softmax(out.data.numpy()[0])

    pred_adv = np.argmax(prob)
    prob_adv = prob[pred_adv]
    if pred_adv != pred_orig and prob_adv >= 0.9:
        print('Attack successful..')
        print('Prob [%s]: %f' %(cifar10_class_names[pred_adv], prob_adv))
        print()
        return True
    else:
        print('Prob [%s]: %f' %(cifar10_class_names[pred_orig], prob[pred_orig]))


def scale(x, scale=5):
    return cv2.resize(x, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


while True:
    bounds = [(0, shape[0]-1), (0, shape[1]), (0, 255), (0, 255), (0, 255)] * d
    result = differential_evolution(optimize, bounds, maxiter=iters, popsize=popsize, tol=1e-5, callback=callback)

    adv_img = perturb(result.x)
    inp = Variable(torch.from_numpy(preprocess(adv_img)).float().unsqueeze(0))
    out = model(inp)
    prob = softmax(out.data.numpy()[0])
    print('Prob [%s]: %f --> Prob[%s]: %f' %(cifar10_class_names[pred_orig], prob_orig[pred_orig], cifar10_class_names[pred_adv], prob_adv))

    cv2.imshow('adversarial image', scale(adv_img[..., ::-1]))

    key = 0
    while True:
        print("Press 'esc' to exit, 'space' to re-run..", end="\r")
        key = cv2.waitKey(100) & 0xFF
        if key == 27:
            break
        elif key == ord('s'):
            cv2.imwrite('adv_img.png', scale(adv_img[..., ::-1]))
        elif key == 32:
            break
    if key == 27:
        break
cv2.destroyAllWindows()
