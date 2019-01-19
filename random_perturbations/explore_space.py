""" Generate random perturbations.
There are two random vectors here (tensors of shape (32, 32, 3)),
    vec1 - max norm, eps
    vec2 - L2 norm, rad

    vec2 lies on a unit hypersphere and vec1 on hypercube

Controls:
    'r' - generate new vec2
    'e' - generate new vec1

Basically, perturbation vec1 is added to input image to check how robust the classifier is.
And to explore a small region around the current adversarial image, (you can imagine them to be
vectors in 32*32*4 dimensional space) we add vec2, a random vector inside a unit hypersphere. Radius can
be increased by changing rad. Press 'r' and 'e' for changing vec2 and vec1 respectively.

From 'Explaining and Harnessing Adversarial Examples' - https://arxiv.org/abs/1412.6572,
    '''
    The direction of perturbation, rather than the specific point in space, matters most.
    Space is not full of pockets of adversarial examples that finely tile the reals like the rational numbers.
    '''
This code is to test this.

"""
import numpy as np
import cv2
from torch.autograd import Variable
import argparse
import torch
from models import BasicCNN

np.random.seed(0)

cifar10_class_names = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, default='images/horse.png', help='path to image')

args = parser.parse_args()
image_path = args.img

def random_vector_surface(shape=(32, 32, 3)):
    # generates a random vector on the surface of hypersphere
    mat = np.random.normal(size=shape)
    norm = np.linalg.norm(mat)
    return mat/norm

def random_vector_volume(shape=(32, 32, 3)):
    # generates a random vector in the volume of unit hypersphere
    d = np.random.rand() ** (1 / np.prod(shape))

    return random_vector_surface() * d


window_pert = 'perturbation'
cv2.namedWindow(window_pert)


eps, rad = 0, 0

def get_radius(x):
    global eps, rad
    eps = cv2.getTrackbarPos('eps', window_pert)
    rad = cv2.getTrackbarPos('radius', window_pert)


cv2.createTrackbar('eps', window_pert, 1, 255, get_radius)
cv2.createTrackbar('radius', window_pert, 0, 255, get_radius)
orig = cv2.imread(image_path)[..., ::-1] # BGR -> RGB
orig = cv2.resize(orig, (32, 32))


vec1 = random_vector_surface()
vec2 = random_vector_volume()
pert = np.zeros((32, 32, 3), dtype=np.float32)


model = BasicCNN()
saved = torch.load("cifar10_basiccnn.pth.tar", map_location='cpu')
model.load_state_dict(saved['state_dict'])
model.eval()

img = orig.astype(np.float32)/255.0
img = img.transpose(2, 0, 1)

def scale(x, scale=10):
    return cv2.resize(x, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


while True:

    key = cv2.waitKey(50) & 0xFF
    if key == 27:
        break

    elif key == ord('e'):
        vec1 = random_vector_surface()

    elif key == ord('r'):
        vec2 = random_vector_volume()

    pert = (eps/255.0) * np.sign(vec1) + (rad/255.0) * vec2

    inp = torch.from_numpy(img).float().unsqueeze(0)

    prob = softmax(model(inp).data.numpy())[0]
    pred = np.argmax(prob)

    # add perturbation to image
    inp = torch.clamp(inp + torch.from_numpy(pert.transpose(2, 0, 1)).float().unsqueeze(0), min=0, max=1)

    # predict on the adversarial image
    prob_adv = softmax(model(inp).data.numpy())[0]
    pred_adv = np.argmax(prob_adv)

    print("%s [%f] ---> %s [%f]" %(cifar10_class_names[pred], prob[pred], cifar10_class_names[pred_adv], prob_adv[pred_adv]))
    print()

    adv = inp.numpy()[0]
    adv = adv.transpose(1, 2, 0)

    adv = adv * 255.0
    adv = adv[..., ::-1] # RGB to BGR
    adv = np.clip(adv, 0, 255).astype(np.uint8)

    cv2.imshow(window_pert, scale(pert))
    cv2.imshow('vec1', scale((eps/255.0)*np.sign(vec1)))
    cv2.imshow('vec2', (rad/255.0)*scale(vec2))
    cv2.imshow('orig', scale(adv))
cv2.destroyAllWindows()
