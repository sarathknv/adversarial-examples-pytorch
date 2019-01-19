""" Spatially Transformed Adversarial Examples
    Paper link: https://arxiv.org/abs/1801.02612
"""
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
import argparse
from model_mnist import Basic_CNN


def CWLoss(logits, target, kappa=0):
    # inputs to the softmax function are called logits.
    # https://arxiv.org/pdf/1608.04644.pdf
    target = torch.ones(logits.size(0)).type(logits.type()).fill_(target)
    target_one_hot = torch.eye(10).type(logits.type())[target.long()]

    # workaround here.
    # subtract large value from target class to find other max value
    # https://github.com/carlini/nn_robust_attacks/blob/master/l2_attack.py
    real = torch.sum(target_one_hot*logits, 1)
    other = torch.max((1-target_one_hot)*logits - (target_one_hot*10000), 1)[0]
    kappa = torch.zeros_like(other).fill_(kappa)

    return torch.sum(torch.max(other-real, kappa))


class Loss_flow(nn.Module):
    def __init__(self, neighbours=np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])):
        super(Loss_flow, self).__init__()

        filters = []
        for i in range(neighbours.shape[0]):
            for j in range(neighbours.shape[1]):
                if neighbours[i][j] == 1:
                    filter = np.zeros((1, neighbours.shape[0], neighbours.shape[1]))
                    filter[0][i][j] = -1
                    filter[0][neighbours.shape[0]//2][neighbours.shape[1]//2] = 1
                    filters.append(filter)

        filters = np.array(filters)
        self.filters = torch.from_numpy(filters).float()

    def forward(self, f):
        # TODO: padding
        '''
        f - f.size() =  [1, h, w, 2]
            f[0, :, :, 0] - u channel
            f[0, :, :, 1] - v channel
        '''
        f_u = f[:, :, :, 0].unsqueeze(1)
        f_v = f[:, :, :, 1].unsqueeze(1)

        diff_u = F.conv2d(f_u, self.filters)[0][0] # don't use squeeze
        diff_u_sq = torch.mul(diff_u, diff_u)

        diff_v = F.conv2d(f_v, self.filters)[0][0] # don't use squeeze
        diff_v_sq = torch.mul(diff_v, diff_v)

        dist = torch.sqrt(torch.sum(diff_u_sq, dim=0) + torch.sum(diff_v_sq, dim=0))
        return torch.sum(dist)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, default='images/1.jpg', help='path to image')
    parser.add_argument('--target', type=int, required=True, help='Target label')
    parser.add_argument('--gpu', action="store_true", default=False)
    parser.add_argument('--tau', type=float, required=False, default=10, help='balance flow loss')
    parser.add_argument('--lr', type=float, required=False, default=0.005, help='Learning rate')

    args = parser.parse_args()
    img_path = args.img
    target   = args.target
    gpu = args.gpu
    tau = args.tau
    lr = args.lr
    IMG_SIZE = 28
    mean = 0 # for flow initialization
    std = 0.01

    print('Spatially Transformed Adversarial Examples')
    print()


    orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
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


    # prediction before attack
    x = Variable(torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0), requires_grad=True)

    out = model(x)
    pred = np.argmax(out.data.cpu().numpy())
    print('Prediction before attack: %s' %(pred))

    if pred == target:
        print('Prediction is same as target class.')
        exit()


    # flow, grid, loss_functions
    theta = torch.tensor([[1, 0, 0], [0, 1, 0]]).unsqueeze(0).float() # identity transformation
    grid = F.affine_grid(theta, x.size()) # flow = 0. This is base grid
    # grid.size() = (1, h, w, 2)

    f = Variable(torch.zeros_like(grid).float(), requires_grad=True)
    torch.nn.init.normal_(f, mean=0, std=0.01)

    grid_new = grid + f
    grid_new = grid_new.clamp(min=-1, max=1)
    x_new = F.grid_sample(x, grid_new, mode='bilinear')


    optimizer = torch.optim.SGD([f,], lr=lr) # optimizer = torch.optim.LBFGS([f, ], lr=lr)

    loss_flow = Loss_flow()
    loss_adv = CWLoss

    i=0
    while True:
        optimizer.zero_grad()

        logits = model(x_new) # .detach() for LBFGS
        pred = np.argmax(logits.data.numpy())

        loss = loss_adv(logits, target) + tau*loss_flow(f)
        loss.backward()
        optimizer.step()

        # update variables and predict on adversarial image
        grid_new = grid + f
        grid_new = grid_new.clamp(min=-1, max=1)
        x_new = F.grid_sample(x, grid_new, mode='bilinear')

        pred_adv = np.argmax(model(x_new).data.numpy())

        i+=1
        print("step %d: [%d] \t" %(i, pred_adv))


        adv = x_new.data[0][0]
        adv = np.clip(adv.numpy(), -1, 1)
        adv = (adv * 0.5 + 0.5)*255
        adv = adv.astype(np.uint8)

        cv2.imshow('adv', adv)
        cv2.imshow('orig', orig)
        key = cv2.waitKey(500) & 0xFF
        key2 = 0
        if key == 32:
            while True:
                key2 = cv2.waitKey(100) & 0xFF
                if key2 == 32 or key2 == 27:
                    break
                if key2 == ord('s'):
                    cv2.imwrite('adv.png', adv)
                    cv2.imwrite('orig.png', orig)
        if pred_adv == target:
            while True:
                key2 = cv2.waitKey(100) & 0xFF
                if key2 == 32 or key2 == 27:
                    break
                if key2 == ord('s'):
                    cv2.imwrite('images/results/%d_%d.png'%(9, target), adv)

        if key == 27 or key2 == 27:
            break
    print()
    cv2.destroyAllWindows()
