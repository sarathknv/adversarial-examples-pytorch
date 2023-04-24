import numpy as np
from tqdm import tqdm
import random
import cv2

import torch
import torch.backends
import torch.nn as nn
from torchvision import datasets, transforms

from ..models import VGG16
import kornia


def validate(model, val_loader, device):
    model.eval()
    total_samples = 0
    correct_pred = 0
    accuracy = None

    with torch.no_grad():
        with tqdm(val_loader, desc='Val') as pbar:
            for i, (X, y) in enumerate(pbar):
                X = X.float().to(device)
                y = y.long().to(device)

                outputs = model(X)
                _, y_pred = torch.max(outputs.data, 1)

                correct_pred += (y_pred == y).sum().item()
                total_samples += X.size(0)
                accuracy = correct_pred / total_samples
                pbar.set_postfix(acc=accuracy)
    return accuracy


def main(args):
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = VGG16()
    model.load_state_dict(checkpoint['state_dict'])
    model.to(args.device)
    model.eval()

    criterion = nn.CrossEntropyLoss().to(args.device)
    val_data = datasets.CIFAR10('./data', train=False, download=True)

    img_np, label = val_data[10]
    img_np = np.asarray(img_np)
    img_np_320 = cv2.resize(img_np, (0, 0), fx=10, fy=10, interpolation=cv2.INTER_CUBIC)

    img_t = kornia.utils.image_to_tensor(img_np.copy(), keepdim=False)  # (B, C, H, W)
    img_t = img_t / 255.0
    img_t = img_t.to(args.device)

    outputs = model(img_t)
    _, y_pred = torch.max(outputs, 1)

    factor = torch.asarray(0.1 * random.uniform(-np.pi, np.pi), requires_grad=True)
    alpha = args.alpha_grad if not args.is_random else args.alpha_random

    while True:
        img_t_new = kornia.enhance.adjust_hue(img_t, factor)

        outputs_new = model(img_t_new)
        _, y_pred_new = torch.max(outputs_new, 1)

        loss = criterion(outputs_new, y_pred_new)
        loss.backward()

        print(label, y_pred.item(), y_pred_new.item(), img_t.mean().item(), img_t_new.mean().item(), factor.item(), factor.grad.item())

        # factor.data = factor.data + alpha * factor.grad
        # factor.grad = None
        factor.data = torch.clamp(factor.data + args.step_size * factor.grad, min=-3, max=3)
        factor.grad = None

        img_new_np = kornia.utils.tensor_to_image(img_t_new) * 255.0
        img_new_np = img_new_np.astype(np.uint8)
        img_new_np_320 = cv2.resize(img_new_np, (0, 0), fx=10, fy=10, interpolation=cv2.INTER_CUBIC)

        cv2.imshow('perturbed', img_new_np_320)
        cv2.imshow('img', img_np_320)
        key = cv2.waitKey(5) & 0xFF
        if key == 27:
            break

    cv2.destroyAllWindows()
    for i in range (1,5):
        cv2.waitKey(1)


if __name__ == '__main__':
    class Args:
        checkpoint = 'vgg16.pth.tar'
        num_workers = 0
        num_steps = 10
        is_random = True
        alpha_grad = 0.001
        alpha_random = 0.2
        step_size = 0.00785
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    args = Args()

    main(args)
