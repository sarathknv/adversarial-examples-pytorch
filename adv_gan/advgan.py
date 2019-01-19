import torch
import torch.nn.functional as F
import target_models
from generators import Generator_MNIST as Generator

import cv2
import numpy as np
import os
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AdvGAN for MNIST')
    parser.add_argument('--model', type=str, default="Model_C", required=False, choices=["Model_A", "Model_B", "Model_C"], help='model name (default: Model_C)')
    parser.add_argument('--target', type=int, required=False, help='Target label')
    parser.add_argument('--bound', type=float, default=0.3, choices=[0.2, 0.3], required=False, help='Perturbation bound (0.2 or 0.3)')
    parser.add_argument('--img', type=str, default='images/0.jpg', required=False, help='Image to perturb')

    args = parser.parse_args()
    model_name = args.model
    target = args.target
    thres = args.bound
    img_path = args.img

    is_targeted = False
    if target in range(0, 10):
        is_targeted = True


    # load target_model
    f = getattr(target_models, model_name)(1, 10)
    checkpoint_path_f = os.path.join('saved', 'target_models', 'best_%s_mnist.pth.tar'%(model_name))
    checkpoint_f = torch.load(checkpoint_path_f, map_location='cpu')
    f.load_state_dict(checkpoint_f["state_dict"])
    f.eval()


    # load corresponding generator
    G = Generator()
    checkpoint_name_G = '%s_target_%d.pth.tar'%(model_name, target) if is_targeted else '%s_untargeted.pth.tar'%(model_name)
    checkpoint_path_G = os.path.join('saved', 'generators', 'bound_%.1f'%(thres), checkpoint_name_G)
    checkpoint_G = torch.load(checkpoint_path_G, map_location='cpu')
    G.load_state_dict(checkpoint_G['state_dict'])
    G.eval()


    # load img and preprocess as required by f and G
    orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = orig.copy().astype(np.float32)
    img = img[None, None, :, :]/255.0


    x = torch.from_numpy(img)
    pert = G(x).data.clamp(min=-thres, max=thres)
    x_adv = x + pert
    x_adv = x_adv.clamp(min=0, max=1)


    adversarial_img = x_adv.data.squeeze().numpy()
    perturbation = pert.data.squeeze().numpy()


    # prediction before and after attack
    prob_before, y_before = torch.max(F.softmax(f(x), 1), 1)
    prob_after, y_after = torch.max(F.softmax(f(x_adv), 1), 1)

    print('Prediction before attack: %d [Prob: %0.4f]'%(y_before.item(), prob_before.item()))
    print('After attack: %d [Prob: %0.4f]'%(y_after.item(), prob_after.item()))


    while True:
        cv2.imshow('Adversarial Image', adversarial_img)
        cv2.imshow('Image', orig)

        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            break
        if key == ord('s'):
            d = 0
            adversarial_img = adversarial_img*255
            adversarial_img = adversarial_img.astype(np.uint8)
            cv2.imwrite('targeted_1_%d_%d.png'%(target, y_after.item()), adversarial_img)

cv2.destroyAllWindows()
