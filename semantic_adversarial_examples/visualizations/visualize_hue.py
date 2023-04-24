import numpy as np
import torch
import torch.backends
from torchvision import datasets, transforms
import cv2
import kornia


def nothing(x):
    pass


def main(args):
    window_name = 'img_new'
    cv2.namedWindow(window_name)
    cv2.namedWindow('img')
    cv2.createTrackbar('h (-pi to pi)', window_name, 500, 1000, nothing)

    val_data = datasets.CIFAR10('./data', train=False, download=True)

    img_np, label = val_data[0]
    img_np = np.asarray(img_np)
    img_np_320 = cv2.resize(img_np, (0, 0), fx=10, fy=10, interpolation=cv2.INTER_CUBIC)

    img_t = kornia.utils.image_to_tensor(img_np.copy(), keepdim=False)  # (B, C, H, W)
    img_t = img_t / 255.0

    while True:
        factor = cv2.getTrackbarPos('h (-pi to pi)', window_name)
        factor = ((factor - 500) / 500) * torch.pi  # -pi to pi
        print(factor / torch.pi)

        img_new_t = kornia.enhance.adjust_hue(img_t, factor)

        img_new_np = kornia.utils.tensor_to_image(img_new_t) * 255.0
        img_new_np = img_new_np.astype(np.uint8)
        img_new_np_320 = cv2.resize(img_new_np, (0, 0), fx=10, fy=10, interpolation=cv2.INTER_CUBIC)

        cv2.imshow(window_name, img_new_np_320)
        cv2.imshow('img', img_np_320)
        key = cv2.waitKey(50) & 0xFF
        if key == 27:
            break

    cv2.destroyAllWindows()
    for i in range (1,5):
        cv2.waitKey(1)


if __name__ == '__main__':
    class Args:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')


    args = Args()

    main(args)
