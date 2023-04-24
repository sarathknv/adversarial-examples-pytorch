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
    cv2.createTrackbar('rot (0 to 3600)', window_name, 100, 720, nothing)

    val_data = datasets.CIFAR10('./data', train=False, download=True)

    img_np, label = val_data[0]
    img_np = np.asarray(img_np)
    img_np_320 = cv2.resize(img_np, (0, 0), fx=10, fy=10, interpolation=cv2.INTER_CUBIC)

    img_t = kornia.utils.image_to_tensor(img_np.copy(), keepdim=False)  # (B, C, H, W)
    img_t = img_t / 255.0

    while True:
        angle = cv2.getTrackbarPos('rot (0 to 3600)', window_name)
        print(angle)

        angle = torch.ones(1) * angle
        center = torch.ones(1, 2)
        center[..., 0] = img_t.shape[3] / 2
        center[..., 1] = img_t.shape[2] / 2

        scale = torch.ones(1, 2)

        M = kornia.geometry.get_rotation_matrix2d(center, angle, scale)
        _, _, h, w = img_t.shape
        img_new_t = kornia.geometry.warp_affine(img_t, M, dsize=(h, w))

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
