import torch
import torch.backends
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import cv2

import attacks
from models import ResNet50, VGG16


def validate(model, val_loader, device):
    model.eval()
    total_samples = 0
    correct_pred = 0
    accuracy = None

    with torch.no_grad():
        with tqdm(val_loader, desc='Val') as pbar:
            for i, (x, y) in enumerate(pbar):
                x = x.float().to(device)
                y = y.long().to(device)

                outputs = model(x)
                _, y_pred = torch.max(outputs.data, 1)

                correct_pred += (y_pred == y).sum().item()
                total_samples += x.size(0)
                accuracy = correct_pred / total_samples
                pbar.set_postfix(acc=accuracy)
    return accuracy


def main(args):
    val_data = datasets.CIFAR10('./data', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                ]))

    val_loader = DataLoader(val_data,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers)

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    # model = VGG16()
    model = ResNet50()
    model.load_state_dict(checkpoint['state_dict'])
    model.to(args.device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    # val_acc = validate(model, val_loader, args.device)
    # print('Baseline accuracy: {}'.format(val_acc))

    # Semantic adversarial examples.
    model.eval()
    count = 0
    with tqdm(val_loader, desc='adv') as pbar:
        for i, (x, y) in enumerate(pbar):
            x = x.float().to(args.device)
            y = y.long().to(args.device)

            outputs = model(x)
            _, y_pred = torch.max(outputs.data, 1)

            x_adv, y_adv, factor = attacks.contrast_gradient(
                x,
                y,
                model,
                criterion,
                step_size=args.step_size,
                alpha=args.alpha,
                beta=args.beta,
                max_iter=args.max_iter,
                device=args.device,
                verbose=False
            )

            if y_adv.item() != y_pred.item() and y_pred.item() == y.item():
                img = x.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0
                img = img.astype(np.uint8)
                img = img[..., ::-1]

                img_adv = x_adv.detach().squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0
                img_adv = img_adv.astype(np.uint8)
                img_adv = img_adv[..., ::-1]

                cv2.imwrite('examples/contrast/{}_y_{}_y_pred_{}_y_adv_{}_factor_{:.3f}.png'
                            .format(i, y.item(), y_pred.item(), y_adv.item(), factor.item()), img_adv)
                cv2.imwrite('examples/contrast/{}_y_{}_y_pred_{}.png'
                            .format(i, y.item(), y_pred.item()), img)
                count += 1
            pbar.set_postfix(num_examples='{}/{}'.format(count, args.num_examples))

            if count == args.num_examples:
                break


if __name__ == '__main__':
    class Args:
        alpha = 0.7
        beta = 1.3
        max_iter = 10  # T
        step_size = 2.5 * (beta - alpha) / (2 * max_iter)

        checkpoint = 'weights/resnet50.pth.tar'
        num_workers = 0
        batch_size = 1
        num_examples = 20

        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        # device = torch.device('cpu')
    args = Args()

    main(args)
