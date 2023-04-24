import torch
import torch.backends
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

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
    train_data = datasets.CIFAR10('./data', train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomCrop(32, padding=4),
                                      transforms.RandomRotation(20),
                                      transforms.ToTensor(),
                                  ]))

    val_data = datasets.CIFAR10('./data', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                ]))

    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)

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

    val_acc = validate(model, val_loader, args.device)
    print('Baseline accuracy: {}'.format(val_acc))

    # Semantic adversarial examples.
    model.eval()
    total_samples = 0
    correct_pred = 0
    correct_pred_adv = 0
    accuracy = None

    with tqdm(val_loader, desc='adv') as pbar:
        for i, (x, y) in enumerate(pbar):
            x = x.float().to(args.device)
            y = y.long().to(args.device)

            outputs = model(x)
            _, y_pred = torch.max(outputs.data, 1)

            x_adv, y_adv, factor = attacks.hue_gradient(
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
            # x_adv, y_adv = attacks.hue_random(
            #     x,
            #     y,
            #     model,
            #     alpha=alpha,
            #     beta=beta,
            #     max_iter=max_iter,
            #     device=args.device,
            #     verbose=False
            # )

            correct_pred += (y_pred == y).sum().item()
            correct_pred_adv += (y_adv == y).sum().item()
            total_samples += x.size(0)
            accuracy = correct_pred / total_samples
            accuracy_adv = correct_pred_adv / total_samples
            pbar.set_postfix(acc=accuracy, acc_adv=accuracy_adv)
    print('Original accuracy: {}\nAttack accuracy: {}'.format(accuracy, accuracy_adv))


if __name__ == '__main__':
    class Args:
        alpha = -torch.pi
        beta = torch.pi
        max_iter = 10  # T
        step_size = 2.5 * (beta - alpha) / (2 * max_iter)

        checkpoint = 'resnet50.pth.tar'
        num_workers = 0
        batch_size = 100

        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        # device = torch.device('cpu')
    args = Args()

    main(args)
