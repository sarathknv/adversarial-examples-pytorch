import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.autograd import Variable

import numpy as np
import cv2
import argparse
import time
import shutil

# from tensorboardX import SummaryWriter

from prepare_dataset import load_dataset
import target_models


def save_checkpoint(state, checkpoint_name, best_name):
    torch.save(state, checkpoint_name)
    if state['is_best']==True:
        shutil.copyfile(checkpoint_name, best_name)


def train(model, train_loader, criterion, optimizer, epoch, epochs):
    model.train()

    n = 0
    train_loss, train_acc = 0.0, 0.0

    for i, (X, y) in enumerate(train_loader):
        X = Variable(X.float().cuda())
        y = Variable(y.long().cuda())

        optimizer.zero_grad()

        out = model(X)
        _, y_pred = torch.max(out.data, 1)

        loss = criterion(out, y)
        loss.backward()

        optimizer.step()

        train_loss += loss.item() * X.size(0)
        train_acc += torch.sum(y_pred == y.data).item()

        n += X.size(0)

        print('Train [%2d/%2d]: [%4d/%4d]\tLoss: %1.4f\tAcc: %1.8f'
            %(epoch+1, epochs, i+1, len(train_loader), train_loss/n, train_acc/n), end="\r")

    return train_loss/n, train_acc/n


def test(model, test_loader, criterion, epoch, epochs):
    model.eval()

    n = 0
    test_loss, test_acc = 0.0, 0.0

    for i, (X, y) in enumerate(test_loader):
        X = Variable(X.float().cuda())
        y = Variable(y.long().cuda())

        out = model(X)
        _, y_pred = torch.max(out.data, 1)

        loss = criterion(out, y)

        test_loss += loss.item() * X.size(0)
        test_acc += torch.sum(y_pred == y.data).item()

        n += X.size(0)

        print('Test [%2d/%2d]: [%4d/%4d]\tLoss: %1.4f\tAcc: %1.4f'
            %(epoch+1, epochs, i+1, len(test_loader), test_loss/n, test_acc/n), end="\r")

    return test_loss/n, test_acc/n




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train your model..')

    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fmnist', 'cifar10', 'cifar100'], required=False, help='dataset (default: mnist)')
    parser.add_argument('--model', type=str, default="Model_A", required=False, choices=["Model_A", "Model_B", "Model_C"], help='model name (default: Model_C)')
    parser.add_argument('--pretrained', type=int, default=1, choices=[0, 1], required=False, help='load imagenet weights? (default: True)')

    parser.add_argument('--epochs', type=int, default=50, required=False, help='no. of epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=128, required=False, help='batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.01, required=False, help='learning rate (default: 0.01)')

    parser.add_argument('--seed', type=int, default=0, required=False, help='random seed (default: 0)')
    parser.add_argument('--num_workers', type=int, default=4, required=False, help='no. of workers (default: 4)')

    args = parser.parse_args()
    dataset_name = args.dataset
    model_name = args.model
    pretrained = args.pretrained
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    seed = args.seed
    num_workers = args.num_workers

    print('dataset:    ', dataset_name)
    print('model:      ', model_name)
    print('pretrained: ', pretrained)
    print('epochs:     ', epochs)
    print('batch_size: ', batch_size)
    print('lr:         ', lr)
    print('seed:       ', seed)
    print('num_workers: ', num_workers)
    print('-'*30)


    torch.manual_seed(seed)

    # load dataset
    train_data, test_data, in_channels, num_classes = load_dataset(dataset_name)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)


    # load model
    model = getattr(target_models, model_name)(in_channels, num_classes)
    model.cuda()


    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().cuda()
    scheduler = StepLR(optimizer, step_size=5, gamma=0.3)


    # writer = SummaryWriter(log_dir="visualization/graphs", comment="loss_acc_curves")


    best_acc = 0
    for epoch in range(epochs):

        t0 = time.time()
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch, epochs)
        t1 = time.time()

        test_loss, test_acc = test(model, test_loader, criterion, epoch, epochs)

        scheduler.step()

        print("  "*40)
        print('Epoch [%3d/%3d]'%(epoch+1, epochs))
        print('-'*30)
        print('Total Time: %0.2f'%(t1 - t0))

        print('Train loss: %.8f' %(train_loss))
        print('Test loss:  %.8f' %(test_loss))

        print('Train Acc: %.8f' %(train_acc))
        print('Test Acc:  %.8f' %(test_acc))
        print()

        # writer.add_scalar('train_loss', train_loss, epoch)
        # writer.add_scalar('train_acc', train_acc, epoch)
        # writer.add_scalar('test_loss', test_loss, epoch)
        # writer.add_scalar('test_acc', test_acc, epoch)

        is_best = test_acc > best_acc
        best_acc = max(best_acc, test_acc)
        save_checkpoint({"epoch": epoch,
                        "state_dict": model.state_dict(),
                        "best_acc": best_acc,
                        "optimizer": optimizer.state_dict(),
                        "is_best": is_best,
                        }, checkpoint_name="saved/target_models/checkpoint_%d_%s_%s.pth.tar"%(epoch, model_name, dataset_name),
                            best_name="best_%s_%s.pth.tar"%(model_name, dataset_name))

    # writer.close()
