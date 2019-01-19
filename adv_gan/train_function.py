import torch
from torch.autograd import Variable


def train(G, D, f, target, is_targeted, thres, criterion_adv, criterion_gan, alpha, beta, train_loader, optimizer_G, optimizer_D, epoch, epochs, device, num_steps=3, verbose=True):
    n = 0
    acc = 0
    num_steps = num_steps

    G.train()
    D.train()
    for i, (img, label) in enumerate(train_loader):
        valid = Variable(torch.FloatTensor(img.size(0), 1).fill_(1.0).to(device), requires_grad=False)
        fake = Variable(torch.FloatTensor(img.size(0), 1).fill_(0.0).to(device), requires_grad=False)

        img_real = Variable(img.to(device))

        optimizer_G.zero_grad()

        pert = torch.clamp(G(img_real), -thres, thres)
        img_fake = pert + img_real
        img_fake = img_fake.clamp(min=0, max=1)

        y_pred = f(img_fake)

        if is_targeted:
            y_target = Variable(torch.ones_like(label).fill_(target).to(device))
            loss_adv = criterion_adv(y_pred, y_target, is_targeted)
            acc += torch.sum(torch.max(y_pred, 1)[1] == y_target).item()
        else:
            y_true = Variable(label.to(device))
            loss_adv = criterion_adv(y_pred, y_true, is_targeted)
            acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item()

        loss_gan = criterion_gan(D(img_fake), valid)
        loss_hinge = torch.mean(torch.max(torch.zeros(1, ).type(y_pred.type()), torch.norm(pert.view(pert.size(0), -1), p=2, dim=1) - thres))

        loss_g = loss_adv + alpha*loss_gan + beta*loss_hinge

        loss_g.backward(torch.ones_like(loss_g))
        optimizer_G.step()

        optimizer_D.zero_grad()
        if i % num_steps == 0:
            loss_real = criterion_gan(D(img_real), valid)
            loss_fake = criterion_gan(D(img_fake.detach()), fake)

            loss_d = 0.5*loss_real + 0.5*loss_fake

            loss_d.backward(torch.ones_like(loss_d))
            optimizer_D.step()

        n += img.size(0)

        print("Epoch [%d/%d]: [%d/%d], D Loss: %1.4f, G Loss: %3.4f [H %3.4f, A %3.4f], Acc: %.4f"
            %(epoch+1, epochs, i, len(train_loader), loss_d.mean().item(), loss_g.mean().item(), loss_hinge.mean().item(), loss_adv.mean().item(), acc/n) , end="\r")
    return acc/n
