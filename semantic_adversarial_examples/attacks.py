import torch
import numpy as np
import kornia
from tqdm import tqdm


def hue_gradient(x, y, model, criterion, step_size=0.001, alpha=-np.pi,
                 beta=np.pi, max_iter=10, device='cpu', verbose=False):
    model.eval()

    factor = np.random.uniform(alpha, beta, size=x.size(0))
    factor = torch.asarray(factor, dtype=torch.float32, device=device, requires_grad=True)
    x_adv = kornia.enhance.adjust_hue(x, factor)

    logits = model(x_adv)
    _, y_adv = torch.max(logits, 1)
    loss = criterion(logits, y_adv)
    loss.backward()

    with tqdm(range(max_iter), desc='hue_grad', disable=not verbose) as pbar:
        for i in pbar:
            factor.data = torch.clamp(factor.data + step_size * torch.sign(factor.grad), min=alpha, max=beta)
            factor.grad = None
            x_adv = kornia.enhance.adjust_hue(x, factor)

            logits = model(x_adv)
            _, y_adv = torch.max(logits, 1)

            asr = (y != y_adv).sum().item() / x_adv.size(0)
            pbar.set_postfix(asr=asr, factor=factor.mean().item())
            loss = criterion(logits, y_adv)
            loss.backward()
    return x_adv, y_adv, factor.detach()


def hue_random(x, y, model, alpha=-np.pi, beta=np.pi,
               max_iter=10000, device='cpu', verbose=False):
    model.eval()

    mask = torch.ones(x.size(0), dtype=torch.bool, device=device)
    factor = torch.zeros(x.size(0), dtype=torch.float32, device=device)

    with tqdm(range(max_iter), desc='hue_rand', disable=not verbose) as pbar:
        for i in pbar:
            temp = torch.empty_like(factor).uniform_(alpha, beta)
            factor = mask * factor + ~mask * temp
            x_adv = kornia.enhance.adjust_hue(x, factor)

            logits = model(x_adv)
            _, y_adv = torch.max(logits, 1)

            asr = (y != y_adv).sum().item() / x_adv.size(0)
            pbar.set_postfix(asr=asr, factor=factor.mean().item())

            # Stop updating factor if y_adv != y for that sample.
            mask = y_adv != y
            if mask.sum() == x.size(0):
                pbar.close()
                break
    return x_adv, y_adv, factor.detach()


def saturation_gradient(x, y, model, criterion, step_size=0.001, alpha=0.7,
                        beta=1.3, max_iter=10, device='cpu', verbose=False):
    model.eval()

    factor = np.random.uniform(alpha, beta, size=x.size(0))
    factor = torch.asarray(factor, dtype=torch.float32, device=device, requires_grad=True)
    x_adv = kornia.enhance.adjust_saturation(x, factor)

    logits = model(x_adv)
    _, y_adv = torch.max(logits, 1)
    loss = criterion(logits, y_adv)
    loss.backward()

    with tqdm(range(max_iter), desc='saturation_grad', disable=not verbose) as pbar:
        for i in pbar:
            factor.data = torch.clamp(factor.data + step_size * torch.sign(factor.grad), min=alpha, max=beta)
            factor.grad = None
            x_adv = kornia.enhance.adjust_saturation(x, factor)

            logits = model(x_adv)
            _, y_adv = torch.max(logits, 1)

            asr = (y != y_adv).sum().item() / x_adv.size(0)
            pbar.set_postfix(asr=asr, factor=factor.mean().item())
            loss = criterion(logits, y_adv)
            loss.backward()
    return x_adv, y_adv, factor.detach()


def saturation_random(x, y, model, alpha=0.7, beta=1.3,
                      max_iter=1000, device='cpu', verbose=False):
    model.eval()

    mask = torch.ones(x.size(0), dtype=torch.bool, device=device)
    factor = torch.ones(x.size(0), dtype=torch.float32, device=device)

    with tqdm(range(max_iter), desc='saturation_rand', disable=not verbose) as pbar:
        for i in pbar:
            temp = torch.empty_like(factor).uniform_(alpha, beta)
            factor = mask * factor + ~mask * temp
            x_adv = kornia.enhance.adjust_saturation(x, factor)

            logits = model(x_adv)
            _, y_adv = torch.max(logits, 1)

            asr = (y != y_adv).sum().item() / x_adv.size(0)
            pbar.set_postfix(asr=asr, factor=factor.mean().item())

            # Stop updating factor if y_adv != y for that sample.
            mask = y_adv != y
            if mask.sum() == x.size(0):
                pbar.close()
                break
    return x_adv, y_adv, factor.detach()


def brightness_gradient(x, y, model, criterion, step_size=0.001, alpha=-0.2,
                        beta=0.2, max_iter=10, device='cpu', verbose=False):
    model.eval()

    factor = np.random.uniform(alpha, beta, size=x.size(0))
    factor = torch.asarray(factor, dtype=torch.float32, device=device, requires_grad=True)
    x_adv = kornia.enhance.adjust_brightness(x, factor)

    logits = model(x_adv)
    _, y_adv = torch.max(logits, 1)
    loss = criterion(logits, y_adv)
    loss.backward()

    with tqdm(range(max_iter), desc='brightness_grad', disable=not verbose) as pbar:
        for i in pbar:
            factor.data = torch.clamp(factor.data + step_size * torch.sign(factor.grad), min=alpha, max=beta)
            factor.grad = None
            x_adv = kornia.enhance.adjust_brightness(x, factor)

            logits = model(x_adv)
            _, y_adv = torch.max(logits, 1)

            asr = (y != y_adv).sum().item() / x_adv.size(0)
            pbar.set_postfix(asr=asr, factor=factor.mean().item())
            loss = criterion(logits, y_adv)
            loss.backward()
    return x_adv, y_adv, factor.detach()


def brightness_random(x, y, model, alpha=-0.2, beta=0.2,
                      max_iter=1000, device='cpu', verbose=False):
    model.eval()

    mask = torch.ones(x.size(0), dtype=torch.bool, device=device)
    factor = torch.zeros(x.size(0), dtype=torch.float32, device=device)

    with tqdm(range(max_iter), desc='brightness_rand', disable=not verbose) as pbar:
        for i in pbar:
            temp = torch.empty_like(factor).uniform_(alpha, beta)
            factor = mask * factor + ~mask * temp
            x_adv = kornia.enhance.adjust_brightness(x, factor)

            logits = model(x_adv)
            _, y_adv = torch.max(logits, 1)

            asr = (y != y_adv).sum().item() / x_adv.size(0)
            pbar.set_postfix(asr=asr, factor=factor.mean().item())

            # Stop updating factor if y_adv != y for that sample.
            mask = y_adv != y
            if mask.sum() == x.size(0):
                pbar.close()
                break
    return x_adv, y_adv, factor.detach()


def contrast_gradient(x, y, model, criterion, step_size=0.001, alpha=0.7,
                      beta=1.3, max_iter=10, device='cpu', verbose=False):
    model.eval()

    factor = np.random.uniform(alpha, beta, size=x.size(0))
    factor = torch.asarray(factor, dtype=torch.float32, device=device, requires_grad=True)
    x_adv = kornia.enhance.adjust_contrast(x, factor)

    logits = model(x_adv)
    _, y_adv = torch.max(logits, 1)
    loss = criterion(logits, y_adv)
    loss.backward()

    with tqdm(range(max_iter), desc='contrast_grad', disable=not verbose) as pbar:
        for i in pbar:
            factor.data = torch.clamp(factor.data + step_size * torch.sign(factor.grad), min=alpha, max=beta)
            factor.grad = None
            x_adv = kornia.enhance.adjust_contrast(x, factor)

            logits = model(x_adv)
            _, y_adv = torch.max(logits, 1)

            asr = (y != y_adv).sum().item() / x_adv.size(0)
            pbar.set_postfix(asr=asr, factor=factor.mean().item())
            loss = criterion(logits, y_adv)
            loss.backward()
    return x_adv, y_adv, factor.detach()


def contrast_random(x, y, model, alpha=0.7, beta=1.3,
                    max_iter=1000, device='cpu', verbose=False):
    model.eval()

    mask = torch.ones(x.size(0), dtype=torch.bool, device=device)
    factor = torch.zeros(x.size(0), dtype=torch.float32, device=device)

    with tqdm(range(max_iter), desc='contrast_rand', disable=not verbose) as pbar:
        for i in pbar:
            temp = torch.empty_like(factor).uniform_(alpha, beta)
            factor = mask * factor + ~mask * temp
            x_adv = kornia.enhance.adjust_contrast(x, factor)

            logits = model(x_adv)
            _, y_adv = torch.max(logits, 1)

            asr = (y != y_adv).sum().item() / x_adv.size(0)
            pbar.set_postfix(asr=asr, factor=factor.mean().item())

            # Stop updating factor if y_adv != y for that sample.
            mask = y_adv != y
            if mask.sum() == x.size(0):
                pbar.close()
                break
    return x_adv, y_adv, factor.detach()


def rotation_gradient(x, y, model, criterion, step_size=0.001, alpha=-10,
                      beta=10, max_iter=10, device='cpu', verbose=False):
    model.eval()

    b, c, h, w = x.size()
    angle = np.random.uniform(alpha, beta, size=b)
    angle = torch.asarray(angle, dtype=torch.float32, device=device, requires_grad=True)

    center = torch.ones(b, 2).to(device)
    center[..., 0] = w / 2
    center[..., 1] = h / 2

    scale = torch.ones(b, 2).to(device)

    M = kornia.geometry.get_rotation_matrix2d(center, angle, scale)
    x_adv = kornia.geometry.warp_affine(x, M, dsize=(h, w))

    logits = model(x_adv)
    _, y_adv = torch.max(logits, 1)
    loss = criterion(logits, y_adv)
    loss.backward()

    with tqdm(range(max_iter), desc='rotation_grad', disable=not verbose) as pbar:
        for i in pbar:
            angle.data = torch.clamp(angle.data + step_size * torch.sign(angle.grad), min=alpha, max=beta)
            angle.grad = None

            M = kornia.geometry.get_rotation_matrix2d(center, angle, scale)
            x_adv = kornia.geometry.warp_affine(x, M, dsize=(h, w))

            logits = model(x_adv)
            _, y_adv = torch.max(logits, 1)

            asr = (y != y_adv).sum().item() / x_adv.size(0)
            pbar.set_postfix(asr=asr, factor=angle.mean().item())
            loss = criterion(logits, y_adv)
            loss.backward()
    return x_adv, y_adv, angle.detach()


def rotation_random(x, y, model, alpha=-10, beta=10,
                    max_iter=1000, device='cpu', verbose=False):
    model.eval()

    mask = torch.ones(x.size(0), dtype=torch.bool, device=device)
    b, c, h, w = x.size()
    angle = np.random.uniform(alpha, beta, size=b)
    angle = torch.asarray(angle, dtype=torch.float32, device=device, requires_grad=True)

    center = torch.ones(1, 2)
    center[..., 0] = w / 2
    center[..., 1] = h / 2

    scale = torch.ones(1, 2)

    with tqdm(range(max_iter), desc='rotation_rand', disable=not verbose) as pbar:
        for i in pbar:
            temp = torch.empty_like(angle).uniform_(alpha, beta)
            angle = mask * angle + ~mask * temp

            M = kornia.geometry.get_rotation_matrix2d(center, angle, scale)
            x_adv = kornia.geometry.warp_affine(x, M, dsize=(h, w))

            logits = model(x_adv)
            _, y_adv = torch.max(logits, 1)

            asr = (y != y_adv).sum().item() / x_adv.size(0)
            pbar.set_postfix(asr=asr, factor=angle.mean().item())

            # Stop updating factor if y_adv != y for that sample.
            mask = y_adv != y
            if mask.sum() == x.size(0):
                pbar.close()
                break
    return x_adv, y_adv, angle.detach()
