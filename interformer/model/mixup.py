import torch
import numpy as np


def manifold_mixup(x, lam):
    # add the virtual node
    N = x.size(0)
    final_x = lam * x[torch.arange(0, N, 2), :] + (1. - lam) * x[torch.arange(1, N, 2), :]
    return final_x


def mixup_losses(loss_fns, y_hat, y_gt, lam):
    # parameters: y_gt=(N, 1)
    all_losses = []
    for i, (loss_fn, w) in enumerate(loss_fns):
        N = y_gt.size(0)
        l1 = loss_fn(y_hat[i], y_gt[torch.arange(0, N, 2)])
        l2 = loss_fn(y_hat[i], y_gt[torch.arange(1, N, 2)])
        total_loss = lam * l1 + (1. - lam) * l2
        all_losses.append(total_loss * w)
    loss = sum(all_losses)
    return loss
