import math
import torch
import torch.nn.functional as F


def image_metrics(pred, target):
    err = (pred - target).float()
    mse = float(err.square().mean())
    psnr = float('inf') if mse == 0 else -10.0 * math.log10(mse)
    mu_x = F.avg_pool2d(pred, 7, 1, 3)
    mu_y = F.avg_pool2d(target, 7, 1, 3)
    var_x = F.avg_pool2d(pred * pred, 7, 1, 3) - mu_x * mu_x
    var_y = F.avg_pool2d(target * target, 7, 1, 3) - mu_y * mu_y
    cov = F.avg_pool2d(pred * target, 7, 1, 3) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2))
    ssim = float(ssim_map.mean())
    return {'mse': mse, 'psnr': psnr, 'ssim': ssim}
