"""Paper metrics and the Charbonnier training loss; no prediction clipping."""

from math import exp

import torch
from torch import nn
from torch.nn import functional as F


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True,
          data_range=1.0, padding=0):
    mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padding, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True, data_range=1.0,
         padding=0):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(
        img1, img2, window, window_size, channel, size_average,
        data_range=data_range, padding=padding,
    )


class PSNR(nn.Module):
    """Per-image PSNR without silently clipping predictions or targets."""

    def __init__(self, data_range=1.0):
        super().__init__()
        if data_range <= 0:
            raise ValueError("PSNR data_range must be positive")
        self.data_range = float(data_range)

    def forward(self, prediction, target):
        if prediction.shape != target.shape:
            raise ValueError("PSNR prediction and target must have identical shapes")
        mse = (prediction - target).square().flatten(1).mean(dim=1)
        mse = mse.clamp_min(torch.finfo(mse.dtype).tiny)
        return (10.0 * torch.log10((self.data_range ** 2) / mse)).mean()


class SAM(nn.Module):
    """Mean spectral angle in degrees with explicit zero-vector handling."""

    def forward(self, prediction, target):
        if prediction.shape != target.shape:
            raise ValueError("SAM prediction and target must have identical shapes")
        dot = (prediction * target).sum(dim=1)
        prediction_norm = prediction.square().sum(dim=1).sqrt()
        target_norm = target.square().sum(dim=1).sqrt()
        prediction_nonzero = prediction_norm > 1e-12
        target_nonzero = target_norm > 1e-12
        valid = prediction_nonzero | target_nonzero
        if not torch.any(valid):
            return prediction.new_zeros(())
        both_nonzero = prediction_nonzero & target_nonzero
        angles = prediction.new_full(dot.shape, torch.pi / 2.0)
        cosine = (
            dot[both_nonzero]
            / (prediction_norm[both_nonzero] * target_norm[both_nonzero])
        ).clamp(-1.0, 1.0)
        angles[both_nonzero] = torch.acos(cosine)
        return torch.rad2deg(angles[valid]).mean()


class SSIM(nn.Module):
    """SSIM with an explicit dynamic range and no zero-padded border."""

    def __init__(self, data_range=1.0, window_size=11):
        super().__init__()
        if data_range <= 0:
            raise ValueError("SSIM data_range must be positive")
        self.data_range = float(data_range)
        self.window_size = int(window_size)

    def forward(self, prediction, target):
        return ssim(
            prediction,
            target,
            window_size=self.window_size,
            data_range=self.data_range,
            padding=0,
        )


class ERGAS(nn.Module):
    """ERGAS(prediction, target), normalized by target band means."""

    def __init__(self, r=1):
        super().__init__()
        self.r = r

    def forward(self, prediction, target):
        if prediction.shape != target.shape:
            raise ValueError("ERGAS prediction and target must have identical shapes")
        rmse_per_band = (prediction - target).square().mean(dim=(-2, -1)).sqrt()
        target_mean = target.mean(dim=(-2, -1)).abs().clamp_min(
            torch.finfo(target.dtype).eps
        )
        return (
            100
            * self.r
            * ((rmse_per_band / target_mean).square().mean(dim=1)).sqrt()
        ).mean()


def metric_suite():
    return {"psnr": PSNR(), "ssim": SSIM(), "sam": SAM(), "ergas": ERGAS()}


def charbonnier_loss(prediction, target):
    return 0.125 * torch.sqrt((prediction - target).square() + 1e-6).sum()
