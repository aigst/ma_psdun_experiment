import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import ConditionEncoder


class TCM(nn.Module):
    """Normalize and denoise the positive raw/dark bucket sequences."""

    reference_scale = 0.35

    def __init__(self, channels=32, max_relative_residual=0.15):
        super().__init__()
        self.max_relative_residual = max_relative_residual
        self.net = nn.Sequential(
            nn.Conv1d(2, channels, 7, padding=3), nn.GELU(),
            nn.Conv1d(channels, channels, 7, padding=3), nn.GELU(),
            nn.Conv1d(channels, channels, 5, padding=2), nn.GELU(),
            nn.Conv1d(channels, 1, 3, padding=1), nn.Tanh(),
        )

    def forward(self, raw, dark):
        baseline = raw - dark
        scale = baseline.flatten(1).std(dim=1, keepdim=True).clamp_min(1e-4)
        raw_n = (raw - raw.mean(dim=-1, keepdim=True)) / scale[:, :, None]
        dark_n = (dark - dark.mean(dim=-1, keepdim=True)) / scale[:, :, None]
        base_n = (baseline - baseline.mean(dim=-1, keepdim=True)) / scale[:, :, None]
        delta = self.net(torch.cat([raw_n, dark_n], dim=1))
        y = base_n + self.max_relative_residual * delta * (base_n.abs() + 0.05)
        return y * self.reference_scale


class MultiODTCM(nn.Module):
    """Learned fusion of multiple OD captures after per-capture normalization."""

    reference_scale = 0.35

    def __init__(self, od_channels=4, channels=48, max_relative_residual=0.15,
                 adaptive_fusion_scale=0.0, preserve_od_amplitude=False):
        super().__init__()
        self.od_channels = od_channels
        self.max_relative_residual = max_relative_residual
        self.adaptive_fusion_scale = float(adaptive_fusion_scale)
        self.preserve_od_amplitude = bool(preserve_od_amplitude)
        self.net = nn.Sequential(
            nn.Conv1d(2 * od_channels, channels, 7, padding=3), nn.GELU(),
            nn.Conv1d(channels, channels, 7, padding=3), nn.GELU(),
            nn.Conv1d(channels, od_channels, 5, padding=2), nn.Tanh(),
        )
        # Start with equal OD weights; training can suppress low-SNR captures.
        self.od_logits = nn.Parameter(torch.zeros(od_channels))
        if self.adaptive_fusion_scale > 0.0:
            # A bounded, zero-initialized gate provides per-sample/per-time OD
            # weighting while preserving equal fusion at initialization.
            self.gate = nn.Conv1d(2 * od_channels, od_channels, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
        else:
            self.gate = None

    def normalize_captures(self, raw, dark):
        if raw.shape[1] != self.od_channels or dark.shape[1] != self.od_channels:
            raise ValueError(f"expected {self.od_channels} OD channels, got {raw.shape[1]} and {dark.shape[1]}")
        baseline = raw - dark
        centered = baseline - baseline.mean(dim=-1, keepdim=True)
        if self.preserve_od_amplitude:
            # Center captures independently to remove their DC/background, but
            # use one scale for the complete OD stack. This keeps the measured
            # attenuation ratios while remaining invariant to ADC units.
            scale = centered.flatten(1).std(dim=1, keepdim=True)[:, :, None].clamp_min(1e-4)
        else:
            scale = centered.flatten(2).std(dim=2, keepdim=True).clamp_min(1e-4)
        raw_n = (raw - raw.mean(dim=-1, keepdim=True)) / scale
        dark_n = (dark - dark.mean(dim=-1, keepdim=True)) / scale
        base_n = centered / scale
        return raw_n, dark_n, base_n

    def forward(self, raw, dark):
        raw_n, dark_n, base_n = self.normalize_captures(raw, dark)
        delta = self.net(torch.cat([raw_n, dark_n], dim=1))
        corrected = base_n + self.max_relative_residual * delta * (base_n.abs() + 0.05)
        logits = self.od_logits.view(1, self.od_channels, 1)
        if self.gate is not None:
            logits = logits + self.adaptive_fusion_scale * torch.tanh(self.gate(torch.cat([raw_n, dark_n], dim=1)))
        weights = torch.softmax(logits, dim=1)
        return (corrected * weights).sum(dim=1, keepdim=True) * self.reference_scale


class Prior(nn.Module):
    """Two-level conditional residual U-Net used as a denoising prior."""

    def __init__(self, cond_dim=32, residual_scale=0.05):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
        )
        self.mid = nn.Sequential(
            nn.Conv2d(64 + cond_dim, 96, 3, padding=1), nn.GELU(),
            nn.Conv2d(96, 96, 3, padding=1), nn.GELU(),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(96, 64, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(64 + 32, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, condition):
        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        c = condition[:, :, None, None].expand(-1, -1, h2.shape[-2], h2.shape[-1])
        h = self.mid(torch.cat([h2, c], dim=1))
        h = self.dec2(h)
        # Keep each unrolled prior close to identity early in training.  A
        # large residual here lets a few noisy samples drive all later stages
        # into the [0, 1] clamp and permanently erase structure.
        correction = self.residual_scale * torch.tanh(self.out(torch.cat([h, h1], dim=1)))
        return (x + correction).clamp(0.0, 1.0)


class MAPSDUN(nn.Module):
    def __init__(self, operator, stages=6, backprojection_gain_init=1.5,
                 lowpass_kernel=7, od_channels=1, prior_residual_scale=0.05,
                 shared_prior=False, adaptive_fusion_scale=0.0,
                 preserve_od_amplitude=False):
        super().__init__()
        self.operator = operator
        self.stages = stages
        if lowpass_kernel < 1 or lowpass_kernel % 2 == 0:
            raise ValueError("lowpass_kernel must be a positive odd integer")
        self.lowpass_kernel = lowpass_kernel
        self.od_channels = od_channels
        self.tcm = (TCM() if od_channels == 1 else
                    MultiODTCM(od_channels=od_channels,
                               adaptive_fusion_scale=adaptive_fusion_scale,
                               preserve_od_amplitude=preserve_od_amplitude))
        self.condition = ConditionEncoder()
        self.prior_residual_scale = float(prior_residual_scale)
        self.shared_prior = bool(shared_prior)
        # With very few labelled objects, independently parameterized priors
        # at every stage overfit the training shapes.  Sharing one prior
        # across stages keeps the unrolled physics while reducing capacity.
        prior_count = 1 if self.shared_prior else stages
        self.priors = nn.ModuleList([Prior(residual_scale=prior_residual_scale) for _ in range(prior_count)])
        # A sigmoid parameterization avoids the saturated clamp failure in v1.
        self.rho_logits = nn.Parameter(torch.full((stages,), -3.2))
        # For centered 0/1 masks, E[A^T A / M] is about 0.25 I.  Start with
        # its inverse gain, while allowing training to adapt to sampling
        # noise and the absolute label scale of a real acquisition.
        self.backprojection_gain = nn.Parameter(torch.tensor(float(backprojection_gain_init)))
        self.backprojection_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, raw, dark, cond, image_hw):
        y = self.tcm(raw, dark).squeeze(1)
        # The fixed TCM reference is calibrated at r=0.2.  For a centered
        # Bernoulli operator, measurement energy scales approximately as
        # 1/sqrt(r); adapt the calibrated signal to the requested sampling
        # rate, including the real full-rate (r=1) acquisition.
        rate = cond[..., 0].clamp_min(1e-4)
        y = y * torch.sqrt(0.2 / rate)[:, None]
        b = y.shape[0]
        x0 = self.operator.adjoint(y).reshape(b, 1, *image_hw)
        # A binary random-mask backprojection contains high-frequency
        # cross-talk.  The optical target is spatially smooth, so apply a
        # fixed low-pass at initialization; this removes speckle artifacts
        # without per-image min/max amplification and leaves later priors to
        # restore sharper structure.
        gain = self.backprojection_gain.clamp(0.5, 8.0)
        bias = self.backprojection_bias.clamp(-0.25, 0.5)
        x = (bias + gain * x0).clamp(0.0, 1.0)
        if self.lowpass_kernel > 1:
            pad = self.lowpass_kernel // 2
            x = F.avg_pool2d(x, self.lowpass_kernel, 1, pad)
        c = self.condition(cond)
        rho = 0.5 * torch.sigmoid(self.rho_logits)
        for k in range(self.stages):
            prior = self.priors[0] if self.shared_prior else self.priors[k]
            residual = self.operator.forward(x.flatten(1)) - y
            update = self.operator.adjoint(residual).reshape_as(x)
            x = (x - rho[k] * update).clamp(0.0, 1.0)
            x = prior(x, c)
        return x, y


def _ssim_loss(pred, target):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_x = F.avg_pool2d(pred, 7, 1, 3)
    mu_y = F.avg_pool2d(target, 7, 1, 3)
    var_x = F.avg_pool2d(pred * pred, 7, 1, 3) - mu_x * mu_x
    var_y = F.avg_pool2d(target * target, 7, 1, 3) - mu_y * mu_y
    cov = F.avg_pool2d(pred * target, 7, 1, 3) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
    )
    return 1.0 - ssim.mean()


def loss_fn(pred, target, y_corr, operator):
    image_l1 = (pred - target).abs().mean()
    image_mse = (pred - target).square().mean()
    consistency = (operator.forward(pred.flatten(1)) - y_corr).abs().mean()
    tv = (pred[:, :, 1:] - pred[:, :, :-1]).abs().mean() + (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean()
    return image_l1 + 0.5 * image_mse + 0.2 * _ssim_loss(pred, target) + 0.05 * consistency + 0.01 * tv
