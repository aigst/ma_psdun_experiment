import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .noise import sample_noise_sequences


class MeasurementOperator:
    """Random binary SPI operator with an optional symmetric optical PSF."""
    def __init__(self, n, m, seed=0, device="cpu", patterns=None,
                 psf_sigma=0.0, image_hw=None, psf_sigma_y=None, psf_angle=0.0):
        if patterns is None:
            g = torch.Generator(device="cpu").manual_seed(seed)
            patterns = torch.randint(0, 2, (m, n), generator=g, dtype=torch.float32)
        patterns = patterns.float()
        if tuple(patterns.shape) != (m, n):
            raise ValueError(f"patterns must have shape {(m, n)}, got {tuple(patterns.shape)}")
        self.patterns = patterns.to(device)
        # Center each pixel over the measured random-pattern ensemble.  This
        # removes the DC component without assuming a Hadamard basis.
        self.centered_patterns = self.patterns - self.patterns.mean(dim=0, keepdim=True)
        self.scale = 1.0 / max(m, 1) ** 0.5
        self.psf_sigma = float(psf_sigma)
        self.psf_sigma_y = float(psf_sigma if psf_sigma_y is None else psf_sigma_y)
        self.psf_angle = float(psf_angle)
        if self.psf_sigma < 0.0:
            raise ValueError("psf_sigma must be non-negative")
        if self.psf_sigma_y < 0.0:
            raise ValueError("psf_sigma_y must be non-negative")
        if (self.psf_sigma > 0.0) != (self.psf_sigma_y > 0.0):
            raise ValueError("psf_sigma and psf_sigma_y must both be zero or both be positive")
        if self.psf_sigma > 0.0:
            if image_hw is None:
                side = math.isqrt(n)
                if side * side != n:
                    raise ValueError("image_hw is required for non-square PSF inputs")
                image_hw = (side, side)
            if len(image_hw) != 2 or int(image_hw[0]) * int(image_hw[1]) != n:
                raise ValueError(f"image_hw must contain {n} pixels, got {image_hw}")
            self.image_hw = (int(image_hw[0]), int(image_hw[1]))
            radius = max(1, int(math.ceil(4.0 * max(self.psf_sigma, self.psf_sigma_y))))
            coords_y = torch.arange(-radius, radius + 1, dtype=torch.float32)
            coords_x = torch.arange(-radius, radius + 1, dtype=torch.float32)
            yy, xx = torch.meshgrid(coords_y, coords_x, indexing="ij")
            angle = math.radians(self.psf_angle)
            rotated_x = xx * math.cos(angle) + yy * math.sin(angle)
            rotated_y = -xx * math.sin(angle) + yy * math.cos(angle)
            kernel = torch.exp(-0.5 * (
                rotated_x.square() / self.psf_sigma ** 2
                + rotated_y.square() / self.psf_sigma_y ** 2
            ))
            kernel = (kernel / kernel.sum()).reshape(1, 1, *kernel.shape)
            self.psf_kernel = kernel.to(self.patterns.device)
        else:
            self.image_hw = None
            self.psf_kernel = None

    def _blur(self, x):
        if self.psf_kernel is None:
            return x
        if x.shape[-1] != self.image_hw[0] * self.image_hw[1]:
            raise ValueError(f"expected flattened images with {self.image_hw[0] * self.image_hw[1]} pixels")
        height, width = self.image_hw
        image = x.reshape(-1, 1, height, width)
        radius_y = self.psf_kernel.shape[-2] // 2
        radius_x = self.psf_kernel.shape[-1] // 2
        # Symmetric zero-boundary convolution is self-adjoint for this
        # symmetric kernel, so reusing it in adjoint() gives H^T exactly.
        return F.conv2d(image, self.psf_kernel,
                        padding=(radius_y, radius_x)).reshape_as(x)

    def forward(self, x):
        """Centered measurement used by the reconstruction objective."""
        return self._blur(x) @ self.centered_patterns.T * self.scale

    def forward_raw(self, x):
        """Positive detector bucket signal before DC removal."""
        return self._blur(x) @ self.patterns.T * self.scale

    @staticmethod
    def center_measurement(y):
        return y - y.mean(dim=-1, keepdim=True)

    def adjoint(self, y):
        # Match the 1/sqrt(M) scale used by forward().
        return self._blur(y @ self.centered_patterns * self.scale)


class ConditionEncoder(nn.Module):
    """Encode r, s, wavelength as log-scaled continuous features."""
    def __init__(self, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, out_dim))

    def forward(self, cond):
        r, s, wavelength = cond.unbind(-1)
        z = torch.stack([
            torch.logit(r.clamp(1e-4, 1 - 1e-4)),
            torch.log10(s.clamp_min(1e-6)),
            (wavelength - 1050.0) / 650.0,
        ], dim=-1)
        return self.net(z)


class TCM(nn.Module):
    """Learn only a bounded correction around the observable dark-subtracted signal."""
    def __init__(self, channels=16, max_relative_residual=0.25):
        super().__init__()
        self.max_relative_residual = max_relative_residual
        self.net = nn.Sequential(
            nn.Conv1d(2, channels, 5, padding=2), nn.SiLU(),
            nn.Conv1d(channels, channels, 5, padding=2), nn.SiLU(),
            nn.Conv1d(channels, 1, 3, padding=1), nn.Tanh(),
        )

    def forward(self, raw, dark):
        baseline = raw - dark
        delta = self.net(torch.cat([raw, dark], dim=1))
        return baseline + self.max_relative_residual * delta * (baseline.abs() + 1e-3)


def structured_images(batch, size, seed=0):
    """Create smooth, sparse geometric-like synthetic SPI targets."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
    out = torch.zeros(batch, size, size)
    for i in range(batch):
        img = torch.zeros(size, size)
        for _ in range(3):
            cx, cy = torch.empty(2).uniform_(-.7, .7, generator=g)
            sx, sy = torch.empty(2).uniform_(.08, .35, generator=g)
            amp = torch.empty(1).uniform_(.4, 1.0, generator=g).item()
            img += amp * torch.exp(-((xx-cx)**2/(2*sx*sx) + (yy-cy)**2/(2*sy*sy)))
        if i % 2 == 0:
            img += .35 * ((xx > -.35) & (xx < .35)).float()
        out[i] = img.clamp(0, 1)
    return out.reshape(batch, -1)


def simulate_measurements(
    x,
    op,
    intensity,
    seed=0,
    photon_peak=200.0,
    *,
    noise_profile=None,
    noise_mode="poisson",
    noise_scale=0.005,
    dark_noise_scale=0.005,
    od_labels=None,
):
    """Generate detector buckets with either synthetic or empirical noise.

    ``noise_mode='empirical'`` disables the independent Poisson/readout draws
    and replays normalized dark residual sequences from ``noise_profile``.
    ``hybrid`` retains photon noise and adds the empirical correlated term.
    The scale parameters map normalized real ADC residuals to the simulator's
    normalized image units and must be calibrated from a bright/dark pair.
    """
    if noise_mode not in {"poisson", "empirical", "hybrid"}:
        raise ValueError(f"unknown noise_mode: {noise_mode}")
    if noise_mode in {"empirical", "hybrid"} and noise_profile is None:
        raise ValueError("noise_profile is required for empirical or hybrid noise")
    g = torch.Generator(device=x.device).manual_seed(seed)
    ideal = op.forward_raw(x)
    m = ideal.shape[-1]
    t = torch.linspace(0.0, 1.0, m, device=x.device)[None, :]
    alpha = torch.empty(x.shape[0], device=x.device).uniform_(0.2, 0.5, generator=g)
    gain = intensity.pow(alpha).unsqueeze(-1)
    rise = torch.empty(x.shape[0], device=x.device).uniform_(0.4, 2.0, generator=g).unsqueeze(-1)
    fall = torch.empty(x.shape[0], device=x.device).uniform_(0.3, 1.5, generator=g).unsqueeze(-1)
    relaxation = 1.0 + 0.12 * (1.0 - torch.exp(-t * rise)) * torch.exp(-t * fall)
    drift = torch.empty(x.shape[0], device=x.device).uniform_(0.02, 0.15, generator=g).unsqueeze(-1)
    dark_level = drift * ideal.detach().amax(dim=-1, keepdim=True)
    dark = dark_level * (0.9 + 0.2 * t)
    signal = ideal * gain * relaxation + dark
    if noise_mode in {"poisson", "hybrid"}:
        # The detector sees non-negative photon rates.  Sign is not discarded:
        # the reconstruction uses the centered 0/1 measurement ensemble.
        photon = torch.poisson(signal.clamp_min(0) * photon_peak, generator=g) / photon_peak
        readout = torch.randn(signal.shape, generator=g, device=x.device) * noise_scale
        raw = photon + readout
        dark_obs = dark + torch.randn(dark.shape, generator=g, device=x.device) * dark_noise_scale
    else:
        raw, dark_obs = signal, dark
    if noise_mode in {"empirical", "hybrid"}:
        correlated = sample_noise_sequences(
            noise_profile, x.shape[0], m, generator=g, device=x.device,
            dtype=signal.dtype, od_labels=od_labels,
        )
        dark_correlated = sample_noise_sequences(
            noise_profile, x.shape[0], m, generator=g, device=x.device,
            dtype=signal.dtype, od_labels=od_labels,
        )
        raw = raw + noise_scale * correlated
        dark_obs = dark_obs + dark_noise_scale * dark_correlated
    return raw, dark_obs
