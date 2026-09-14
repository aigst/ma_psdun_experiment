import argparse
import json
import time
from pathlib import Path

import torch

from ma_psdun.core import MeasurementOperator, simulate_measurements, structured_images
from ma_psdun.eval import image_metrics
from ma_psdun.model import MAPSDUN, loss_fn
from ma_psdun.noise import load_noise_profile


def split_ids(n=1000, seed=123):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randperm(n, generator=g).tolist()
    a, b = int(n * 0.8), int(n * 0.9)
    return {'train': ids[:a], 'val': ids[a:b], 'test': ids[b:]}


def make_condition(rate, intensity, size):
    return torch.stack([
        torch.full_like(intensity, rate),
        intensity,
        torch.full_like(intensity, 550.0),
    ], dim=-1)


def evaluate_fixed(model, op, size, rate, intensity_value, seed, samples, device, photon_peak, noise_kwargs=None):
    target = structured_images(samples, size, seed=seed).to(device)
    intensity = torch.full((samples,), intensity_value, device=device)
    raw, dark = simulate_measurements(target, op, intensity, seed=seed, photon_peak=photon_peak, **(noise_kwargs or {}))
    cond = make_condition(rate, intensity, size)
    with torch.no_grad():
        pred, y = model(raw[:, None], dark[:, None], cond, (size, size))
    metrics = image_metrics(pred, target.reshape(samples, 1, size, size))
    metrics['measurement_l1'] = float((op.forward(pred.flatten(1)) - y).abs().mean())
    metrics['corr'] = float(torch.corrcoef(torch.stack([pred.flatten(), target.flatten()]))[0, 1])
    metrics['pred_std'] = float(pred.std())
    metrics['target_std'] = float(target.std())
    return metrics, pred, target


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp-dir', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--size', type=int, default=128)
    p.add_argument('--sampling-rate', type=float, default=0.2)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--stages', type=int, default=6)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--operator-seed', type=int, default=123)
    p.add_argument('--intensity', type=float, default=0.1)
    p.add_argument('--photon-peak', type=float, default=200.0)
    p.add_argument('--noise-profile', default='', help='JSON profile produced by tools/analyze_noise.py')
    p.add_argument('--noise-mode', choices=['poisson', 'empirical', 'hybrid'], default='poisson')
    p.add_argument('--noise-scale', type=float, default=0.005)
    p.add_argument('--dark-noise-scale', type=float, default=0.005)
    p.add_argument('--val-every', type=int, default=100)
    p.add_argument('--val-samples', type=int, default=64)
    p.add_argument('--test-samples', type=int, default=64)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--resume', default='')
    a = p.parse_args()
    torch.manual_seed(a.seed)
    torch.set_float32_matmul_precision('high')
    device = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    n = a.size * a.size
    m = max(1, int(n * a.sampling_rate))
    exp = Path(a.exp_dir)
    exp.mkdir(parents=True, exist_ok=True)
    if a.noise_mode != 'poisson' and not a.noise_profile:
        raise ValueError('--noise-profile is required for empirical or hybrid noise')
    noise_profile = load_noise_profile(a.noise_profile) if a.noise_profile else None
    noise_kwargs = {
        'noise_profile': noise_profile,
        'noise_mode': a.noise_mode,
        'noise_scale': a.noise_scale,
        'dark_noise_scale': a.dark_noise_scale,
    }
    (exp / 'config.json').write_text(json.dumps(vars(a), indent=2))

    op = MeasurementOperator(n, m, seed=a.operator_seed, device=device)
    model = MAPSDUN(op, stages=a.stages).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    start = 0
    if a.resume:
        ck = torch.load(a.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        start = ck['step'] + 1

    # All runs use identical validation/test targets so 8-GPU sweeps are
    # directly comparable; only the training stream and model seed differ.
    val_seed, test_seed = 424243, 424242
    val_target = structured_images(a.val_samples, a.size, seed=val_seed).to(device)
    val_intensity = torch.full((a.val_samples,), a.intensity, device=device)
    val_raw, val_dark = simulate_measurements(val_target, op, val_intensity, seed=val_seed, photon_peak=a.photon_peak, **noise_kwargs)
    val_cond = make_condition(a.sampling_rate, val_intensity, a.size)

    best_mse = float('inf')
    best_ssim = -float('inf')
    t0 = time.time()
    model.train()
    for step in range(start, a.steps):
        target = structured_images(a.batch_size, a.size, seed=a.seed + step).to(device)
        intensity = torch.full((a.batch_size,), a.intensity, device=device)
        raw, dark = simulate_measurements(target, op, intensity, seed=a.seed + step, photon_peak=a.photon_peak, **noise_kwargs)
        cond = make_condition(a.sampling_rate, intensity, a.size)
        pred, y = model(raw[:, None], dark[:, None], cond, (a.size, a.size))
        loss = loss_fn(pred, target.reshape(a.batch_size, 1, a.size, a.size), y, op)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 20 == 0 or step == a.steps - 1:
            torch.save({'step': step, 'model': model.state_dict(), 'optimizer': opt.state_dict()}, exp / 'checkpoint_latest.pt')
            rec = {'step': step, 'loss': float(loss.detach()), 'elapsed_s': time.time() - t0}
            with (exp / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(rec) + '\n')
            print(json.dumps(rec), flush=True)

        if step % a.val_every == 0 or step == a.steps - 1:
            model.eval()
            with torch.no_grad():
                val_pred, val_y = model(val_raw[:, None], val_dark[:, None], val_cond, (a.size, a.size))
                val_metrics = image_metrics(val_pred, val_target.reshape(a.val_samples, 1, a.size, a.size))
                val_metrics['step'] = step
                val_metrics['corr'] = float(torch.corrcoef(torch.stack([val_pred.flatten(), val_target.flatten()]))[0, 1])
            model.train()
            with (exp / 'validation.jsonl').open('a') as f:
                f.write(json.dumps(val_metrics) + '\n')
            if val_metrics['mse'] < best_mse:
                best_mse = val_metrics['mse']
                torch.save({'step': step, 'model': model.state_dict(), 'optimizer': opt.state_dict(), 'validation': val_metrics}, exp / 'checkpoint_best_mse.pt')
            if val_metrics['ssim'] > best_ssim:
                best_ssim = val_metrics['ssim']
                # The default checkpoint follows the structural acceptance
                # metric; the MSE-selected checkpoint remains available for
                # pixelwise comparisons.
                torch.save({'step': step, 'model': model.state_dict(), 'optimizer': opt.state_dict(), 'validation': val_metrics}, exp / 'checkpoint_best.pt')

    best_path = exp / 'checkpoint_best.pt'
    if best_path.exists():
        best = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best['model'])
    model.eval()
    test_metrics, pred, target = evaluate_fixed(model, op, a.size, a.sampling_rate, a.intensity, test_seed, a.test_samples, device, a.photon_peak, noise_kwargs)
    test_metrics['checkpoint_step'] = int(best.get('step', a.steps - 1) if best_path.exists() else a.steps - 1)
    test_metrics['checkpoint_metric'] = 'ssim'
    test_metrics['photon_peak'] = a.photon_peak
    (exp / 'eval.json').write_text(json.dumps(test_metrics, indent=2))
    torch.save({'pred': pred[:8].cpu(), 'target': target[:8].reshape(8, 1, a.size, a.size).cpu()}, exp / 'eval_samples.pt')
    (exp / 'DONE').write_text('completed\n')
    print(json.dumps(test_metrics), flush=True)


if __name__ == '__main__':
    main()
