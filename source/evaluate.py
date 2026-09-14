import argparse, json
from pathlib import Path
import torch
from ma_psdun.core import MeasurementOperator, simulate_measurements, structured_images
from ma_psdun.model import MAPSDUN
from ma_psdun.eval import image_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp-dir', required=True); p.add_argument('--device', default='cuda:0')
    p.add_argument('--size', type=int, default=128); p.add_argument('--sampling-rate', type=float, required=True)
    p.add_argument('--stages', type=int, default=1); p.add_argument('--seed', type=int, default=999)
    p.add_argument('--operator-seed', type=int, default=123)
    p.add_argument('--intensity', type=float, default=0.0)
    p.add_argument('--samples', type=int, default=32)
    a = p.parse_args(); torch.manual_seed(a.seed)
    device = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    n, m = a.size*a.size, max(1, int(a.size*a.size*a.sampling_rate))
    op = MeasurementOperator(n, m, seed=a.operator_seed, device=device)
    model = MAPSDUN(op, stages=a.stages).to(device)
    ck = torch.load(Path(a.exp_dir)/'checkpoint_latest.pt', map_location=device)
    model.load_state_dict(ck['model']); model.eval()
    with torch.no_grad():
        target = structured_images(a.samples, a.size, seed=a.seed).to(device)
        if a.intensity > 0:
            intensity = torch.full((a.samples,), a.intensity, device=device)
        else:
            intensity = 10 ** torch.linspace(-3, 0, a.samples, device=device)
        raw, dark = simulate_measurements(target, op, intensity, seed=a.seed)
        cond = torch.stack([torch.full_like(intensity, a.sampling_rate), intensity, torch.full_like(intensity, 550.)], -1)
        pred, y = model(raw[:, None], dark[:, None], cond, (a.size, a.size))
        metrics = image_metrics(pred, target.reshape(a.samples, 1, a.size, a.size))
        metrics['measurement_l1'] = float((op.forward(pred.flatten(1)) - y).abs().mean())
        metrics['checkpoint_step'] = ck['step']; metrics['samples'] = a.samples
        Path(a.exp_dir, 'eval.json').write_text(json.dumps(metrics, indent=2))
        torch.save({'pred': pred[:4].cpu(), 'target': target[:4].reshape(4, 1, a.size, a.size).cpu()}, Path(a.exp_dir, 'eval_samples.pt'))
        print(json.dumps(metrics), flush=True)

if __name__ == '__main__': main()
