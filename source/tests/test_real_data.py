from pathlib import Path
import numpy as np
from PIL import Image

from ma_psdun.real_data import load_real_dataset


def test_real_loader_pairs_dark_then_bright_and_loads_pattern(tmp_path):
    root = tmp_path / 'data' / 'bar1-20260830' / 'OD0'
    root.mkdir(parents=True)
    (tmp_path / 'data' / 'bar1-20260830' / 'SI').mkdir()
    np.savetxt(root / 'traindata.txt', np.arange(8, dtype=np.float32))
    Image.new('L', (4, 4), 128).save(tmp_path / 'data' / 'bar1-20260830' / 'SI' / 'SI_AP.png')
    pat = tmp_path / 'patterns.tif'
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(pat, save_all=True, append_images=[Image.fromarray(np.ones((4, 4), dtype=np.uint8)) for _ in range(3)])
    patterns, samples = load_real_dataset(tmp_path / 'data', pat)
    assert patterns.shape == (4, 16)
    assert set(np.unique(patterns).tolist()) == {0.0, 1.0}
    assert len(samples) == 1
    assert np.allclose(samples[0]['y_raw'], np.arange(1, 8, 2))
    assert np.allclose(samples[0]['y_dark'], np.arange(0, 8, 2))
    assert abs(float(samples[0]['y'].mean())) < 1e-6
    assert np.isfinite(samples[0]['y']).all()
