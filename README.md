# MA-PSDUN experiment code bundle

This bundle contains the source code used for the new-dataset MA-PSDUN reconstruction and calibrated sim-to-real experiments. It includes the model package, data preparation, synthetic simulator, training/evaluation entry points, launcher scripts, and tests.

## Contents

- `source/ma_psdun/`: measurement operator, unrolled model, OD attenuation, detector noise, labels, and evaluation helpers.
- `source/new_dataset_domain_bridge_train.py`: train-object-calibrated simulator -> synthetic pretraining -> real fine-tuning experiment used for the paper-guided repeat.
- `source/domain_bridge_train.py`: synthetic dataset and simulator utilities shared by the domain-bridge entry point.
- `source/new_sample_train.py`: new sample dataset loader, OD fusion, preprocessing, loss, and training helpers.
- `source/tools/prepare_new_dataset.py`: lossless conversion/audit of `DAQrawdata` and `imagedata` into the trainer layout.
- `source/ring_blind_train.py`: object-disjoint selection/final training helpers and split audits.
- `source/tests/`: unit and contract tests, including the split, attenuation, noise, and new-data tests.
- `runners/run_new_dataset_20260913.sh`: multi-GPU new-dataset selection/final-deployment runner. Credentials are intentionally removed; set URLs through environment variables.

The bundle intentionally excludes raw datasets, checkpoints, prediction tensors, generated images, `__pycache__`, pytest caches, and result directories. Those are experiment artifacts rather than source code.

## Environment

Recommended: Python 3.10+, PyTorch 2.x, NumPy, Pillow, and pytest. Install the pinned minimum dependencies with:

```bash
python3 -m pip install -r source/requirements.txt
```

Run the focused regression suite:

```bash
PYTHONPATH="$PWD/source" pytest -q \
  source/tests/test_domain_bridge_split.py \
  source/tests/test_attenuation_simulator.py \
  source/tests/test_noise.py \
  source/tests/test_new_data_split.py
```

## Paper-guided calibrated sim-to-real run

The strict protocol used in the final repeat is represented by:

```bash
PYTHONPATH="$PWD/source" python3 source/new_dataset_domain_bridge_train.py \
  --data-root /path/to/prepared_dataset \
  --patterns /path/to/prepared_dataset/pattern.tif \
  --out-dir /path/to/results/domain_bridge_paper_guided \
  --device cuda:0 \
  --seeds 2026091101,2026091102 \
  --threads 16 \
  --synthetic-size 256 \
  --pretrain-steps 120 \
  --scratch-epochs 80 \
  --finetune-epochs 80 \
  --finetune-lr 3e-5 \
  --synthetic-mix 0.25 \
  --noise-mode hybrid \
  --skip-hparam-sweep \
  --log-every 10
```

The script enforces the object-level Train/Validation/Locked-test/Unlabelled split, fits nuisance/attenuation/noise from Train only, writes `SELECTION_LOCKED_BEFORE_TEST` before reading locked-test labels, and emits machine-readable audits and checkpoints.

## 2026-09-13 runner

The runner uses `DATA_URL` and `SOURCE_URL` environment variables. Example:

```bash
export DATA_URL='http://<internal-host>/<dataset>.zip'
export SOURCE_URL='http://<internal-host>/ma_psdun_source.tar.gz'
export MA_PSDUN_ROOT=/path/to/run-root
bash runners/run_new_dataset_20260913.sh
```

Do not put internal credentials into this bundle or commit them to shell history. Supply authentication using the approved environment or internal download tooling.

## Provenance

- Original source tree: `/root/new_exp_src`
- New-data experiment: `EID-20260909-ma-psdun-new-dataset`
- Paper-guided result root: `/root/ma_psdun_new_20260909/domain_bridge_paper_guided_20260911`
- Bundle creation date: 2026-09-14
