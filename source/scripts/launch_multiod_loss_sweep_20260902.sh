#!/usr/bin/env bash
set -euo pipefail

# Hyperparameter sweep with the original object-level train/val/test split.
# Checkpoints are selected by validation SSIM; obj18--obj20 remain held out.
ROOT=/sci_persistent_storage/ma_psdun_v2_20260830
SRC="$ROOT/new_exp_src"
DATA="$ROOT/data/sample/sample"
PATTERNS="$ROOT/data/4096.tif"
OUT="$ROOT/new_results_20260902_multiod_loss"
mkdir -p "$OUT"

run_one() {
  local gpu="$1" name="$2" stages="$3" consistency="$4" ssim_weight="$5" binary_weight="$6" seed="$7"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    nohup python3 "$SRC/new_sample_train.py" \
      --data-root "$DATA" \
      --patterns "$PATTERNS" \
      --exp-dir "$OUT/$name" \
      --preprocess detrend \
      --detrend-width 31 \
      --fusion multi \
      --label-resample bilinear \
      --condition-mode constant \
      --stages "$stages" \
      --gain-init 4.0 \
      --lowpass-kernel 3 \
      --prior-residual-scale 0.50 \
      --epochs 700 \
      --lr 1e-4 \
      --weight-decay 1e-5 \
      --consistency-weight "$consistency" \
      --binary-weight "$binary_weight" \
      --ssim-weight "$ssim_weight" \
      --seed "$seed" \
      --log-every 50 \
      --save-every 100 \
      --test-objects obj18-20260901,obj19-20260901,obj20-20260901 \
      --val-objects obj15-20260901,obj16-20260901,obj17-20260901 \
      > "$OUT/$name.log" 2>&1 &
  echo "$name pid=$! gpu=$gpu stages=$stages consistency=$consistency ssim_weight=$ssim_weight binary_weight=$binary_weight seed=$seed"
}

run_one 0 val_multi_s12_c010_ssim050 12 0.01 0.50 0.0 20260906
run_one 1 val_multi_s12_c000_ssim100 12 0.00 1.00 0.0 20260907
run_one 2 val_multi_s12_c010_ssim100 12 0.01 1.00 0.0 20260908
run_one 3 val_multi_s08_c010_ssim050_bin010 8 0.01 0.50 0.10 20260909
wait
