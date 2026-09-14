#!/usr/bin/env bash
set -euo pipefail

ROOT=/sci_persistent_storage/ma_psdun_v2_20260830
SRC="$ROOT/new_exp_src"
DATA="$ROOT/data/sample/sample"
PATTERNS="$ROOT/data/4096.tif"
OUT="$ROOT/new_results_20260902_full_fit"
mkdir -p "$OUT"

run_one() {
  local gpu="$1" name="$2" seed="$3"
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
      --stages 12 \
      --gain-init 4.0 \
      --lowpass-kernel 3 \
      --prior-residual-scale 0.50 \
      --fit-all \
      --epochs 1000 \
      --lr 1e-4 \
      --weight-decay 1e-5 \
      --consistency-weight 0.01 \
      --ssim-weight 0.2 \
      --tv-weight 0.0 \
      --edge-weight 0.05 \
      --seed "$seed" \
      --log-every 100 \
      --save-every 200 \
      > "$OUT/$name.log" 2>&1 &
  echo "$name pid=$! gpu=$gpu seed=$seed"
}

run_one 0 full_fit_seed0 20261100
run_one 1 full_fit_seed1 20261101
run_one 2 full_fit_seed2 20261102
run_one 3 full_fit_seed3 20261103
wait
