#!/usr/bin/env bash
set -euo pipefail

ROOT=/sci_persistent_storage/ma_psdun_v2_20260830
SRC="$ROOT/new_exp_src"
DATA="$ROOT/data/sample/sample"
PATTERNS="$ROOT/data/4096.tif"
OUT="$ROOT/new_results_20260902_adaptive_od"
mkdir -p "$OUT"

run_one() {
  local gpu="$1" name="$2" scale="$3" seed="$4"
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
      --adaptive-fusion-scale "$scale" \
      --train-all-nontest \
      --epochs 500 \
      --lr 1e-4 \
      --weight-decay 1e-5 \
      --consistency-weight 0.01 \
      --ssim-weight 0.2 \
      --tv-weight 0.0 \
      --edge-weight 0.05 \
      --seed "$seed" \
      --log-every 50 \
      --save-every 100 \
      --test-objects obj18-20260901,obj19-20260901,obj20-20260901 \
      --val-objects obj15-20260901,obj16-20260901,obj17-20260901 \
      > "$OUT/$name.log" 2>&1 &
  echo "$name pid=$! gpu=$gpu adaptive_scale=$scale seed=$seed"
}

run_one 0 adaptive025 0.25 20261020
run_one 1 adaptive050 0.50 20261021
run_one 2 adaptive100 1.00 20261022
run_one 3 adaptive200 2.00 20261023
wait
