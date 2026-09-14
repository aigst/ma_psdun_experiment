#!/usr/bin/env bash
set -euo pipefail

ROOT=/sci_persistent_storage/ma_psdun_v2_20260830
SRC="$ROOT/new_exp_src"
DATA="$ROOT/data/sample/sample"
PATTERNS="$ROOT/data/4096.tif"
OUT="$ROOT/new_results_20260902_loo"
mkdir -p "$OUT"

run_one() {
  local gpu="$1" name="$2" heldout="$3" seed="$4"
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
      --test-objects "$heldout" \
      --val-objects obj17-20260901 \
      > "$OUT/$name.log" 2>&1 &
  echo "$name pid=$! gpu=$gpu heldout=$heldout seed=$seed"
}

run_one 0 loo_obj18 obj18-20260901 20261010
run_one 1 loo_obj19 obj19-20260901 20261011
run_one 2 loo_obj20 obj20-20260901 20261012
run_one 3 loo_obj19_seed2 obj19-20260901 20261013
wait
