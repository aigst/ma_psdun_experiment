#!/usr/bin/env bash
set -euo pipefail

ROOT=/sci_persistent_storage/ma_psdun_v2_20260830
SRC="$ROOT/new_exp_src"
DATA="$ROOT/data/sample/sample"
PATTERNS="$ROOT/data/4096.tif"
OUT="$ROOT/new_results_20260902_multiod_augmented"
mkdir -p "$OUT"

run_one() {
  local gpu="$1" name="$2" pseudo="$3" noise="$4" stages="$5" seed="$6"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    nohup python3 "$SRC/multiod_augmented_train.py" \
      --data-root "$DATA" --patterns "$PATTERNS" --exp-dir "$OUT/$name" \
      --stages "$stages" --prior-residual-scale 0.50 \
      --epochs 700 --lr 1e-4 --weight-decay 1e-5 \
      --pseudo-weight "$pseudo" --pseudo-noise "$noise" \
      --consistency-weight 0.01 --ssim-weight 0.2 --edge-weight 0.05 --tv-weight 0.0 \
      --seed "$seed" \
      --test-objects obj18-20260901,obj19-20260901,obj20-20260901 \
      --val-objects obj15-20260901,obj16-20260901,obj17-20260901 \
      > "$OUT/$name.log" 2>&1 &
  echo "$name pid=$! gpu=$gpu pseudo=$pseudo noise=$noise stages=$stages seed=$seed"
}

run_one 0 aug_multi_pw010_n000_s12 0.01 0.00 12 20260960
run_one 1 aug_multi_pw020_n010_s12 0.02 0.01 12 20260961
run_one 2 aug_multi_pw050_n010_s12 0.05 0.01 12 20260962
run_one 3 aug_multi_pw100_n005_s08 0.10 0.005 8 20260963
wait
