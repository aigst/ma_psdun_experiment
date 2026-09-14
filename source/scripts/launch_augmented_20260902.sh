#!/usr/bin/env bash
set -euo pipefail

ROOT=/sci_persistent_storage/ma_psdun_v2_20260830
SRC="$ROOT/new_exp_src"
DATA="$ROOT/data/sample/sample"
PATTERNS="$ROOT/data/4096.tif"
OUT="$ROOT/new_results_20260902_augmented"
mkdir -p "$OUT"

run_one() {
  local gpu="$1" name="$2" pseudo_weight="$3" noise_std="$4" stages="$5" prior="$6"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    nohup python3 "$SRC/augmented_train.py" \
      --data-root "$DATA" \
      --patterns "$PATTERNS" \
      --exp-dir "$OUT/$name" \
      --preprocess detrend \
      --fusion mean \
      --label-resample bilinear \
      --stages "$stages" \
      --gain-init 4.0 \
      --lowpass-kernel 3 \
      --prior-residual-scale "$prior" \
      --epochs 500 \
      --lr 1e-4 \
      --weight-decay 1e-5 \
      --consistency-weight 0.05 \
      --ssim-weight 0.2 \
      --pseudo-weight "$pseudo_weight" \
      --pseudo-noise-std "$noise_std" \
      --seed 20260902 \
      --test-objects obj18-20260901,obj19-20260901,obj20-20260901 \
      --val-objects obj15-20260901,obj16-20260901,obj17-20260901 \
      > "$OUT/$name.log" 2>&1 &
  echo "$name pid=$! gpu=$gpu pseudo_weight=$pseudo_weight noise_std=$noise_std stages=$stages prior=$prior"
}

run_one 0 aug_w025_n005_s12 0.25 0.05 12 0.50
run_one 1 aug_w025_n010_s12 0.25 0.10 12 0.50
run_one 2 aug_w050_n010_s12 0.50 0.10 12 0.50
run_one 3 aug_w025_n010_s08 0.25 0.10 8 0.50
wait
