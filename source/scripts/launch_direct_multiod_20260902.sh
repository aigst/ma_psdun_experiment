#!/usr/bin/env bash
set -euo pipefail

ROOT=/sci_persistent_storage/ma_psdun_v2_20260830
SRC="$ROOT/new_exp_src"
DATA="$ROOT/data/sample/sample"
PATTERNS="$ROOT/data/4096.tif"
OUT="$ROOT/new_results_20260902_direct_multiod"
mkdir -p "$OUT"

run_one() {
  local gpu="$1" name="$2" base="$3" lr="$4" edge="$5" ssim="$6" seed="$7"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    nohup python3 "$SRC/direct_multiod_train.py" \
      --data-root "$DATA" \
      --patterns "$PATTERNS" \
      --exp-dir "$OUT/$name" \
      --label-resample bilinear \
      --detrend-width 31 \
      --base "$base" \
      --gain 4.0 \
      --lowpass 3 \
      --epochs 1000 \
      --lr "$lr" \
      --weight-decay 1e-5 \
      --edge-weight "$edge" \
      --ssim-weight "$ssim" \
      --seed "$seed" \
      --test-objects obj18-20260901,obj19-20260901,obj20-20260901 \
      --val-objects obj15-20260901,obj16-20260901,obj17-20260901 \
      > "$OUT/$name.log" 2>&1 &
  echo "$name pid=$! gpu=$gpu base=$base lr=$lr edge=$edge ssim=$ssim seed=$seed"
}

run_one 0 direct_multi_b024_lr3e4_e010_s020 24 3e-4 0.10 0.20 20260910
run_one 1 direct_multi_b032_lr3e4_e010_s020 32 3e-4 0.10 0.20 20260911
run_one 2 direct_multi_b024_lr3e4_e000_s050 24 3e-4 0.00 0.50 20260912
run_one 3 direct_multi_b016_lr5e4_e020_s020 16 5e-4 0.20 0.20 20260913
wait
