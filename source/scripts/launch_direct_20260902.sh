#!/usr/bin/env bash
set -euo pipefail

ROOT=/sci_persistent_storage/ma_psdun_v2_20260830
SRC="$ROOT/new_exp_src"
DATA="$ROOT/data/sample/sample"
PATTERNS="$ROOT/data/4096.tif"
OUT="$ROOT/new_results_20260902_direct"
mkdir -p "$OUT"

run_one() {
  local gpu="$1" name="$2" base="$3" edge="$4" lr="$5"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    nohup python3 "$SRC/direct_train.py" \
      --data-root "$DATA" \
      --patterns "$PATTERNS" \
      --exp-dir "$OUT/$name" \
      --label-resample bilinear \
      --detrend-width 31 \
      --base "$base" \
      --gain 4.0 \
      --lowpass 3 \
      --epochs 800 \
      --lr "$lr" \
      --weight-decay 1e-5 \
      --edge-weight "$edge" \
      --seed 20260902 \
      > "$OUT/$name.log" 2>&1 &
  echo "$name pid=$! gpu=$gpu base=$base edge=$edge lr=$lr"
}

run_one 0 direct_b16_e000 16 0.0 5e-4
run_one 1 direct_b16_e010 16 0.1 5e-4
run_one 2 direct_b16_e020 16 0.2 5e-4
run_one 3 direct_b08_e010 8 0.1 1e-3
wait
