#!/usr/bin/env bash
set -euo pipefail

ROOT=/sci_persistent_storage/ma_psdun_v2_20260830
SRC="$ROOT/new_exp_src"
DATA="$ROOT/data/sample/sample"
PATTERNS="$ROOT/data/4096.tif"
OUT="$ROOT/new_results_20260902_singleod_allnt"
mkdir -p "$OUT"

run_one() {
  local gpu="$1" name="$2" stages="$3" condition="$4" edge="$5" seed="$6"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    nohup python3 "$SRC/new_sample_train.py" \
      --data-root "$DATA" --patterns "$PATTERNS" --exp-dir "$OUT/$name" \
      --preprocess detrend --detrend-width 31 --fusion none --label-resample bilinear \
      --condition-mode "$condition" --stages "$stages" --gain-init 4.0 --lowpass-kernel 3 \
      --prior-residual-scale 0.50 --train-all-nontest --epochs 500 --lr 1e-4 \
      --weight-decay 1e-5 --consistency-weight 0.01 --ssim-weight 0.2 --tv-weight 0.0 \
      --edge-weight "$edge" --seed "$seed" --log-every 50 --save-every 100 \
      --test-objects obj18-20260901,obj19-20260901,obj20-20260901 \
      --val-objects obj15-20260901,obj16-20260901,obj17-20260901 \
      > "$OUT/$name.log" 2>&1 &
  echo "$name pid=$! gpu=$gpu stages=$stages condition=$condition edge=$edge seed=$seed"
}

run_one 0 singleod_s12_od_e005 12 od 0.05 20261000
run_one 1 singleod_s12_const_e005 12 constant 0.05 20261001
run_one 2 singleod_s08_od_e005 8 od 0.05 20261002
run_one 3 singleod_s12_od_e000 12 od 0.00 20261003
wait
