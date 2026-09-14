#!/usr/bin/env bash
set -euo pipefail

ROOT=/sci_persistent_storage/ma_psdun_v2_20260830
SRC="$ROOT/new_exp_src"
DATA="$ROOT/data/sample/sample"
PATTERNS="$ROOT/data/4096.tif"
OUT="$ROOT/new_results_20260902_new_domain"
mkdir -p "$OUT"

run_one() {
  local gpu="$1" name="$2" seed="$3" extra="$4"
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
      $extra \
      > "$OUT/$name.log" 2>&1 &
  echo "$name pid=$! gpu=$gpu seed=$seed extra=$extra"
}

run_one 0 new_only6 20260970 '--train-all-nontest --train-objects obj11-20260901,obj13-20260901,obj14-20260901,obj15-20260901,obj16-20260901,obj17-20260901'
run_one 1 new_only3 20260971 '--train-objects obj11-20260901,obj13-20260901,obj14-20260901'
run_one 2 repeat_new3 20260972 '--train-all-nontest --repeat-new 3'
run_one 3 repeat_new5 20260973 '--train-all-nontest --repeat-new 5'
wait
