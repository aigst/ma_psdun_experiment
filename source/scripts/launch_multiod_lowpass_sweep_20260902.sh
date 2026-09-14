#!/usr/bin/env bash
set -euo pipefail

ROOT=/sci_persistent_storage/ma_psdun_v2_20260830
SRC="$ROOT/new_exp_src"
DATA="$ROOT/data/sample/sample"
PATTERNS="$ROOT/data/4096.tif"
OUT="$ROOT/new_results_20260902_multiod_lowpass"
mkdir -p "$OUT"

run_one() {
  local gpu="$1" name="$2" lowpass="$3" prior="$4" edge="$5" seed="$6"
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
      --lowpass-kernel "$lowpass" \
      --prior-residual-scale "$prior" \
      --train-all-nontest \
      --epochs 500 \
      --lr 1e-4 \
      --weight-decay 1e-5 \
      --consistency-weight 0.01 \
      --ssim-weight 0.2 \
      --tv-weight 0.0 \
      --edge-weight "$edge" \
      --seed "$seed" \
      --log-every 50 \
      --save-every 100 \
      --test-objects obj18-20260901,obj19-20260901,obj20-20260901 \
      --val-objects obj15-20260901,obj16-20260901,obj17-20260901 \
      > "$OUT/$name.log" 2>&1 &
  echo "$name pid=$! gpu=$gpu lowpass=$lowpass prior=$prior edge=$edge seed=$seed"
}

run_one 0 allnt_multi_lp1_p050_e005 1 0.50 0.05 20260930
run_one 1 allnt_multi_lp3_p050_e005 3 0.50 0.05 20260931
run_one 2 allnt_multi_lp5_p050_e005 5 0.50 0.05 20260932
run_one 3 allnt_multi_lp1_p025_e010 1 0.25 0.10 20260933
wait
