#!/usr/bin/env bash
set -euo pipefail

ROOT=/sci_persistent_storage/ma_psdun_v2_20260830
SRC="$ROOT/new_exp_src"
DATA="$ROOT/data/sample/sample"
PATTERNS="$ROOT/data/4096.tif"
OUT="$ROOT/new_results_20260902_detrend_width"
mkdir -p "$OUT"

run_one() {
  local gpu="$1" name="$2" mode="$3" width="$4" sigma="$5"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    nohup python3 "$SRC/new_sample_train.py" \
      --data-root "$DATA" \
      --patterns "$PATTERNS" \
      --exp-dir "$OUT/$name" \
      --preprocess "$mode" \
      --detrend-width "$width" \
      --gauss-sigma "$sigma" \
      --fusion mean \
      --label-resample bilinear \
      --condition-mode constant \
      --stages 12 \
      --gain-init 4.0 \
      --lowpass-kernel 3 \
      --prior-residual-scale 0.50 \
      --epochs 500 \
      --lr 1e-4 \
      --weight-decay 1e-5 \
      --consistency-weight 0.05 \
      --ssim-weight 0.2 \
      --seed 20260901 \
      --log-every 50 \
      --save-every 100 \
      --test-objects obj18-20260901,obj19-20260901,obj20-20260901 \
      --val-objects obj15-20260901,obj16-20260901,obj17-20260901 \
      > "$OUT/$name.log" 2>&1 &
  echo "$name pid=$! gpu=$gpu mode=$mode width=$width sigma=$sigma"
}

run_one 0 detrend_w031 detrend 31 40
run_one 1 detrend_w063 detrend 63 40
run_one 2 detrend_w255 detrend 255 40
run_one 3 detrend_robust_w127 detrend_robust 127 40
wait
