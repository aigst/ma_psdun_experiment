#!/usr/bin/env bash
set -euo pipefail

# Keep OD0--OD3 as separate channels so the learned MultiODTCM can estimate
# capture reliability instead of averaging the repeated measurements first.
ROOT=/sci_persistent_storage/ma_psdun_v2_20260830
SRC="$ROOT/new_exp_src"
DATA="$ROOT/data/sample/sample"
PATTERNS="$ROOT/data/4096.tif"
OUT="$ROOT/new_results_20260902_multiod"
mkdir -p "$OUT"

run_one() {
  local gpu="$1" name="$2" resize="$3" stages="$4" prior="$5" consistency="$6" seed="$7"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    nohup python3 "$SRC/new_sample_train.py" \
      --data-root "$DATA" \
      --patterns "$PATTERNS" \
      --exp-dir "$OUT/$name" \
      --preprocess detrend \
      --detrend-width 31 \
      --fusion multi \
      --label-resample "$resize" \
      --condition-mode constant \
      --stages "$stages" \
      --gain-init 4.0 \
      --lowpass-kernel 3 \
      --prior-residual-scale "$prior" \
      --train-all-nontest \
      --epochs 500 \
      --lr 1e-4 \
      --weight-decay 1e-5 \
      --consistency-weight "$consistency" \
      --ssim-weight 0.2 \
      --seed "$seed" \
      --log-every 50 \
      --save-every 100 \
      --test-objects obj18-20260901,obj19-20260901,obj20-20260901 \
      --val-objects obj15-20260901,obj16-20260901,obj17-20260901 \
      > "$OUT/$name.log" 2>&1 &
  echo "$name pid=$! gpu=$gpu resize=$resize stages=$stages prior=$prior consistency=$consistency seed=$seed"
}

run_one 0 multi_bilinear_s12_c050 bilinear 12 0.50 0.05 20260904
run_one 1 multi_bilinear_s08_c050 bilinear 8 0.50 0.05 20260904
run_one 2 multi_bilinear_s12_c010 bilinear 12 0.50 0.01 20260905
run_one 3 multi_lanczos_s12_c050 lanczos 12 0.50 0.05 20260904
wait
