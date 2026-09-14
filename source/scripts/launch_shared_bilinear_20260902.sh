#!/usr/bin/env bash
set -euo pipefail

ROOT=/sci_persistent_storage/ma_psdun_v2_20260830
SRC="$ROOT/new_exp_src"
DATA="$ROOT/data/sample/sample"
PATTERNS="$ROOT/data/4096.tif"
OUT="$ROOT/new_results_20260902_shared_bilinear"
mkdir -p "$OUT"

run_one() {
  local gpu="$1" name="$2" stages="$3" prior="$4"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    nohup python3 "$SRC/new_sample_train.py" \
      --data-root "$DATA" \
      --patterns "$PATTERNS" \
      --exp-dir "$OUT/$name" \
      --preprocess detrend \
      --fusion mean \
      --label-resample bilinear \
      --condition-mode constant \
      --stages "$stages" \
      --shared-prior \
      --gain-init 4.0 \
      --lowpass-kernel 3 \
      --prior-residual-scale "$prior" \
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
  echo "$name pid=$! gpu=$gpu stages=$stages prior=$prior"
}

run_one 0 shared_s12_p050 12 0.50
run_one 1 shared_s08_p050 8 0.50
run_one 2 shared_s06_p050 6 0.50
run_one 3 shared_s12_p030 12 0.30
wait
