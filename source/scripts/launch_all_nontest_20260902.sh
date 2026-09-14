#!/usr/bin/env bash
set -euo pipefail

# Final-fit sweep: all labelled objects except obj18--obj20 are used for
# fitting.  The three test objects remain completely held out.
ROOT=/sci_persistent_storage/ma_psdun_v2_20260830
SRC="$ROOT/new_exp_src"
DATA="$ROOT/data/sample/sample"
PATTERNS="$ROOT/data/4096.tif"
OUT="$ROOT/new_results_20260902_all_nontest"
mkdir -p "$OUT"

run_one() {
  local gpu="$1" name="$2" resize="$3" stages="$4" prior="$5" consistency="$6" weight_decay="$7" seed="$8"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    nohup python3 "$SRC/new_sample_train.py" \
      --data-root "$DATA" \
      --patterns "$PATTERNS" \
      --exp-dir "$OUT/$name" \
      --preprocess detrend \
      --detrend-width 31 \
      --fusion mean \
      --label-resample "$resize" \
      --condition-mode constant \
      --stages "$stages" \
      --gain-init 4.0 \
      --lowpass-kernel 3 \
      --prior-residual-scale "$prior" \
      --train-all-nontest \
      --epochs 500 \
      --lr 1e-4 \
      --weight-decay "$weight_decay" \
      --consistency-weight "$consistency" \
      --ssim-weight 0.2 \
      --seed "$seed" \
      --log-every 50 \
      --save-every 100 \
      --test-objects obj18-20260901,obj19-20260901,obj20-20260901 \
      --val-objects obj15-20260901,obj16-20260901,obj17-20260901 \
      > "$OUT/$name.log" 2>&1 &
  echo "$name pid=$! gpu=$gpu resize=$resize stages=$stages prior=$prior consistency=$consistency weight_decay=$weight_decay seed=$seed"
}

run_one 0 allnt_lanczos_s12_c050_wd1e5 lanczos 12 0.50 0.05 1e-5 20260901
run_one 1 allnt_bilinear_s12_c050_wd1e5 bilinear 12 0.50 0.05 1e-5 20260901
run_one 2 allnt_bilinear_s12_c010_wd1e5 bilinear 12 0.50 0.01 1e-5 20260902
run_one 3 allnt_bilinear_s08_c020_wd1e4 bilinear 8 0.50 0.02 1e-4 20260903
wait
