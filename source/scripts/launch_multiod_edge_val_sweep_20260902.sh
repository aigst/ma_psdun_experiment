#!/usr/bin/env bash
set -euo pipefail

ROOT=/sci_persistent_storage/ma_psdun_v2_20260830
SRC="$ROOT/new_exp_src"
DATA="$ROOT/data/sample/sample"
PATTERNS="$ROOT/data/4096.tif"
OUT="$ROOT/new_results_20260902_multiod_edge_val"
mkdir -p "$OUT"

run_one() {
  local gpu="$1" name="$2" tv="$3" edge="$4" seed="$5"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    nohup python3 "$SRC/new_sample_train.py" \
      --data-root "$DATA" --patterns "$PATTERNS" --exp-dir "$OUT/$name" \
      --preprocess detrend --detrend-width 31 --fusion multi --label-resample bilinear \
      --condition-mode constant --stages 12 --gain-init 4.0 --lowpass-kernel 3 \
      --prior-residual-scale 0.50 --epochs 700 --lr 1e-4 --weight-decay 1e-5 \
      --consistency-weight 0.01 --ssim-weight 0.2 --tv-weight "$tv" --edge-weight "$edge" \
      --seed "$seed" --log-every 50 --save-every 100 \
      --test-objects obj18-20260901,obj19-20260901,obj20-20260901 \
      --val-objects obj15-20260901,obj16-20260901,obj17-20260901 \
      > "$OUT/$name.log" 2>&1 &
  echo "$name pid=$! gpu=$gpu tv=$tv edge=$edge seed=$seed"
}

run_one 0 val_edge0_tv0 0.000 0.000 20260980
run_one 1 val_edge005_tv0 0.000 0.050 20260981
run_one 2 val_edge010_tv0 0.000 0.100 20260982
run_one 3 val_edge005_tv010 0.010 0.050 20260983
wait
