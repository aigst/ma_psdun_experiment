#!/usr/bin/env bash
set -euo pipefail

ROOT=${MA_PSDUN_ROOT:-/sci_persistent_storage/ma_psdun_v2_20260903}
SRC=${MA_PSDUN_SRC:-$ROOT/source/all25_fullfit}
DATA=${MA_PSDUN_DATA:-$ROOT/data/sample}
PATTERNS=${MA_PSDUN_PATTERNS:-$ROOT/data/4096.tif}
OUT=${MA_PSDUN_ALL25_STRICT_FINAL_OUT:-$ROOT/results_all25_strict_final_cvmean_20260904}
PYTHON=${MA_PSDUN_PYTHON:-python3}

TRAIN=bar1-20260830,bar2-20260830,bar3-20260830,bar4-20260830,bar5-20260830,bar6-20260831,bar7-20260831,bar8-20260831,bar9-20260831,bar10-20260831,obj11-20260901,obj12-20260901,obj13-20260901,obj14-20260901,obj17-20260901,obj21-20260902,obj22-20260902,obj23-20260902,obj24-20260903,obj25-20260903
TEST=obj18-20260901,obj19-20260901,obj20-20260901
AUDIT=obj15-20260901,obj16-20260901
FORBIDDEN=obj15-20260901,obj16-20260901,obj19-20260901

common=(
  --data-root "$DATA" --patterns "$PATTERNS" --test-objects "$TEST"
  --audit-objects "$AUDIT" --forbidden-supervised-objects "$FORBIDDEN"
  --label-resample bilinear --psf-sigma 1.0 --stages 12 --gain-init 4.0
  --lowpass-kernel 3 --prior-residual-scale 0.5 --epochs 600 --lr 1e-4
  --weight-decay 1e-5 --consistency-weight 0.05 --ssim-weight 0.2
  --tv-weight 0.0 --edge-weight 0.0 --log-every 100
)

run_one() {
  local gpu=$1 seed=$2
  local name=seed${seed}
  mkdir -p "$OUT/$name"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON" "$SRC/ring_blind_train.py" --mode final \
      --exp-dir "$OUT/$name" --train-objects "$TRAIN" --seed "$seed" \
      --preprocess detrend --detrend-width 127 --fusion mean \
      --dihedral-augmentation "${common[@]}" \
      > "$OUT/$name.log" 2>&1 &
  echo "$name pid=$! gpu=$gpu"
}

run_one 0 20261101
run_one 1 20261102
run_one 2 20261103
run_one 3 20261104
wait

CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PYTHON" "$SRC/ring_blind_train.py" --mode ensemble \
    --exp-dir "$OUT/ensemble" \
    --members "$OUT/seed20261101,$OUT/seed20261102,$OUT/seed20261103,$OUT/seed20261104"
echo completed > "$OUT/FINAL_DONE"
