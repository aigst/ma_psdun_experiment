#!/usr/bin/env bash
set -euo pipefail

ROOT=/sci_persistent_storage/ma_psdun_v2_20260830
SRC="$ROOT/new_exp_src"
OUT="$ROOT/ring_blind_20260902/final"
PYTHON=/tmp/ma-psdun-cu121/bin/python
TRAIN=bar1-20260830,bar10-20260831,bar2-20260830,bar3-20260830,bar4-20260830,bar5-20260830,bar6-20260831,bar7-20260831,bar8-20260831,bar9-20260831,obj11-20260901,obj13-20260901,obj14-20260901,obj17-20260901
mkdir -p "$OUT"

run_one() {
  local gpu="$1" seed="$2"
  local name="seed${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON" "$SRC/ring_blind_train.py" \
      --mode final --exp-dir "$OUT/$name" \
      --train-objects "$TRAIN" --fusion mean --epochs 600 --seed "$seed" \
      --dihedral-augmentation > "$OUT/$name.log" 2>&1 &
  echo "$name pid=$! gpu=$gpu"
}

run_one 0 20260931
run_one 1 20260932
run_one 2 20260933
run_one 3 20260934
wait

CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PYTHON" "$SRC/ring_blind_train.py" \
    --mode ensemble --exp-dir "$OUT/ensemble" \
    --members "$OUT/seed20260931,$OUT/seed20260932,$OUT/seed20260933,$OUT/seed20260934"
