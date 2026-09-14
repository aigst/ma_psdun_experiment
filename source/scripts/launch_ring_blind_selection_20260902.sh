#!/usr/bin/env bash
set -euo pipefail

ROOT=/sci_persistent_storage/ma_psdun_v2_20260830
SRC="$ROOT/new_exp_src"
OUT="$ROOT/ring_blind_20260902/selection"
PYTHON=/tmp/ma-psdun-cu121/bin/python
TRAIN=bar1-20260830,bar10-20260831,bar2-20260830,bar3-20260830,bar4-20260830,bar5-20260830,bar7-20260831,bar8-20260831,bar9-20260831,obj11-20260901,obj14-20260901
VALIDATION=bar6-20260831,obj13-20260901,obj17-20260901
mkdir -p "$OUT"

run_one() {
  local gpu="$1" name="$2" augmentation="$3" edge="$4" dice="$5" pyramid="$6"
  local augmentation_flag=()
  if [[ "$augmentation" == "yes" ]]; then
    augmentation_flag=(--dihedral-augmentation)
  fi
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON" "$SRC/ring_blind_train.py" \
      --mode select --exp-dir "$OUT/$name" \
      --train-objects "$TRAIN" --val-objects "$VALIDATION" \
      --fusion mean --epochs 600 --seed 20260930 \
      --edge-weight "$edge" --dice-weight "$dice" --pyramid-weight "$pyramid" \
      "${augmentation_flag[@]}" > "$OUT/$name.log" 2>&1 &
  echo "$name pid=$! gpu=$gpu augmentation=$augmentation edge=$edge dice=$dice pyramid=$pyramid"
}

run_one 0 baseline no 0.00 0.00 0.00
run_one 1 dihedral yes 0.00 0.00 0.00
run_one 2 dihedral_dice yes 0.05 0.10 0.00
run_one 3 dihedral_pyramid yes 0.05 0.10 0.10
wait
