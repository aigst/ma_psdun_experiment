#!/usr/bin/env bash
set -euo pipefail

out_root="${1:-experiments/final2-syn}"
steps="${STEPS:-1500}"
mkdir -p "$out_root"

# Eight independent jobs are used so each configuration has its own model and
# optimizer state. CUDA_VISIBLE_DEVICES remaps the selected card to cuda:0.
declare -a gpus=(0 1 2 3 4 5 6 7)
declare -a names=(r05s1 r10s1 r20s1 r30s1 r50s1 r20s01 r20s001 r50s01)
declare -a rates=(0.05 0.10 0.20 0.30 0.50 0.20 0.20 0.50)
declare -a intensities=(0.1 0.1 0.1 0.1 0.1 0.01 0.001 0.01)

pids=()
for i in "${!gpus[@]}"; do
  gpu="${gpus[$i]}"
  name="${names[$i]}"
  rate="${rates[$i]}"
  intensity="${intensities[$i]}"
  exp_dir="${out_root}-${name}"
  mkdir -p "$exp_dir"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python3 train.py \
      --exp-dir "$exp_dir" \
      --size 128 \
      --sampling-rate "$rate" \
      --intensity "$intensity" \
      --stages 6 \
      --steps "$steps" \
      --batch-size 16 \
      --seed 123 \
      --operator-seed 123 \
      --photon-peak 200 \
      --device cuda:0 \
      >"$exp_dir/train.log" 2>&1
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
exit "$status"
