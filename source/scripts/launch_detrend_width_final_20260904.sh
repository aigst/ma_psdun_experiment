#!/usr/bin/env bash
set -euo pipefail

ROOT=${MA_PSDUN_ROOT:-/sci_persistent_storage/ma_psdun_v2_20260903}
SRC=${MA_PSDUN_SRC:-$ROOT/source}
DATA=${MA_PSDUN_DATA:-$ROOT/data/sample}
PATTERNS=${MA_PSDUN_PATTERNS:-$ROOT/data/4096.tif}
OUT=${MA_PSDUN_WIDTH_FINAL_OUT:-$ROOT/results_detrend_width_final_20260904}
PYTHON=${MA_PSDUN_PYTHON:-python3}

TEST=obj18-20260901,obj19-20260901,obj20-20260901
AUDIT=obj15-20260901,obj16-20260901
FORBIDDEN=obj15-20260901,obj16-20260901,obj19-20260901
ALL=bar1-20260830,bar10-20260831,bar2-20260830,bar3-20260830,bar4-20260830,bar5-20260830,bar6-20260831,bar7-20260831,bar8-20260831,bar9-20260831,obj11-20260901,obj13-20260901,obj14-20260901,obj17-20260901,obj22-20260902,obj24-20260903,obj25-20260903

mkdir -p "$OUT/final"
common=(
  --data-root "$DATA" --patterns "$PATTERNS"
  --test-objects "$TEST" --audit-objects "$AUDIT"
  --forbidden-supervised-objects "$FORBIDDEN"
  --label-resample bilinear --condition-mode constant
  --preprocess detrend --detrend-width 127 --fusion mean
  --stages 12 --gain-init 4.0 --lowpass-kernel 3 --prior-residual-scale 0.5
  --epochs 600 --lr 1e-4 --weight-decay 1e-5 --consistency-weight 0.05
  --ssim-weight 0.2 --tv-weight 0.0 --edge-weight 0.0
  --binary-weight 0.0 --dihedral-augmentation --log-every 100
)

run_final() {
  local gpu=$1 seed=$2
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON" "$SRC/ring_blind_train.py" \
      --mode final --exp-dir "$OUT/final/seed$seed" \
      --train-objects "$ALL" --seed "$seed" "${common[@]}" \
      > "$OUT/final/seed$seed.log" 2>&1 &
}

run_final 0 20261471
run_final 1 20261472
run_final 2 20261473
run_final 3 20261474
wait

members=$OUT/final/seed20261471,$OUT/final/seed20261472,$OUT/final/seed20261473,$OUT/final/seed20261474
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PYTHON" "$SRC/ring_blind_train.py" \
    --mode ensemble --exp-dir "$OUT/final/ensemble" --members "$members"

selection_dirs=$ROOT/results_detrend_width_cv_20260904/selection/fold1,$ROOT/results_detrend_width_cv_20260904/selection/fold2,$ROOT/results_detrend_width_cv_20260904/selection/fold3,$ROOT/results_detrend_width_cv_20260904/selection/fold4
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PYTHON" "$SRC/ring_blind_calibrate.py" \
    --selection-dirs "$selection_dirs" \
    --ensemble-predictions "$OUT/final/ensemble/predictions.pt" \
    --exp-dir "$OUT/final/calibrated" > "$OUT/final/calibration.log" 2>&1

"$PYTHON" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
calibration = json.loads((root / "final" / "calibrated" / "calibration.json").read_text())
summary = {
    "status": "completed",
    "configuration": {
        "preprocess": "detrend",
        "detrend_width": 127,
        "fusion": "mean",
        "stages": 12,
        "epochs": 600,
        "member_seeds": [20261471, 20261472, 20261473, 20261474],
    },
    "primary_test": calibration["primary_test"],
    "audit_ring_holdout": calibration["audit_ring_holdout"],
    "calibration": calibration["selected"],
    "selection_objects": calibration["selection_objects"],
    "object_level_disjoint": True,
    "test_used_for_selection": False,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
(root / "PIPELINE_DONE").write_text("completed\n")
print(json.dumps(summary, ensure_ascii=False))
PY
