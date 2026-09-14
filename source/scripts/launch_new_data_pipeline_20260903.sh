#!/usr/bin/env bash
set -euo pipefail

ROOT=${MA_PSDUN_ROOT:-/sci_persistent_storage/ma_psdun_v2_20260903}
SRC=${MA_PSDUN_SRC:-$ROOT/source}
DATA=${MA_PSDUN_DATA:-$ROOT/data/sample}
PATTERNS=${MA_PSDUN_PATTERNS:-$ROOT/data/4096.tif}
OUT=${MA_PSDUN_OUT:-$ROOT/results}
PYTHON=${MA_PSDUN_PYTHON:-python3}

TEST=obj18-20260901,obj19-20260901,obj20-20260901
AUDIT=obj15-20260901,obj16-20260901
FORBIDDEN=obj15-20260901,obj16-20260901,obj19-20260901
SELECT_TRAIN=bar1-20260830,bar10-20260831,bar2-20260830,bar3-20260830,bar4-20260830,bar5-20260830,bar7-20260831,bar8-20260831,bar9-20260831,obj11-20260901,obj14-20260901,obj22-20260902,obj24-20260903,obj25-20260903
VALIDATION=bar6-20260831,obj13-20260901,obj17-20260901
FINAL_TRAIN=$SELECT_TRAIN,$VALIDATION

mkdir -p "$OUT/selection" "$OUT/final"

common_args=(
  --data-root "$DATA"
  --patterns "$PATTERNS"
  --test-objects "$TEST"
  --audit-objects "$AUDIT"
  --forbidden-supervised-objects "$FORBIDDEN"
  --label-resample bilinear
  --condition-mode constant
  --stages 12
  --gain-init 4.0
  --lowpass-kernel 3
  --prior-residual-scale 0.5
  --epochs 600
  --lr 1e-4
  --weight-decay 1e-5
  --consistency-weight 0.05
  --ssim-weight 0.2
  --tv-weight 0.0
  --dihedral-augmentation
  --log-every 50
)

run_selection() {
  local gpu=$1 name=$2 preprocess=$3 fusion=$4 adaptive=$5 edge=$6 dice=$7 pyramid=$8
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON" "$SRC/ring_blind_train.py" \
      --mode select \
      --exp-dir "$OUT/selection/$name" \
      --train-objects "$SELECT_TRAIN" \
      --val-objects "$VALIDATION" \
      --preprocess "$preprocess" \
      --detrend-width 31 \
      --fusion "$fusion" \
      --adaptive-fusion-scale "$adaptive" \
      --edge-weight "$edge" \
      --dice-weight "$dice" \
      --pyramid-weight "$pyramid" \
      --seed 20260930 \
      "${common_args[@]}" > "$OUT/selection/$name.log" 2>&1 &
  echo "selection $name pid=$! gpu=$gpu"
}

run_selection 0 mean_detrend detrend mean 0.0 0.00 0.00 0.00
run_selection 1 mean_robust detrend_robust mean 0.0 0.00 0.00 0.00
run_selection 2 multi_adaptive detrend multi 0.25 0.00 0.00 0.00
run_selection 3 mean_edge_pyramid detrend mean 0.0 0.05 0.10 0.10
wait

BEST_NAME=$(
  "$PYTHON" - "$OUT/selection" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/selection.json")):
    result = json.loads(path.read_text())
    rows.append((float(result["best_validation_ssim"]), path.parent.name))
if len(rows) != 4:
    raise SystemExit(f"expected four completed selection runs, found {len(rows)}")
rows.sort(reverse=True)
(root / "ranking.json").write_text(json.dumps([
    {"name": name, "best_validation_ssim": score} for score, name in rows
], indent=2) + "\n")
print(rows[0][1])
PY
)
echo "selected configuration: $BEST_NAME"

case "$BEST_NAME" in
  mean_detrend)
    selected_args=(--preprocess detrend --detrend-width 31 --fusion mean)
    ;;
  mean_robust)
    selected_args=(--preprocess detrend_robust --detrend-width 31 --fusion mean)
    ;;
  multi_adaptive)
    selected_args=(--preprocess detrend --detrend-width 31 --fusion multi --adaptive-fusion-scale 0.25)
    ;;
  mean_edge_pyramid)
    selected_args=(--preprocess detrend --detrend-width 31 --fusion mean --edge-weight 0.05 --dice-weight 0.10 --pyramid-weight 0.10)
    ;;
  *)
    echo "unknown selected configuration: $BEST_NAME" >&2
    exit 1
    ;;
esac

run_final() {
  local gpu=$1 seed=$2
  local name=seed$seed
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON" "$SRC/ring_blind_train.py" \
      --mode final \
      --exp-dir "$OUT/final/$name" \
      --train-objects "$FINAL_TRAIN" \
      --seed "$seed" \
      "${selected_args[@]}" \
      "${common_args[@]}" > "$OUT/final/$name.log" 2>&1 &
  echo "final $name pid=$! gpu=$gpu"
}

run_final 0 20260941
run_final 1 20260942
run_final 2 20260943
run_final 3 20260944
wait

members=$OUT/final/seed20260941,$OUT/final/seed20260942,$OUT/final/seed20260943,$OUT/final/seed20260944
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PYTHON" "$SRC/ring_blind_train.py" \
    --mode ensemble \
    --exp-dir "$OUT/final/ensemble" \
    --members "$members"

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PYTHON" "$SRC/ring_blind_calibrate.py" \
    --selection-dir "$OUT/selection/$BEST_NAME" \
    --ensemble-predictions "$OUT/final/ensemble/predictions.pt" \
    --exp-dir "$OUT/final/calibrated" > "$OUT/final/calibration.log" 2>&1

"$PYTHON" - "$OUT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
ranking = json.loads((root / "selection" / "ranking.json").read_text())
calibration = json.loads((root / "final" / "calibrated" / "calibration.json").read_text())
checkpoints = sorted(root.glob("final/seed*/checkpoint_final.pt"))

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

summary = {
    "status": "completed",
    "selection_ranking": ranking,
    "selected_configuration": ranking[0]["name"],
    "primary_test": calibration["primary_test"],
    "audit_ring_holdout": calibration["audit_ring_holdout"],
    "calibration": calibration["selected"],
    "checkpoint_sha256": {path.parent.name: sha256(path) for path in checkpoints},
    "object_level_disjoint": True,
    "test_used_for_selection": False,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
(root / "PIPELINE_DONE").write_text("completed\n")
print(json.dumps(summary, ensure_ascii=False))
PY

