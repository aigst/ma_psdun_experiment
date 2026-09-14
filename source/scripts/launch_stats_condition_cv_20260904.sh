#!/usr/bin/env bash
set -euo pipefail

ROOT=${MA_PSDUN_ROOT:-/sci_persistent_storage/ma_psdun_v2_20260903}
SRC=${MA_PSDUN_SRC:-$ROOT/source}
DATA=${MA_PSDUN_DATA:-$ROOT/data/sample}
PATTERNS=${MA_PSDUN_PATTERNS:-$ROOT/data/4096.tif}
OUT=${MA_PSDUN_STATS_OUT:-$ROOT/results_stats_condition_cv_20260904}
PYTHON=${MA_PSDUN_PYTHON:-python3}
CONDITION_MODE=${MA_PSDUN_STATS_CONDITION_MODE:-stats}
BINARY_WEIGHT=${MA_PSDUN_BINARY_WEIGHT:-0.0}

TEST=obj18-20260901,obj19-20260901,obj20-20260901
AUDIT=obj15-20260901,obj16-20260901
FORBIDDEN=obj15-20260901,obj16-20260901,obj19-20260901
VAL_1=bar1-20260830,bar5-20260830,obj11-20260901,obj24-20260903
VAL_2=bar2-20260830,bar6-20260831,obj13-20260901,obj25-20260903
VAL_3=bar3-20260830,bar7-20260831,bar9-20260831,obj14-20260901
VAL_4=bar4-20260830,bar8-20260831,bar10-20260831,obj17-20260901,obj22-20260902
TRAIN_1=bar10-20260831,bar2-20260830,bar3-20260830,bar4-20260830,bar6-20260831,bar7-20260831,bar8-20260831,bar9-20260831,obj13-20260901,obj14-20260901,obj17-20260901,obj22-20260902,obj25-20260903
TRAIN_2=bar1-20260830,bar10-20260831,bar3-20260830,bar4-20260830,bar5-20260830,bar7-20260831,bar8-20260831,bar9-20260831,obj11-20260901,obj14-20260901,obj17-20260901,obj22-20260902,obj24-20260903
TRAIN_3=bar1-20260830,bar10-20260831,bar2-20260830,bar4-20260830,bar5-20260830,bar6-20260831,bar8-20260831,obj11-20260901,obj13-20260901,obj17-20260901,obj22-20260902,obj24-20260903,obj25-20260903
TRAIN_4=bar1-20260830,bar2-20260830,bar3-20260830,bar5-20260830,bar6-20260831,bar7-20260831,bar9-20260831,obj11-20260901,obj13-20260901,obj14-20260901,obj24-20260903,obj25-20260903
ALL=bar1-20260830,bar10-20260831,bar2-20260830,bar3-20260830,bar4-20260830,bar5-20260830,bar6-20260831,bar7-20260831,bar8-20260831,bar9-20260831,obj11-20260901,obj13-20260901,obj14-20260901,obj17-20260901,obj22-20260902,obj24-20260903,obj25-20260903

mkdir -p "$OUT/selection" "$OUT/final"
common=(--data-root "$DATA" --patterns "$PATTERNS"
  --test-objects "$TEST" --audit-objects "$AUDIT" --forbidden-supervised-objects "$FORBIDDEN"
  --label-resample bilinear --condition-mode "$CONDITION_MODE" --preprocess detrend --detrend-width 31 --fusion mean
  --stages 12 --gain-init 4.0 --lowpass-kernel 3 --prior-residual-scale 0.5
  --epochs 600 --lr 1e-4 --weight-decay 1e-5 --consistency-weight 0.05
  --ssim-weight 0.2 --tv-weight 0.0 --edge-weight 0.0 --binary-weight "$BINARY_WEIGHT" --dihedral-augmentation --log-every 100)

run_select() {
  local gpu=$1 fold=$2
  local tr_var=TRAIN_$fold val_var=VAL_$fold
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON" "$SRC/ring_blind_train.py" --mode select \
      --exp-dir "$OUT/selection/fold$fold/stats" \
      --train-objects "${!tr_var}" --val-objects "${!val_var}" \
      --seed "$((20261240 + fold))" "${common[@]}" \
      > "$OUT/selection/fold$fold-stats.log" 2>&1 &
}

for fold in 1 2 3 4; do
  run_select $((fold - 1)) "$fold"
done
wait

python3 - "$OUT/selection" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for fold in range(1, 5):
    d = root / f"fold{fold}" / "stats"
    result = json.loads((d / "selection.json").read_text())
    split = json.loads((d / "split_audit.json").read_text())
    rows.append({"fold": fold, "ssim": float(result["best_validation_ssim"]), "objects": split["validation"]})
total = sum(len(row["objects"]) for row in rows)
ranking = [{"name": "stats", "cross_validation_ssim": sum(row["ssim"] * len(row["objects"]) for row in rows) / total, "folds": rows}]
(root / "ranking.json").write_text(json.dumps(ranking, indent=2) + "\n")
print(json.dumps(ranking, indent=2))
PY
echo completed > "$OUT/SELECTION_DONE"

run_final() {
  local gpu=$1 seed=$2
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON" "$SRC/ring_blind_train.py" --mode final \
      --exp-dir "$OUT/final/seed$seed" --train-objects "$ALL" --seed "$seed" \
      "${common[@]}" > "$OUT/final/seed$seed.log" 2>&1 &
}
run_final 0 20261251
run_final 1 20261252
run_final 2 20261253
run_final 3 20261254
wait

members=$OUT/final/seed20261251,$OUT/final/seed20261252,$OUT/final/seed20261253,$OUT/final/seed20261254
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PYTHON" "$SRC/ring_blind_train.py" --mode ensemble \
  --exp-dir "$OUT/final/ensemble" --members "$members"

selection_dirs=$OUT/selection/fold1/stats,$OUT/selection/fold2/stats,$OUT/selection/fold3/stats,$OUT/selection/fold4/stats
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PYTHON" "$SRC/ring_blind_calibrate.py" \
  --selection-dirs "$selection_dirs" \
  --ensemble-predictions "$OUT/final/ensemble/predictions.pt" \
  --exp-dir "$OUT/final/calibrated" > "$OUT/final/calibration.log" 2>&1

python3 - "$OUT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
ranking = json.loads((root / "selection" / "ranking.json").read_text())
calibration = json.loads((root / "final" / "calibrated" / "calibration.json").read_text())
summary = {
    "status": "completed",
    "selected_configuration": "stats",
    "selection_ranking": ranking,
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
