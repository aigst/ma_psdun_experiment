#!/usr/bin/env bash
set -euo pipefail

ROOT=${MA_PSDUN_ROOT:-/sci_persistent_storage/ma_psdun_v2_20260903}
SRC=${MA_PSDUN_SRC:-$ROOT/source}
DATA=${MA_PSDUN_DATA:-$ROOT/data/sample}
PATTERNS=${MA_PSDUN_PATTERNS:-$ROOT/data/4096.tif}
OUT=${MA_PSDUN_WIDTH_OUT:-$ROOT/results_detrend_width_cv_20260904}
WIDTH=${MA_PSDUN_DETREND_WIDTH:-127}
PYTHON=${MA_PSDUN_PYTHON:-python3}

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

mkdir -p "$OUT/selection"
common=(
  --data-root "$DATA" --patterns "$PATTERNS"
  --test-objects "$TEST" --audit-objects "$AUDIT"
  --forbidden-supervised-objects "$FORBIDDEN"
  --label-resample bilinear --condition-mode constant
  --preprocess detrend --detrend-width "$WIDTH" --fusion mean
  --stages 12 --gain-init 4.0 --lowpass-kernel 3 --prior-residual-scale 0.5
  --epochs 600 --lr 1e-4 --weight-decay 1e-5 --consistency-weight 0.05
  --ssim-weight 0.2 --tv-weight 0.0 --edge-weight 0.0
  --binary-weight 0.0 --dihedral-augmentation --log-every 100
)

run_select() {
  local gpu=$1 fold=$2
  local tr_var=TRAIN_$fold val_var=VAL_$fold
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON" "$SRC/ring_blind_train.py" --mode select \
      --exp-dir "$OUT/selection/fold$fold" \
      --train-objects "${!tr_var}" --val-objects "${!val_var}" \
      --seed "$((20261400 + fold))" "${common[@]}" \
      > "$OUT/selection/fold$fold.log" 2>&1 &
}

for fold in 1 2 3 4; do
  run_select $((fold - 1)) "$fold"
done
wait

"$PYTHON" - "$OUT" "$WIDTH" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
width = int(sys.argv[2])
rows = []
for fold in range(1, 5):
    d = root / "selection" / f"fold{fold}"
    result = json.loads((d / "selection.json").read_text())
    split = json.loads((d / "split_audit.json").read_text())
    rows.append({"fold": fold, "ssim": float(result["best_validation_ssim"]), "objects": split["validation"]})
total = sum(len(row["objects"]) for row in rows)
summary = {
    "status": "selection_completed",
    "preprocess": "detrend",
    "detrend_width": width,
    "selection_ranking": [{
        "name": f"mean_detrend_w{width}",
        "cross_validation_ssim": sum(row["ssim"] * len(row["objects"]) for row in rows) / total,
        "folds": rows,
    }],
    "object_level_disjoint": True,
    "test_used_for_selection": False,
}
(root / "selection" / "ranking.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
(root / "SELECTION_DONE").write_text("completed\n")
print(json.dumps(summary, ensure_ascii=False))
PY
