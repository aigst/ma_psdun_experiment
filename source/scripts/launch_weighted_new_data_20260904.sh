#!/usr/bin/env bash
set -euo pipefail

ROOT=${MA_PSDUN_ROOT:-/sci_persistent_storage/ma_psdun_v2_20260903}
SRC=${MA_PSDUN_SRC:-$ROOT/source}
DATA=${MA_PSDUN_DATA:-$ROOT/data/sample}
PATTERNS=${MA_PSDUN_PATTERNS:-$ROOT/data/4096.tif}
OUT=${MA_PSDUN_WEIGHTED_OUT:-$ROOT/results_weighted_new_20260904}
PYTHON=${MA_PSDUN_PYTHON:-python3}

TEST=obj18-20260901,obj19-20260901,obj20-20260901
AUDIT=obj15-20260901,obj16-20260901
FORBIDDEN=obj15-20260901,obj16-20260901,obj19-20260901
NEW=obj22-20260902,obj24-20260903,obj25-20260903

VAL_1=bar1-20260830,bar5-20260830,obj11-20260901,obj24-20260903
VAL_2=bar2-20260830,bar6-20260831,obj13-20260901,obj25-20260903
VAL_3=bar3-20260830,bar7-20260831,bar9-20260831,obj14-20260901
VAL_4=bar4-20260830,bar8-20260831,bar10-20260831,obj17-20260901,obj22-20260902
TRAIN_1=bar10-20260831,bar2-20260830,bar3-20260830,bar4-20260830,bar6-20260831,bar7-20260831,bar8-20260831,bar9-20260831,obj13-20260901,obj14-20260901,obj17-20260901,obj22-20260902,obj25-20260903
TRAIN_2=bar1-20260830,bar10-20260831,bar3-20260830,bar4-20260830,bar5-20260830,bar7-20260831,bar8-20260831,bar9-20260831,obj11-20260901,obj14-20260901,obj17-20260901,obj22-20260902,obj24-20260903
TRAIN_3=bar1-20260830,bar10-20260831,bar2-20260830,bar4-20260830,bar5-20260830,bar6-20260831,bar8-20260831,obj11-20260901,obj13-20260901,obj17-20260901,obj22-20260902,obj24-20260903,obj25-20260903
TRAIN_4=bar1-20260830,bar2-20260830,bar3-20260830,bar5-20260830,bar6-20260831,bar7-20260831,bar9-20260831,obj11-20260901,obj13-20260901,obj14-20260901,obj24-20260903,obj25-20260903
ALL=bar1-20260830,bar10-20260831,bar2-20260830,bar3-20260830,bar4-20260830,bar5-20260830,bar6-20260831,bar7-20260831,bar8-20260831,bar9-20260831,obj11-20260901,obj13-20260901,obj14-20260901,obj17-20260901,obj22-20260902,obj24-20260903,obj25-20260903

mkdir -p "$OUT/selection"
common=(
  --data-root "$DATA" --patterns "$PATTERNS"
  --test-objects "$TEST" --audit-objects "$AUDIT"
  --forbidden-supervised-objects "$FORBIDDEN" --new-objects "$NEW"
  --label-resample bilinear --condition-mode constant
  --preprocess detrend --detrend-width 31 --fusion mean
  --stages 12 --gain-init 4.0 --lowpass-kernel 3 --prior-residual-scale 0.5
  --epochs 600 --lr 1e-4 --weight-decay 1e-5 --consistency-weight 0.05
  --ssim-weight 0.2 --tv-weight 0.0 --edge-weight 0.0
  --dice-weight 0.0 --pyramid-weight 0.0 --dihedral-augmentation --log-every 100
)

run_one() {
  local gpu=$1 fold=$2 name=$3 weight=$4 train=$5 validation=$6
  local dir="$OUT/selection/fold$fold/$name"
  mkdir -p "$dir"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON" "$SRC/weighted_new_data_train.py" --mode select \
      --exp-dir "$dir" --train-objects "$train" --val-objects "$validation" \
      --new-object-weight "$weight" --seed "$((20261100 + fold))" \
      "${common[@]}" > "$OUT/selection/fold$fold-$name.log" 2>&1 &
}

for fold in 1 2 3 4; do
  train_var=TRAIN_$fold
  val_var=VAL_$fold
  run_one 0 "$fold" w000 0.00 "${!train_var}" "${!val_var}"
  run_one 1 "$fold" w025 0.25 "${!train_var}" "${!val_var}"
  run_one 2 "$fold" w050 0.50 "${!train_var}" "${!val_var}"
  run_one 3 "$fold" w100 1.00 "${!train_var}" "${!val_var}"
  wait
done

"$PYTHON" - "$OUT/selection" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
scores = defaultdict(list)
for path in sorted(root.glob("fold*/*/selection.json")):
    result = json.loads(path.read_text())
    split = json.loads((path.parent / "split_audit.json").read_text())
    scores[path.parent.name].append({
        "fold": path.parent.parent.name,
        "ssim": float(result["best_validation_ssim"]),
        "objects": split["validation"],
        "new_object_weight": float(result["new_object_weight"]),
    })
if set(scores) != {"w000", "w025", "w050", "w100"} or any(len(v) != 4 for v in scores.values()):
    raise SystemExit(f"expected four weights x four folds, got {dict(scores)}")
ranking = []
for name, rows in scores.items():
    total = sum(len(row["objects"]) for row in rows)
    ranking.append({
        "name": name,
        "new_object_weight": rows[0]["new_object_weight"],
        "cross_validation_ssim": sum(row["ssim"] * len(row["objects"]) for row in rows) / total,
        "folds": rows,
    })
ranking.sort(key=lambda row: row["cross_validation_ssim"], reverse=True)
(root / "ranking.json").write_text(json.dumps(ranking, indent=2) + "\n")
print(json.dumps(ranking, indent=2))
PY
echo completed > "$OUT/SELECTION_DONE"
