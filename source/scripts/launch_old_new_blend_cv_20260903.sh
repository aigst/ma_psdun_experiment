#!/usr/bin/env bash
set -euo pipefail

ROOT=${MA_PSDUN_ROOT:-/sci_persistent_storage/ma_psdun_v2_20260903}
SRC=${MA_PSDUN_SRC:-$ROOT/source}
DATA=${MA_PSDUN_DATA:-$ROOT/data/sample}
PATTERNS=${MA_PSDUN_PATTERNS:-$ROOT/data/4096.tif}
OUT=${MA_PSDUN_BLEND_OUT:-$ROOT/results_blend}
PYTHON=${MA_PSDUN_PYTHON:-/tmp/ma-psdun-cu121/bin/python}
OLD_FINAL=${MA_PSDUN_OLD_FINAL:-$ROOT/reference/old_strict_ensemble_predictions.pt}
NEW_ROOT=${MA_PSDUN_NEW_RESULTS:-$ROOT/results_cv}

TEST=obj18-20260901,obj19-20260901,obj20-20260901
AUDIT=obj15-20260901,obj16-20260901
FORBIDDEN=obj15-20260901,obj16-20260901,obj19-20260901
VAL_1=bar1-20260830,bar5-20260830,obj11-20260901
VAL_2=bar2-20260830,bar6-20260831,obj13-20260901
VAL_3=bar3-20260830,bar7-20260831,bar9-20260831,obj14-20260901
VAL_4=bar4-20260830,bar8-20260831,bar10-20260831,obj17-20260901
TRAIN_1=bar10-20260831,bar2-20260830,bar3-20260830,bar4-20260830,bar6-20260831,bar7-20260831,bar8-20260831,bar9-20260831,obj13-20260901,obj14-20260901,obj17-20260901
TRAIN_2=bar1-20260830,bar10-20260831,bar3-20260830,bar4-20260830,bar5-20260830,bar7-20260831,bar8-20260831,bar9-20260831,obj11-20260901,obj14-20260901,obj17-20260901
TRAIN_3=bar1-20260830,bar10-20260831,bar2-20260830,bar4-20260830,bar5-20260830,bar6-20260831,bar8-20260831,obj11-20260901,obj13-20260901,obj17-20260901
TRAIN_4=bar1-20260830,bar2-20260830,bar3-20260830,bar5-20260830,bar6-20260831,bar7-20260831,bar9-20260831,obj11-20260901,obj13-20260901,obj14-20260901

if [[ -f "$OUT/PIPELINE_DONE" ]]; then
  echo "blend pipeline already completed: $OUT"
  exit 0
fi
if [[ ! -f "$OLD_FINAL" ]]; then
  echo "missing old strict ensemble predictions: $OLD_FINAL" >&2
  exit 1
fi
if [[ ! -f "$NEW_ROOT/final/ensemble/predictions.pt" ]]; then
  echo "missing new-data ensemble predictions: $NEW_ROOT/final/ensemble/predictions.pt" >&2
  exit 1
fi

mkdir -p "$OUT/selection_old" "$OUT/final"
common_args=(
  --data-root "$DATA" --patterns "$PATTERNS"
  --test-objects "$TEST" --audit-objects "$AUDIT"
  --forbidden-supervised-objects "$FORBIDDEN"
  --label-resample bilinear --condition-mode constant
  --preprocess detrend --detrend-width 31 --fusion mean
  --stages 12 --gain-init 4.0 --lowpass-kernel 3
  --prior-residual-scale 0.5 --epochs 600 --lr 1e-4
  --weight-decay 1e-5 --consistency-weight 0.05
  --ssim-weight 0.2 --tv-weight 0.0
  --dihedral-augmentation --log-every 100
)

run_fold() {
  local gpu=$1 fold=$2 train=$3 validation=$4
  local dir="$OUT/selection_old/fold$fold"
  mkdir -p "$dir"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON" "$SRC/ring_blind_train.py" \
      --mode select --exp-dir "$dir" \
      --train-objects "$train" --val-objects "$validation" \
      --seed "$((20260950 + fold))" "${common_args[@]}" \
      > "$OUT/selection_old/fold$fold.log" 2>&1 &
  echo "old-only fold$fold pid=$! gpu=$gpu"
}

run_fold 0 1 "$TRAIN_1" "$VAL_1"
run_fold 1 2 "$TRAIN_2" "$VAL_2"
run_fold 2 3 "$TRAIN_3" "$VAL_3"
run_fold 3 4 "$TRAIN_4" "$VAL_4"
wait

for fold in 1 2 3 4; do
  test -f "$OUT/selection_old/fold$fold/SELECTION_DONE"
done

old_dirs=$OUT/selection_old/fold1,$OUT/selection_old/fold2,$OUT/selection_old/fold3,$OUT/selection_old/fold4
new_dirs=$NEW_ROOT/selection/fold1/mean_detrend,$NEW_ROOT/selection/fold2/mean_detrend,$NEW_ROOT/selection/fold3/mean_detrend,$NEW_ROOT/selection/fold4/mean_detrend
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PYTHON" "$SRC/ring_blind_blend.py" \
    --old-selection-dirs "$old_dirs" \
    --new-selection-dirs "$new_dirs" \
    --old-ensemble-predictions "$OLD_FINAL" \
    --new-ensemble-predictions "$NEW_ROOT/final/ensemble/predictions.pt" \
    --exp-dir "$OUT/final" > "$OUT/blend.log" 2>&1

"$PYTHON" - "$OUT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
blend = json.loads((root / "final" / "blend.json").read_text())

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

summary = {
    "status": "completed",
    "protocol": blend["protocol"],
    "selected": blend["selected"],
    "validation_family_metrics": blend["validation_family_metrics"],
    "primary_test": blend["primary_test"],
    "audit_ring_holdout": blend["audit_ring_holdout"],
    "object_level_disjoint": blend["object_level_disjoint"],
    "test_used_for_selection": blend["test_used_for_selection"],
    "artifact_sha256": {
        name: sha256(root / "final" / name)
        for name in ("blend.json", "predictions.pt", "preview_target_pred_4x.png")
    },
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
(root / "PIPELINE_DONE").write_text("completed\n")
print(json.dumps(summary, ensure_ascii=False))
PY
