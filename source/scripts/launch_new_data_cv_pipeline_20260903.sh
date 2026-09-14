#!/usr/bin/env bash
set -euo pipefail

ROOT=${MA_PSDUN_ROOT:-/sci_persistent_storage/ma_psdun_v2_20260903}
SRC=${MA_PSDUN_SRC:-$ROOT/source}
DATA=${MA_PSDUN_DATA:-$ROOT/data/sample}
PATTERNS=${MA_PSDUN_PATTERNS:-$ROOT/data/4096.tif}
OUT=${MA_PSDUN_CV_OUT:-$ROOT/results_cv}
PYTHON=${MA_PSDUN_PYTHON:-python3}

TEST=obj18-20260901,obj19-20260901,obj20-20260901
AUDIT=obj15-20260901,obj16-20260901
FORBIDDEN=obj15-20260901,obj16-20260901,obj19-20260901
ALL=bar1-20260830,bar10-20260831,bar2-20260830,bar3-20260830,bar4-20260830,bar5-20260830,bar6-20260831,bar7-20260831,bar8-20260831,bar9-20260831,obj11-20260901,obj13-20260901,obj14-20260901,obj17-20260901,obj22-20260902,obj24-20260903,obj25-20260903
VAL_1=bar1-20260830,bar5-20260830,obj11-20260901,obj24-20260903
VAL_2=bar2-20260830,bar6-20260831,obj13-20260901,obj25-20260903
VAL_3=bar3-20260830,bar7-20260831,bar9-20260831,obj14-20260901
VAL_4=bar4-20260830,bar8-20260831,bar10-20260831,obj17-20260901,obj22-20260902
TRAIN_1=bar10-20260831,bar2-20260830,bar3-20260830,bar4-20260830,bar6-20260831,bar7-20260831,bar8-20260831,bar9-20260831,obj13-20260901,obj14-20260901,obj17-20260901,obj22-20260902,obj25-20260903
TRAIN_2=bar1-20260830,bar10-20260831,bar3-20260830,bar4-20260830,bar5-20260830,bar7-20260831,bar8-20260831,bar9-20260831,obj11-20260901,obj14-20260901,obj17-20260901,obj22-20260902,obj24-20260903
TRAIN_3=bar1-20260830,bar10-20260831,bar2-20260830,bar4-20260830,bar5-20260830,bar6-20260831,bar8-20260831,obj11-20260901,obj13-20260901,obj17-20260901,obj22-20260902,obj24-20260903,obj25-20260903
TRAIN_4=bar1-20260830,bar2-20260830,bar3-20260830,bar5-20260830,bar6-20260831,bar7-20260831,bar9-20260831,obj11-20260901,obj13-20260901,obj14-20260901,obj24-20260903,obj25-20260903

mkdir -p "$OUT/selection" "$OUT/final"

common_args=(
  --data-root "$DATA" --patterns "$PATTERNS"
  --test-objects "$TEST" --audit-objects "$AUDIT"
  --forbidden-supervised-objects "$FORBIDDEN"
  --label-resample bilinear --condition-mode constant
  --stages 12 --gain-init 4.0 --lowpass-kernel 3
  --prior-residual-scale 0.5 --epochs 600 --lr 1e-4
  --weight-decay 1e-5 --consistency-weight 0.05
  --ssim-weight 0.2 --tv-weight 0.0
  --dihedral-augmentation --log-every 100
)

run_candidate() {
  local gpu=$1 fold=$2 name=$3 train=$4 validation=$5
  shift 5
  mkdir -p "$OUT/selection/fold$fold"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON" "$SRC/ring_blind_train.py" \
      --mode select --exp-dir "$OUT/selection/fold$fold/$name" \
      --train-objects "$train" --val-objects "$validation" \
      --seed "$((20260950 + fold))" "$@" "${common_args[@]}" \
      > "$OUT/selection/fold$fold/$name.log" 2>&1 &
}

for fold in 1 2 3 4; do
  train_var=TRAIN_$fold
  val_var=VAL_$fold
  train=${!train_var}
  validation=${!val_var}
  run_candidate 0 "$fold" mean_detrend "$train" "$validation" \
    --preprocess detrend --detrend-width 31 --fusion mean
  run_candidate 1 "$fold" mean_robust "$train" "$validation" \
    --preprocess detrend_robust --detrend-width 31 --fusion mean
  run_candidate 2 "$fold" mean_edge_light "$train" "$validation" \
    --preprocess detrend --detrend-width 31 --fusion mean \
    --edge-weight 0.02 --dice-weight 0.05 --pyramid-weight 0.05
  run_candidate 3 "$fold" mean_edge_full "$train" "$validation" \
    --preprocess detrend --detrend-width 31 --fusion mean \
    --edge-weight 0.05 --dice-weight 0.10 --pyramid-weight 0.10
  wait
done

BEST_NAME=$(
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
    })
if any(len(rows) != 4 for rows in scores.values()) or len(scores) != 4:
    raise SystemExit(f"expected four configurations x four folds, got {dict(scores)}")
ranking = []
for name, folds in scores.items():
    total = sum(len(row["objects"]) for row in folds)
    weighted = sum(row["ssim"] * len(row["objects"]) for row in folds) / total
    ranking.append({"name": name, "cross_validation_ssim": weighted, "folds": folds})
ranking.sort(key=lambda row: row["cross_validation_ssim"], reverse=True)
(root / "ranking.json").write_text(json.dumps(ranking, indent=2) + "\n")
print(ranking[0]["name"])
PY
)
echo "selected cross-validation configuration: $BEST_NAME"

case "$BEST_NAME" in
  mean_detrend)
    selected_args=(--preprocess detrend --detrend-width 31 --fusion mean)
    ;;
  mean_robust)
    selected_args=(--preprocess detrend_robust --detrend-width 31 --fusion mean)
    ;;
  mean_edge_light)
    selected_args=(--preprocess detrend --detrend-width 31 --fusion mean --edge-weight 0.02 --dice-weight 0.05 --pyramid-weight 0.05)
    ;;
  mean_edge_full)
    selected_args=(--preprocess detrend --detrend-width 31 --fusion mean --edge-weight 0.05 --dice-weight 0.10 --pyramid-weight 0.10)
    ;;
  *)
    echo "unknown selected configuration: $BEST_NAME" >&2
    exit 1
    ;;
esac

run_final() {
  local gpu=$1 seed=$2
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON" "$SRC/ring_blind_train.py" \
      --mode final --exp-dir "$OUT/final/seed$seed" \
      --train-objects "$ALL" --seed "$seed" \
      "${selected_args[@]}" "${common_args[@]}" \
      > "$OUT/final/seed$seed.log" 2>&1 &
}

run_final 0 20260971
run_final 1 20260972
run_final 2 20260973
run_final 3 20260974
wait

members=$OUT/final/seed20260971,$OUT/final/seed20260972,$OUT/final/seed20260973,$OUT/final/seed20260974
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PYTHON" "$SRC/ring_blind_train.py" \
    --mode ensemble --exp-dir "$OUT/final/ensemble" --members "$members"

selection_dirs=$OUT/selection/fold1/$BEST_NAME,$OUT/selection/fold2/$BEST_NAME,$OUT/selection/fold3/$BEST_NAME,$OUT/selection/fold4/$BEST_NAME
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PYTHON" "$SRC/ring_blind_calibrate.py" \
    --selection-dirs "$selection_dirs" \
    --ensemble-predictions "$OUT/final/ensemble/predictions.pt" \
    --exp-dir "$OUT/final/calibrated" > "$OUT/final/calibration.log" 2>&1

"$PYTHON" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
ranking = json.loads((root / "selection" / "ranking.json").read_text())
calibration = json.loads((root / "final" / "calibrated" / "calibration.json").read_text())
summary = {
    "status": "completed",
    "selection_ranking": ranking,
    "selected_configuration": ranking[0]["name"],
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

