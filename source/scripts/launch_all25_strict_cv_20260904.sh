#!/usr/bin/env bash
set -euo pipefail

ROOT=${MA_PSDUN_ROOT:-/sci_persistent_storage/ma_psdun_v2_20260903}
SRC=${MA_PSDUN_SRC:-$ROOT/source/all25_fullfit}
DATA=${MA_PSDUN_DATA:-$ROOT/data/sample}
PATTERNS=${MA_PSDUN_PATTERNS:-$ROOT/data/4096.tif}
OUT=${MA_PSDUN_ALL25_CV_OUT:-$ROOT/results_all25_strict_cv_20260904}
PYTHON=${MA_PSDUN_PYTHON:-python3}

TEST=obj18-20260901,obj19-20260901,obj20-20260901
AUDIT=obj15-20260901,obj16-20260901
FORBIDDEN=obj15-20260901,obj16-20260901,obj19-20260901

TRAIN_1=bar2-20260830,bar3-20260830,bar4-20260830,bar6-20260831,bar7-20260831,bar8-20260831,bar9-20260831,bar10-20260831,obj13-20260901,obj14-20260901,obj17-20260901,obj22-20260902,obj23-20260902,obj24-20260903,obj25-20260903
VAL_1=bar1-20260830,bar5-20260830,obj11-20260901,obj12-20260901,obj21-20260902
TRAIN_2=bar1-20260830,bar3-20260830,bar4-20260830,bar5-20260830,bar7-20260831,bar8-20260831,bar9-20260831,bar10-20260831,obj11-20260901,obj12-20260901,obj14-20260901,obj17-20260901,obj21-20260902,obj24-20260903,obj25-20260903
VAL_2=bar2-20260830,bar6-20260831,obj13-20260901,obj22-20260902,obj23-20260902
TRAIN_3=bar1-20260830,bar2-20260830,bar4-20260830,bar5-20260830,bar6-20260831,bar8-20260831,bar10-20260831,obj11-20260901,obj12-20260901,obj13-20260901,obj17-20260901,obj21-20260902,obj22-20260902,obj23-20260902,obj25-20260903
VAL_3=bar3-20260830,bar7-20260831,bar9-20260831,obj14-20260901,obj24-20260903
TRAIN_4=bar1-20260830,bar2-20260830,bar3-20260830,bar5-20260830,bar6-20260831,bar7-20260831,bar9-20260831,obj11-20260901,obj12-20260901,obj13-20260901,obj14-20260901,obj21-20260902,obj22-20260902,obj23-20260902,obj24-20260903
VAL_4=bar4-20260830,bar8-20260831,bar10-20260831,obj17-20260901,obj25-20260903

common=(
  --data-root "$DATA" --patterns "$PATTERNS" --test-objects "$TEST"
  --audit-objects "$AUDIT" --forbidden-supervised-objects "$FORBIDDEN"
  --label-resample bilinear --psf-sigma 1.0 --stages 12 --gain-init 4.0
  --lowpass-kernel 3 --prior-residual-scale 0.5 --epochs 600 --lr 1e-4
  --weight-decay 1e-5 --consistency-weight 0.05 --ssim-weight 0.2
  --tv-weight 0.0 --edge-weight 0.0 --log-every 100
)

run_candidate() {
  local fold=$1 gpu=$2 name=$3 train=$4 validation=$5
  shift 5
  mkdir -p "$OUT/selection/fold$fold/$name"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON" "$SRC/ring_blind_train.py" --mode select \
      --exp-dir "$OUT/selection/fold$fold/$name" \
      --train-objects "$train" --val-objects "$validation" \
      --seed "$((20260990 + fold * 10 + gpu))" "$@" "${common[@]}" \
      > "$OUT/selection/fold$fold-$name.log" 2>&1 &
}

for fold in 1 2 3 4; do
  train_var=TRAIN_$fold
  val_var=VAL_$fold
  train=${!train_var}
  validation=${!val_var}
  run_candidate "$fold" 0 mean_dihedral "$train" "$validation" \
    --preprocess detrend --detrend-width 127 --fusion mean --dihedral-augmentation
  run_candidate "$fold" 1 multi_dihedral "$train" "$validation" \
    --preprocess detrend --detrend-width 127 --fusion multi --dihedral-augmentation
  run_candidate "$fold" 2 robust_dihedral "$train" "$validation" \
    --preprocess detrend_robust --detrend-width 127 --fusion mean --dihedral-augmentation
  run_candidate "$fold" 3 stats_dihedral "$train" "$validation" \
    --preprocess detrend --detrend-width 127 --fusion mean --condition-mode stats --dihedral-augmentation
  wait
done

"$PYTHON" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("selection/fold*/ */selection.json")):
    pass
for path in sorted(root.glob("selection/fold*/*/selection.json")):
    fold = int(path.parent.parent.name.removeprefix("fold"))
    name = path.parent.name
    result = json.loads(path.read_text())
    config = json.loads((path.parent / "config.json").read_text())
    rows.append({
        "fold": fold,
        "name": name,
        "best_validation_ssim": float(result["best_validation_ssim"]),
        "objects": json.loads((path.parent / "split_audit.json").read_text())["validation"],
        "test_labels_evaluated": result["test_labels_evaluated"],
        "config": {key: config[key] for key in ("preprocess", "detrend_width", "fusion", "condition_mode", "dihedral_augmentation")},
    })
if len(rows) != 16:
    raise SystemExit(f"expected 16 completed folds, found {len(rows)}")
ranking = []
for name in sorted({row["name"] for row in rows}):
    selected = [row for row in rows if row["name"] == name]
    score = sum(row["best_validation_ssim"] * len(row["objects"]) for row in selected) / sum(len(row["objects"]) for row in selected)
    ranking.append({"name": name, "cross_validation_ssim": score, "folds": selected})
ranking.sort(key=lambda row: row["cross_validation_ssim"], reverse=True)
(root / "ranking.json").write_text(json.dumps(ranking, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(ranking, indent=2, ensure_ascii=False))
PY
echo completed > "$OUT/CV_DONE"
