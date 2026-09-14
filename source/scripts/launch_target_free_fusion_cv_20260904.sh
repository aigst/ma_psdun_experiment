#!/usr/bin/env bash
set -euo pipefail

ROOT=${MA_PSDUN_ROOT:-/sci_persistent_storage/ma_psdun_v2_20260903}
SRC=${MA_PSDUN_SRC:-$ROOT/source}
DATA=${MA_PSDUN_DATA:-$ROOT/data/sample}
PATTERNS=${MA_PSDUN_PATTERNS:-$ROOT/data/4096.tif}
OUT=${MA_PSDUN_FUSION_OUT:-$ROOT/results_fusion_cv_20260904}
PYTHON=${MA_PSDUN_PYTHON:-python3}
EPOCHS=${MA_PSDUN_FUSION_EPOCHS:-600}

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
common=(--data-root "$DATA" --patterns "$PATTERNS" --mode select
  --test-objects "$TEST" --audit-objects "$AUDIT" --forbidden-supervised-objects "$FORBIDDEN"
  --label-resample bilinear --condition-mode constant --preprocess detrend --detrend-width 31
  --stages 12 --gain-init 4.0 --lowpass-kernel 3 --prior-residual-scale 0.5
  --epochs "$EPOCHS" --lr 1e-4 --weight-decay 1e-5 --consistency-weight 0.05
  --ssim-weight 0.2 --tv-weight 0.0 --dihedral-augmentation --log-every 100)

run_one() {
  local gpu=$1 fold=$2 name=$3 fusion=$4
  local tr_var=TRAIN_$fold val_var=VAL_$fold
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PYTHON" "$SRC/ring_blind_train.py" \
    "${common[@]}" --exp-dir "$OUT/selection/fold$fold/$name" \
    --train-objects "${!tr_var}" --val-objects "${!val_var}" --fusion "$fusion" \
    --seed "$((20261040 + fold))" > "$OUT/selection/fold$fold-$name.log" 2>&1 &
}

for fold in 1 2 3 4; do
  run_one 0 "$fold" mean mean
  run_one 1 "$fold" weighted weighted
  run_one 2 "$fold" quality quality
  run_one 3 "$fold" agreement agreement
  wait
done

"$PYTHON" - "$OUT/selection" <<'PY'
import json, sys
from collections import defaultdict
from pathlib import Path
root = Path(sys.argv[1])
scores = defaultdict(list)
for path in sorted(root.glob('fold*/*/selection.json')):
    result = json.loads(path.read_text())
    split = json.loads((path.parent / 'split_audit.json').read_text())
    scores[path.parent.name].append({'fold': path.parent.parent.name, 'ssim': float(result['best_validation_ssim']), 'objects': split['validation']})
if any(len(rows) != 4 for rows in scores.values()) or len(scores) != 4:
    raise SystemExit(f'expected four fusions x four folds, got {dict(scores)}')
ranking = []
for name, rows in scores.items():
    total = sum(len(x['objects']) for x in rows)
    ranking.append({'name': name, 'cross_validation_ssim': sum(x['ssim']*len(x['objects']) for x in rows)/total, 'folds': rows})
ranking.sort(key=lambda x: x['cross_validation_ssim'], reverse=True)
(root / 'ranking.json').write_text(json.dumps(ranking, indent=2) + '\n')
print(json.dumps(ranking, indent=2))
PY
echo completed > "$OUT/SELECTION_DONE"
