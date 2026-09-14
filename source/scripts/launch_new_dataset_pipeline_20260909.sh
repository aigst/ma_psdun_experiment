#!/usr/bin/env bash
set -euo pipefail

ROOT=${MA_PSDUN_ROOT:-/sci_persistent_storage/ma_psdun_new_20260909}
SRC=${MA_PSDUN_SRC:-$ROOT/source}
DATA=${MA_PSDUN_DATA:-$ROOT/data}
PATTERNS=${MA_PSDUN_PATTERNS:-$DATA/pattern.tif}
OUT=${MA_PSDUN_OUT:-$ROOT/results}
PYTHON=${MA_PSDUN_PYTHON:-python3}
GPU_COUNT=${MA_PSDUN_GPU_COUNT:-4}
mkdir -p "$OUT/selection"

TRAIN=sample_01,sample_02,sample_03,sample_04,sample_06,sample_07,sample_08,sample_09,sample_11,sample_13,sample_17,sample_18
VAL=sample_05,sample_10,sample_15,sample_20,sample_24
TEST=sample_12,sample_16,sample_21,sample_22
FORBIDDEN=sample_14,sample_19,sample_23,sample_25

common=(
  --data-root "$DATA" --patterns "$PATTERNS"
  --train-objects "$TRAIN" --val-objects "$VAL" --test-objects "$TEST"
  --audit-objects "" --forbidden-supervised-objects "$FORBIDDEN"
  --label-resample bilinear --condition-mode constant
  --gain-init 4.0 --lowpass-kernel 3 --prior-residual-scale 0.5
  --epochs 600 --lr 1e-4 --weight-decay 1e-5
  --consistency-weight 0.01 --ssim-weight 0.2 --binary-weight 0.0
  --tv-weight 0.0 --dice-weight 0.0 --pyramid-weight 0.0
  --log-every 100
)

run_one() {
  local gpu=$1 name=$2 seed=$3
  shift 3
  mkdir -p "$OUT/selection/$name"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON" "$SRC/ring_blind_train.py" --mode select \
      --exp-dir "$OUT/selection/$name" --seed "$seed" \
      "${common[@]}" "$@" > "$OUT/selection/$name.log" 2>&1 &
  echo "started $name gpu=$gpu"
}

if [[ "$GPU_COUNT" == 1 ]]; then
  run_one 0 mean_d31_s8 202609091 --preprocess detrend --detrend-width 31 --fusion mean --stages 8 --edge-weight 0.00; wait
  run_one 0 mean_d31_s12_edge 202609092 --preprocess detrend --detrend-width 31 --fusion mean --stages 12 --edge-weight 0.05 --dihedral-augmentation; wait
  run_one 0 multi_d31_s12_edge 202609093 --preprocess detrend --detrend-width 31 --fusion multi --stages 12 --edge-weight 0.05 --dihedral-augmentation; wait
  run_one 0 multi_d127_s12_edge 202609094 --preprocess detrend --detrend-width 127 --fusion multi --stages 12 --edge-weight 0.05 --dihedral-augmentation; wait
  run_one 0 multi_robust31_s12 202609095 --preprocess detrend_robust --detrend-width 31 --fusion multi --stages 12 --edge-weight 0.05 --dihedral-augmentation; wait
  run_one 0 multi_d31_s12_tv 202609096 --preprocess detrend --detrend-width 31 --fusion multi --stages 12 --edge-weight 0.05 --tv-weight 0.01 --dihedral-augmentation; wait
  run_one 0 multi_d31_s12_psf05 202609097 --preprocess detrend --detrend-width 31 --fusion multi --stages 12 --psf-sigma 0.5 --edge-weight 0.05 --dihedral-augmentation; wait
  run_one 0 multi_d31_s12_stats 202609098 --preprocess detrend --detrend-width 31 --fusion multi --stages 12 --condition-mode stats --edge-weight 0.05 --dihedral-augmentation; wait
else
  run_one 0 mean_d31_s8 202609091 --preprocess detrend --detrend-width 31 --fusion mean --stages 8 --edge-weight 0.00
  run_one 1 mean_d31_s12_edge 202609092 --preprocess detrend --detrend-width 31 --fusion mean --stages 12 --edge-weight 0.05 --dihedral-augmentation
  run_one 2 multi_d31_s12_edge 202609093 --preprocess detrend --detrend-width 31 --fusion multi --stages 12 --edge-weight 0.05 --dihedral-augmentation
  run_one 3 multi_d127_s12_edge 202609094 --preprocess detrend --detrend-width 127 --fusion multi --stages 12 --edge-weight 0.05 --dihedral-augmentation
  wait
  run_one 0 multi_robust31_s12 202609095 --preprocess detrend_robust --detrend-width 31 --fusion multi --stages 12 --edge-weight 0.05 --dihedral-augmentation
  run_one 1 multi_d31_s12_tv 202609096 --preprocess detrend --detrend-width 31 --fusion multi --stages 12 --edge-weight 0.05 --tv-weight 0.01 --dihedral-augmentation
  run_one 2 multi_d31_s12_psf05 202609097 --preprocess detrend --detrend-width 31 --fusion multi --stages 12 --psf-sigma 0.5 --edge-weight 0.05 --dihedral-augmentation
  run_one 3 multi_d31_s12_stats 202609098 --preprocess detrend --detrend-width 31 --fusion multi --stages 12 --condition-mode stats --edge-weight 0.05 --dihedral-augmentation
  wait
fi

"$PYTHON" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("selection/*/selection.json")):
    result = json.loads(path.read_text())
    config = json.loads((path.parent / "config.json").read_text())
    rows.append({"name": path.parent.name,
                 "validation_ssim": float(result["best_validation_ssim"]),
                 "best_epoch": result.get("best_epoch"),
                 "config": {k: config.get(k) for k in ("preprocess", "detrend_width", "fusion", "condition_mode", "stages", "psf_sigma", "edge_weight", "tv_weight", "dihedral_augmentation")}})
if len(rows) != 8:
    raise SystemExit(f"expected 8 selection results, found {len(rows)}")
rows.sort(key=lambda x: x["validation_ssim"], reverse=True)
(root / "selection_ranking.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
(root / "SELECTED_NAME").write_text(rows[0]["name"] + "\n")
print(json.dumps({"selected": rows[0], "ranking": rows}, ensure_ascii=False, indent=2))
PY

BEST=$(tr -d '[:space:]' < "$OUT/SELECTED_NAME")
case "$BEST" in
  mean_d31_s8) EXTRA=(--preprocess detrend --detrend-width 31 --fusion mean --stages 8 --edge-weight 0.00);;
  mean_d31_s12_edge) EXTRA=(--preprocess detrend --detrend-width 31 --fusion mean --stages 12 --edge-weight 0.05 --dihedral-augmentation);;
  multi_d31_s12_edge) EXTRA=(--preprocess detrend --detrend-width 31 --fusion multi --stages 12 --edge-weight 0.05 --dihedral-augmentation);;
  multi_d127_s12_edge) EXTRA=(--preprocess detrend --detrend-width 127 --fusion multi --stages 12 --edge-weight 0.05 --dihedral-augmentation);;
  multi_robust31_s12) EXTRA=(--preprocess detrend_robust --detrend-width 31 --fusion multi --stages 12 --edge-weight 0.05 --dihedral-augmentation);;
  multi_d31_s12_tv) EXTRA=(--preprocess detrend --detrend-width 31 --fusion multi --stages 12 --edge-weight 0.05 --tv-weight 0.01 --dihedral-augmentation);;
  multi_d31_s12_psf05) EXTRA=(--preprocess detrend --detrend-width 31 --fusion multi --stages 12 --psf-sigma 0.5 --edge-weight 0.05 --dihedral-augmentation);;
  multi_d31_s12_stats) EXTRA=(--preprocess detrend --detrend-width 31 --fusion multi --stages 12 --condition-mode stats --edge-weight 0.05 --dihedral-augmentation);;
  *) echo "unknown selected candidate: $BEST" >&2; exit 1;;
esac

mkdir -p "$OUT/final"
if [[ "$GPU_COUNT" == 1 ]]; then
  for gpu in 0; do
    for seed in 202609100 202609101 202609102 202609103; do
      name="seed${seed}"
      CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        "$PYTHON" "$SRC/fit_new_dataset.py" --data-root "$DATA" --patterns "$PATTERNS" \
          --exp-dir "$OUT/final/$name" --seed "$seed" --epochs 800 --log-every 100 \
          --consistency-weight 0.01 --ssim-weight 0.2 --weight-decay 1e-5 \
          --lowpass-kernel 3 --prior-residual-scale 0.5 "${EXTRA[@]}" \
          > "$OUT/final/$name.log" 2>&1
    done
  done
else
for gpu in 0 1 2 3; do
  seed=$((202609100 + gpu))
  name="seed${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON" "$SRC/fit_new_dataset.py" --data-root "$DATA" --patterns "$PATTERNS" \
      --exp-dir "$OUT/final/$name" --seed "$seed" --epochs 800 --log-every 100 \
      --consistency-weight 0.01 --ssim-weight 0.2 --weight-decay 1e-5 \
      --lowpass-kernel 3 --prior-residual-scale 0.5 "${EXTRA[@]}" \
      > "$OUT/final/$name.log" 2>&1 &
done
wait
fi

for gpu in 0 1 2 3; do
  seed=$((202609100 + gpu))
  "$PYTHON" "$SRC/predict_new_dataset.py" --data-root "$DATA" --patterns "$PATTERNS" \
    --checkpoint "$OUT/final/seed${seed}/checkpoint_final.pt" --output "$OUT/pred_seed${seed}" --device cuda:0
done

"$PYTHON" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
root = Path(sys.argv[1])
dirs = sorted(root.glob("pred_seed*/deployment_predictions.pt"))
if len(dirs) != 4:
    raise SystemExit(f"expected 4 prediction members, found {len(dirs)}")
packs = [torch.load(p, map_location="cpu", weights_only=False) for p in dirs]
pred = torch.stack([p["pred"] for p in packs]).mean(0)
records = packs[0]["records"]
out = root / "ensemble"
out.mkdir(exist_ok=True)
torch.save({"pred": pred, "records": records, "members": [str(p.parent) for p in dirs]}, out / "deployment_predictions.pt")
for image, record in zip(pred.numpy(), records):
    Image.fromarray(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8), mode="L").save(out / f"{record['object_id']}.png")
size = pred.shape[-1]
canvas = Image.new("L", (5 * size * 4, ((len(records) + 4) // 5) * size * 4), 0)
for idx, image in enumerate(pred.numpy()):
    tile = Image.fromarray(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8), mode="L").resize((size * 4, size * 4), Image.Resampling.NEAREST)
    canvas.paste(tile, ((idx % 5) * size * 4, (idx // 5) * size * 4))
canvas.save(out / "deployment_montage.png")
(out / "deployment_manifest.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({"status": "completed", "members": len(dirs), "unlabelled": [r["source_name"] for r in records if not r["labelled"]]}, ensure_ascii=False))
PY
echo completed > "$OUT/PIPELINE_DONE"
