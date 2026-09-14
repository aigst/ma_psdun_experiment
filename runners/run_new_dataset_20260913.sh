#!/usr/bin/env bash
set -euo pipefail

# This run is isolated from all previous sample_new artifacts.
ROOT=${MA_PSDUN_ROOT:-/sci_persistent_storage/ma_psdun_new_20260913}
DOWNLOADS="$ROOT/downloads"
RAW="$ROOT/raw"
SRC="$ROOT/new_exp_src"
DATA="$ROOT/prepared"
OUT="$ROOT/results"
PYTHON=${MA_PSDUN_PYTHON:-python3}
GPU_COUNT=${MA_PSDUN_GPU_COUNT:-4}
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
DATA_URL=${DATA_URL:?Set DATA_URL to the approved internal dataset URL}
SOURCE_URL=${SOURCE_URL:?Set SOURCE_URL to the approved internal source archive URL}
DATA_ARCHIVE="$DOWNLOADS/2026-09-13_15-42-08_sample_new(1).zip"
SOURCE_ARCHIVE="$DOWNLOADS/ma_psdun_source_20260913.tar.gz"
mkdir -p "$DOWNLOADS" "$RAW" "$OUT"
exec > >(tee -a "$ROOT/run.log") 2>&1

download() {
  local url=$1 output=$2
  if [[ ! -f "$output" ]]; then
    curl --noproxy '*' --fail --location --retry 5 --retry-delay 3 \
      --output "$output.part" "$url"
    mv "$output.part" "$output"
  fi
}

download "$DATA_URL" "$DATA_ARCHIVE"
download "$SOURCE_URL" "$SOURCE_ARCHIVE"

if [[ ! -f "$RAW/.extracted" ]]; then
  "$PYTHON" - "$DATA_ARCHIVE" "$RAW" <<'PY'
import sys, zipfile
archive, output = sys.argv[1:]
with zipfile.ZipFile(archive) as zf:
    zf.extractall(output)
PY
  touch "$RAW/.extracted"
fi

if [[ ! -f "$SRC/.extracted" ]]; then
  mkdir -p "$ROOT/source_unpack"
  tar -xzf "$SOURCE_ARCHIVE" -C "$ROOT/source_unpack"
  mv "$ROOT/source_unpack/new_exp_src" "$SRC"
  touch "$SRC/.extracted"
fi

if [[ ! -f "$DATA/dataset_manifest.json" ]]; then
  "$PYTHON" "$SRC/tools/prepare_new_dataset.py" \
    --source "$RAW/sample_new" --output "$DATA"
fi

TRAIN=sample_01,sample_02,sample_03,sample_04,sample_05,sample_07,sample_09,sample_12,sample_13,sample_14,sample_15,sample_16,sample_20,sample_21,sample_22,sample_24,sample_25,sample_27,sample_28,sample_29
VAL=sample_10,sample_11,sample_18,sample_19,sample_23,sample_31
TEST=sample_06,sample_17,sample_30
UNLABELLED=sample_08,sample_26,sample_32
FINAL_TRAIN="$TRAIN,$VAL"

common=(
  --data-root "$DATA" --patterns "$DATA/pattern.tif"
  --train-objects "$TRAIN" --val-objects "$VAL" --test-objects "$TEST"
  --audit-objects "" --forbidden-supervised-objects "$UNLABELLED"
  --label-resample bilinear --label-crop center --label-contrast percentile
  --condition-mode constant --gain-init 4.0 --lowpass-kernel 3
  --prior-residual-scale 0.5 --epochs 600 --lr 1e-4 --weight-decay 1e-5
  --consistency-weight 0.01 --ssim-weight 0.2 --binary-weight 0.0
  --tv-weight 0.0 --dice-weight 0.0 --pyramid-weight 0.0 --log-every 100
)

wait_for_pids() {
  local rc=0 pid status
  for pid in "$@"; do
    if wait "$pid"; then
      :
    else
      status=$?
      if (( rc == 0 )); then
        rc=$status
      fi
    fi
  done
  return "$rc"
}

SELECT_PIDS=()
run_select() {
  local gpu=$1 name=$2 seed=$3
  shift 3
  mkdir -p "$OUT/selection/$name"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON" "$SRC/ring_blind_train.py" --mode select \
      --exp-dir "$OUT/selection/$name" --seed "$seed" \
      "${common[@]}" "$@" > "$OUT/selection/$name.log" 2>&1 &
  SELECT_PIDS+=("$!")
  echo "selection $name gpu=$gpu pid=$!"
}

wait_selection() {
  wait_for_pids "${SELECT_PIDS[@]}"
  SELECT_PIDS=()
}

if [[ "$GPU_COUNT" == 1 ]]; then
  run_select 0 mean_d31_s8 2026091301 --preprocess detrend --detrend-width 31 --fusion mean --stages 8 --edge-weight 0.00
  wait_selection
  run_select 0 mean_d31_s12_edge 2026091302 --preprocess detrend --detrend-width 31 --fusion mean --stages 12 --edge-weight 0.05 --dihedral-augmentation
  wait_selection
  run_select 0 multi_d31_s12_edge 2026091303 --preprocess detrend --detrend-width 31 --fusion multi --stages 12 --edge-weight 0.05 --dihedral-augmentation
  wait_selection
  run_select 0 multi_d127_s12_edge 2026091304 --preprocess detrend --detrend-width 127 --fusion multi --stages 12 --edge-weight 0.05 --dihedral-augmentation
  wait_selection
  run_select 0 multi_robust31_s12 2026091305 --preprocess detrend_robust --detrend-width 31 --fusion multi --stages 12 --edge-weight 0.05 --dihedral-augmentation
  wait_selection
  run_select 0 multi_d31_s12_tv 2026091306 --preprocess detrend --detrend-width 31 --fusion multi --stages 12 --edge-weight 0.05 --tv-weight 0.01 --dihedral-augmentation
  wait_selection
  run_select 0 multi_d31_s12_psf05 2026091307 --preprocess detrend --detrend-width 31 --fusion multi --stages 12 --psf-sigma 0.5 --edge-weight 0.05 --dihedral-augmentation
  wait_selection
  run_select 0 multi_d31_s12_psf10 2026091308 --preprocess detrend --detrend-width 31 --fusion multi --stages 12 --psf-sigma 1.0 --edge-weight 0.05 --dihedral-augmentation
  wait_selection
  run_select 0 multi_d31_s12_stats 2026091309 --preprocess detrend --detrend-width 31 --fusion multi --stages 12 --condition-mode stats --edge-weight 0.05 --dihedral-augmentation
  wait_selection
else
  run_select 0 mean_d31_s8 2026091301 --preprocess detrend --detrend-width 31 --fusion mean --stages 8 --edge-weight 0.00
  run_select 1 mean_d31_s12_edge 2026091302 --preprocess detrend --detrend-width 31 --fusion mean --stages 12 --edge-weight 0.05 --dihedral-augmentation
  run_select 2 multi_d31_s12_edge 2026091303 --preprocess detrend --detrend-width 31 --fusion multi --stages 12 --edge-weight 0.05 --dihedral-augmentation
  run_select 3 multi_d127_s12_edge 2026091304 --preprocess detrend --detrend-width 127 --fusion multi --stages 12 --edge-weight 0.05 --dihedral-augmentation
  wait_selection
  run_select 0 multi_robust31_s12 2026091305 --preprocess detrend_robust --detrend-width 31 --fusion multi --stages 12 --edge-weight 0.05 --dihedral-augmentation
  run_select 1 multi_d31_s12_tv 2026091306 --preprocess detrend --detrend-width 31 --fusion multi --stages 12 --edge-weight 0.05 --tv-weight 0.01 --dihedral-augmentation
  run_select 2 multi_d31_s12_psf05 2026091307 --preprocess detrend --detrend-width 31 --fusion multi --stages 12 --psf-sigma 0.5 --edge-weight 0.05 --dihedral-augmentation
  run_select 3 multi_d31_s12_psf10 2026091308 --preprocess detrend --detrend-width 31 --fusion multi --stages 12 --psf-sigma 1.0 --edge-weight 0.05 --dihedral-augmentation
  wait_selection
  run_select 0 multi_d31_s12_stats 2026091309 --preprocess detrend --detrend-width 31 --fusion multi --stages 12 --condition-mode stats --edge-weight 0.05 --dihedral-augmentation
  wait_selection
fi

"$PYTHON" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for path in sorted((root / "selection").glob("*/selection.json")):
    result = json.loads(path.read_text())
    config = json.loads((path.parent / "config.json").read_text())
    rows.append({
        "name": path.parent.name,
        "validation_ssim": float(result["best_validation_ssim"]),
        "best_epoch": result.get("best_epoch"),
        "config": {k: config.get(k) for k in (
            "preprocess", "detrend_width", "fusion", "condition_mode",
            "stages", "psf_sigma", "edge_weight", "tv_weight",
            "dihedral_augmentation")},
    })
if len(rows) != 9:
    raise SystemExit(f"expected 9 selection results, found {len(rows)}")
rows.sort(key=lambda row: row["validation_ssim"], reverse=True)
(root / "selection_ranking.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
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
  multi_d31_s12_psf10) EXTRA=(--preprocess detrend --detrend-width 31 --fusion multi --stages 12 --psf-sigma 1.0 --edge-weight 0.05 --dihedral-augmentation);;
  multi_d31_s12_stats) EXTRA=(--preprocess detrend --detrend-width 31 --fusion multi --stages 12 --condition-mode stats --edge-weight 0.05 --dihedral-augmentation);;
  *) echo "unknown selected candidate: $BEST" >&2; exit 1;;
esac

# Re-run the selected configuration on the selection train split for a strict,
# locked-test checkpoint. The selected epoch comes from validation only.
STRICT_EPOCH=$(( $(python3 -c 'import json; print(json.load(open("'"$OUT"'/selection_ranking.json"))[0]["best_epoch"])') + 1 ))
mkdir -p "$OUT/strict"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PYTHON" "$SRC/ring_blind_train.py" --mode final \
    --exp-dir "$OUT/strict" --seed 2026091399 \
    "${common[@]}" "${EXTRA[@]}" --epochs "$STRICT_EPOCH" > "$OUT/strict.log" 2>&1

mkdir -p "$OUT/final"
FINAL_PIDS=()
for gpu in 0 1 2 3; do
  seed=$((2026091400 + gpu))
  name="seed${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON" "$SRC/fit_new_dataset.py" \
      --data-root "$DATA" --patterns "$DATA/pattern.tif" \
      --exp-dir "$OUT/final/$name" --seed "$seed" --epochs 800 --log-every 100 \
      --consistency-weight 0.01 --ssim-weight 0.2 --weight-decay 1e-5 \
      --lowpass-kernel 3 --prior-residual-scale 0.5 --fusion multi \
      "${EXTRA[@]}" > "$OUT/final/$name.log" 2>&1 &
  FINAL_PIDS+=("$!")
done
wait_for_pids "${FINAL_PIDS[@]}"

for gpu in 0 1 2 3; do
  seed=$((2026091400 + gpu))
  "$PYTHON" "$SRC/predict_new_dataset.py" \
    --data-root "$DATA" --patterns "$DATA/pattern.tif" \
    --checkpoint "$OUT/final/seed${seed}/checkpoint_final.pt" \
    --output "$OUT/pred_seed${seed}" --device cuda:0
done

"$PYTHON" "$SRC/tools/ensemble_new_dataset.py" \
  --pred-dir "$OUT/pred_seed2026091400" "$OUT/pred_seed2026091401" \
             "$OUT/pred_seed2026091402" "$OUT/pred_seed2026091403" \
  --data-root "$DATA" --output "$OUT/ensemble" --size 64

"$PYTHON" - "$ROOT" "$OUT" "$DATA" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, out, data = map(Path, sys.argv[1:])
def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()
manifest = json.loads((data / 'dataset_manifest.json').read_text())
ranking = json.loads((out / 'selection_ranking.json').read_text())
summary = {
    'status': 'completed',
    'dataset_archive_sha256': sha256(root / 'downloads' / '2026-09-13_15-42-08_sample_new(1).zip'),
    'source_archive_sha256': sha256(root / 'downloads' / 'ma_psdun_source_20260913.tar.gz'),
    'dataset': {
        'objects': manifest['object_count'],
        'labelled': manifest['labelled_count'],
        'unlabelled': manifest['unlabelled_count'],
        'pattern_sha256': manifest['pattern']['sha256'],
    },
    'selection_ranking': ranking,
    'selected_configuration': ranking[0],
    'strict_test': json.loads((out / 'strict' / 'test.json').read_text()),
    'deployment': json.loads((out / 'ensemble' / 'deployment_summary.json').read_text()),
    'checkpoint_sha256': {
        p.parent.name: sha256(p) for p in sorted((out / 'final').glob('seed*/checkpoint_final.pt'))
    },
    'object_level_disjoint': True,
    'test_used_for_selection': False,
}
(out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
(root / 'COMPLETE').write_text('completed\n')
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
