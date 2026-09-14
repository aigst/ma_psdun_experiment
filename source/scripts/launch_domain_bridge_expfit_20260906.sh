#!/usr/bin/env bash
set -euo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

PYTHON_BIN="${MA_PSDUN_PYTHON:-/sci_persistent_storage/ma_psdun_v2_20260903/venv-cu121/bin/python}"
SOURCE_ROOT="${MA_PSDUN_SOURCE_ROOT:-/root/ma_psdun_expfit_20260906}"
DATA_ROOT="${MA_PSDUN_DATA_ROOT:-/sci_persistent_storage/ma_psdun_v2_20260903/data/sample}"
PATTERNS="${MA_PSDUN_PATTERNS:-/sci_persistent_storage/ma_psdun_v2_20260903/data/4096.tif}"
OUT_ROOT="${MA_PSDUN_OUT_ROOT:-/sci_persistent_storage/ma_psdun_v2_20260903/domain_bridge_expfit_v2_20260906}"

if [[ -e "${OUT_ROOT}" ]]; then
  echo "refusing to overwrite existing output directory: ${OUT_ROOT}" >&2
  exit 2
fi
mkdir -p "${OUT_ROOT}/logs"

common=(
  --data-root "${DATA_ROOT}"
  --patterns "${PATTERNS}"
  --out-dir "${OUT_ROOT}"
  --device cuda:0
  --preprocess detrend
  --detrend-width 127
  --gauss-sigma 40.0
  --stages 12
  --lowpass-kernel 3
  --prior-residual-scale 0.50
  --psf-sigma 0.3987402916
  --psf-sigma-y 0.2762783468
  --psf-angle -1.292086482
  --pretrain-steps 120
  --pretrain-lr 1e-4
  --finetune-epochs 80
  --finetune-lr 3e-5
  --synthetic-batch 8
  --synthetic-mix 0.5
  --noise-mode empirical
  --fold-local-noise
  --noise-scale 0.005
  --dark-noise-scale 0.005
  --fit-attenuation
  --attenuation-trim-fraction 0.01
  --attenuation-jitter 0.04
  --preserve-od-amplitude
  --synthetic-dataset-size 960
  --synthetic-generation-batch 32
  --uda-steps 60
  --tto-steps 100
  --seed 20260906
  --split-seed 20260902
  --log-every 20
)

pids=()
for fold in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${fold}" "${PYTHON_BIN}" -u "${SOURCE_ROOT}/domain_bridge_train.py" \
    --mode fold --fold "${fold}" "${common[@]}" \
    >"${OUT_ROOT}/logs/fold${fold}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "one or more folds failed; inspect ${OUT_ROOT}/logs" >&2
  exit 1
fi

"${PYTHON_BIN}" -u "${SOURCE_ROOT}/domain_bridge_train.py" \
  --mode ensemble --out-dir "${OUT_ROOT}" --ensemble-size 4 \
  >"${OUT_ROOT}/logs/ensemble.log" 2>&1

CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" -u "${SOURCE_ROOT}/evaluate_domain_bridge_reload.py" \
  --exp-dir "${OUT_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --patterns "${PATTERNS}" \
  --label-resample lanczos \
  --psf-sigma 0.3987402916 \
  --psf-sigma-y 0.2762783468 \
  --psf-angle -1.292086482 \
  --preserve-od-amplitude \
  --output "${OUT_ROOT}/independent_reload.json" \
  >"${OUT_ROOT}/logs/independent_reload.log" 2>&1

date -Iseconds >"${OUT_ROOT}/COMPLETE"
