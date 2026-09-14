#!/bin/sh
set -eu
VENV=${MA_PSDUN_VENV:-/tmp/ma_psdun_venv_cu124}
LIBROOT="$VENV/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$LIBROOT/cuda_nvrtc/lib:$LIBROOT/cuda_runtime/lib:$LIBROOT/nvjitlink/lib:$LIBROOT/cublas/lib:$LIBROOT/cudnn/lib:$LIBROOT/cufft/lib:$LIBROOT/curand/lib:$LIBROOT/cusolver/lib:$LIBROOT/cusparse/lib:$LIBROOT/nccl/lib:$LIBROOT/nvtx/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [ "${1:-}" = "-" ]; then
    shift
    exec "$VENV/bin/python" -S -c '
import site, sys
site.addsitedir("/tmp/ma_psdun_venv_cu124/lib/python3.12/site-packages")
site.addsitedir("/usr/local/lib/python3.12/dist-packages")
exec(compile(sys.stdin.read(), "<stdin>", "exec"), {"__name__": "__main__", "__file__": "<stdin>"})
' "$@"
fi
exec "$VENV/bin/python" -S -c '
import runpy, site, sys
site.addsitedir("/tmp/ma_psdun_venv_cu124/lib/python3.12/site-packages")
site.addsitedir("/usr/local/lib/python3.12/dist-packages")
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
' "$@"
