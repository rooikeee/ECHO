#!/usr/bin/env bash
# Select one recorded source/native tree without changing editable installs.
set -eo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT/scripts/vllm-ascend-env.sh"
set -u
variant=${1:?Specify original, pr, or optimized}
shift
case "$variant" in
    original|pr) SOURCE="$ROOT/build/lim-copysfa-optimization/versions/$variant" ;;
    optimized) SOURCE="$ROOT/vllm-ascend" ;;
    *) printf 'Unknown variant: %s\n' "$variant" >&2; exit 1 ;;
esac
test -f "$SOURCE/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so"
VENDOR="$SOURCE/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
test -d "$VENDOR/op_impl"
export PYTHONPATH="$SOURCE:$ROOT/vllm${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_CUSTOM_OPP_PATH="$VENDOR"
export LD_LIBRARY_PATH="$VENDOR/op_api/lib:$SOURCE/vllm_ascend${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ECHO_BENCH_VARIANT="$variant"
export ECHO_BENCH_SOURCE="$SOURCE"
exec "$@"
