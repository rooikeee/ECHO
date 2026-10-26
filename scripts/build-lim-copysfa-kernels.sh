#!/usr/bin/env bash
# Rebuild selected device kernels without changing their host signatures.
set -eo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
source scripts/vllm-ascend-env.sh
set -u
BUILD="$ROOT/vllm-ascend/csrc/build"
BIN="$BUILD/binary/ascend910_93"
if [[ "$#" -eq 0 ]]; then
    printf 'Specify fused_lightning_indexer_manage or fused_scatter_copy_sparse_flash_attention.\n' >&2
    exit 1
fi
for op in "$@"; do
    case "$op" in
        fused_lightning_indexer_manage|fused_scatter_copy_sparse_flash_attention) ;;
        *) printf 'Unsupported operator: %s\n' "$op" >&2; exit 1 ;;
    esac
done
BACKUP=$(mktemp -d "$ROOT/build/lim-copysfa-generated.XXXXXX")
shopt -s nullglob
for op in "$@"; do
    cp -R "vllm-ascend/csrc/attention/$op/op_kernel" "$BIN/src/$op/"
    for generated in "$BIN/gen/${op}_ascend910_93_"*.done "$BIN/bin/$op"; do
        if [[ -e "$generated" ]]; then
            mv "$generated" "$BACKUP/"
        fi
    done
    cmake --build "$BUILD" --target "$op" -j "$MAX_JOBS"
done
cmake --build "$BUILD" --target package -j "$MAX_JOBS"
bash "$BUILD/cann-ops-transformer-custom_linux-aarch64.run" --quiet \
    --install-path="$ROOT/vllm-ascend/vllm_ascend/_cann_ops_custom"
