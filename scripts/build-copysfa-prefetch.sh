#!/usr/bin/env bash
# Rebuild the existing fused LIM and CopySFA kernels plus their changed ABI.
set -eo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
source scripts/vllm-ascend-env.sh
set -u
export CMAKE_BUILD_PARALLEL_LEVEL=4 MAX_JOBS=4
BUILD="$ROOT/vllm-ascend/csrc/build"
BIN="$BUILD/binary/ascend910_93"
[[ -f "$BUILD/build.ninja" ]] || { printf 'Run forge-configure-lim.sh first.\n' >&2; exit 1; }
REFERENCE_BUILD=${1:?Pass the prepared vLLM native dependency build directory}
mkdir -p build/copysfa-prefetch-v2
python scripts/build-copysfa-prefetch-binding.py --reference-build "$REFERENCE_BUILD"
cmake --build "$BUILD" --target op_host_aclnnInner -j4
OPS_PROTO_SEPARATE=1 OPS_ACLNN_GEN=1 OPS_PROJECT_NAME=aclnnInner \
OPS_PRODUCT_NAME='ascend910_93;' "$ASCEND_HOME_PATH/tools/opbuild/op_build" \
    "$BUILD/libop_host_aclnnInner.so" "$BUILD/autogen/inner"
cmake --build "$BUILD" --target generate_transformer_adapt_py generate_ops_info generate_compile_cmd -j4
BACKUP=$(mktemp -d "$ROOT/build/copysfa-prefetch-v2/previous.XXXXXX")
shopt -s nullglob
for op in fused_lightning_indexer_manage fused_scatter_copy_sparse_flash_attention; do
    mkdir -p "$BACKUP/$op" "$BIN/src/$op"
    cp -R "vllm-ascend/csrc/attention/$op/op_kernel" "$BIN/src/$op/"
    case "$op" in
        fused_lightning_indexer_manage) CLASS=FusedLightningIndexerManage ;;
        fused_scatter_copy_sparse_flash_attention) CLASS=FusedScatterCopySparseFlashAttention ;;
    esac
    cp "$BUILD/impl/dynamic/$op.py" "$BIN/src/$op/$CLASS.py"
    for generated in "$BIN/gen/${op}_ascend910_93_"*.done "$BIN/bin/$op"; do
        if [[ -e "$generated" ]]; then mv "$generated" "$BACKUP/$op/"; fi
    done
done
cmake --build "$BUILD" --target package -j4
INSTALL="$ROOT/vllm-ascend/vllm_ascend/_cann_ops_custom"
if [[ -d "$INSTALL" ]]; then mv "$INSTALL" "$BACKUP/opp"; fi
if ! bash "$BUILD/cann-ops-transformer-custom_linux-aarch64.run" --quiet --install-path="$INSTALL"; then
    if [[ -d "$INSTALL" ]]; then mv "$INSTALL" "$BACKUP/failed-install"; fi
    if [[ -d "$BACKUP/opp" ]]; then mv "$BACKUP/opp" "$INSTALL"; fi
    exit 1
fi
git show HEAD:vllm-ascend/vllm_ascend/_cann_ops_custom/.gitkeep > "$INSTALL/.gitkeep"
cp vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so "$BACKUP/"
cp build/copysfa-prefetch-v2/binding/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so vllm-ascend/vllm_ascend/
