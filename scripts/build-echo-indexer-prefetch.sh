#!/usr/bin/env bash
# Apply the tracked patch and rebuild the already configured PR 17392 environment.
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PATCH=patches/vllm-ascend-echo-indexer-prefetch.patch
if git apply --check "$PATCH" 2>/dev/null; then
    git apply "$PATCH"
elif ! git apply --reverse --check "$PATCH" 2>/dev/null; then
    printf 'The prefetch patch neither applies nor matches the current source.\n' >&2
    exit 1
fi
source scripts/vllm-ascend-env.sh
set -u
BUILD="$ROOT/vllm-ascend/csrc/build"
if [[ ! -f "$BUILD/build.ninja" || ! -f build/vllm-native/CMakeCache.txt ]]; then
    printf 'Configure the pinned vLLM environment first; see docs/VLLM_ASCEND_ENVIRONMENT.md.\n' >&2
    exit 1
fi

# Keep existing direct kernels. CANN's incremental object merge cannot relink
# already merged bgmv objects; only the two host binding objects changed here.
# Force the binding TU because this direct Make target skips header dependency refresh.
make -C build/vllm-native -f CMakeFiles/vllm_ascend_C.dir/build.make \
    -W "$ROOT/vllm-ascend/csrc/torch_binding.cpp" \
    CMakeFiles/vllm_ascend_C.dir/build -j "$MAX_JOBS"
cp build/vllm-native/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so vllm-ascend/vllm_ascend/

cmake --build "$BUILD" --target op_host_aclnnInner -j "$MAX_JOBS"
OPS_PROTO_SEPARATE=1 OPS_ACLNN_GEN=1 OPS_PROJECT_NAME=aclnnInner \
OPS_PRODUCT_NAME='ascend910_93;' \
    "$ASCEND_HOME_PATH/tools/opbuild/op_build" \
    "$BUILD/libop_host_aclnnInner.so" "$BUILD/autogen/inner"
cmake --build "$BUILD" --target generate_transformer_adapt_py generate_ops_info generate_compile_cmd -j "$MAX_JOBS"

# The upstream copy/compile stamps have order-only source dependencies. Archive
# just this operator's generated files so stale signatures/binaries cannot pass.
BACKUP=$(mktemp -d "$ROOT/build/echo-prefetch-generated.XXXXXX")
BIN="$BUILD/binary/ascend910_93"
mkdir -p "$BIN/src/fused_lightning_indexer_manage"
cp -R vllm-ascend/csrc/attention/fused_lightning_indexer_manage/op_kernel \
    "$BIN/src/fused_lightning_indexer_manage/"
cp "$BUILD/impl/dynamic/fused_lightning_indexer_manage.py" \
    "$BIN/src/fused_lightning_indexer_manage/FusedLightningIndexerManage.py"
shopt -s nullglob
for generated in "$BIN"/gen/fused_lightning_indexer_manage_ascend910_93_*.done \
                 "$BIN/bin/fused_lightning_indexer_manage"; do
    if [[ -e "$generated" ]]; then
        mv "$generated" "$BACKUP/"
    fi
done
cmake --build "$BUILD" -j "$MAX_JOBS"
cmake --build "$BUILD" --target package -j "$MAX_JOBS"
bash "$BUILD/cann-ops-transformer-custom_linux-aarch64.run" --quiet \
    --install-path="$ROOT/vllm-ascend/vllm_ascend/_cann_ops_custom"
