#!/usr/bin/env bash
# Rebuild the two fused operators in this isolated checkout only.
set -eo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
source scripts/vllm-ascend-env.sh
set -u
export CMAKE_BUILD_PARALLEL_LEVEL=4
export MAX_JOBS=4
BUILD="$ROOT/vllm-ascend/csrc/build"
BIN="$BUILD/binary/ascend910_93"
[[ -f "$BUILD/build.ninja" ]] || {
    printf 'Configure the isolated CANN build first; see docs/FORGE_LIM_910.md.\n' >&2
    exit 1
}
BACKUP=$(mktemp -d "$ROOT/build/forge-lim/native-previous.XXXXXX")
mkdir -p "$BIN/src/fused_lightning_indexer_manage"
cp -R vllm-ascend/csrc/attention/fused_lightning_indexer_manage/op_kernel \
    "$BIN/src/fused_lightning_indexer_manage/"
shopt -s nullglob
for path in "$BIN/gen/fused_lightning_indexer_manage_ascend910_93_"*.done "$BIN/bin/fused_lightning_indexer_manage"; do
    if [[ -e "$path" ]]; then mv "$path" "$BACKUP/"; fi
done
cmake --build "$BUILD" --target fused_lightning_indexer_manage -j4
cmake --build "$BUILD" --target package -j4
INSTALL="$ROOT/vllm-ascend/vllm_ascend/_cann_ops_custom"
# Preserve the current package until the replacement installer is available.
if [[ -d "$INSTALL" ]]; then mv "$INSTALL" "$BACKUP/opp"; fi
if ! bash "$BUILD/cann-ops-transformer-custom_linux-aarch64.run" --quiet --install-path="$INSTALL"; then
    if [[ -d "$INSTALL" ]]; then mv "$INSTALL" "$BACKUP/failed-install"; fi
    if [[ -d "$BACKUP/opp" ]]; then mv "$BACKUP/opp" "$INSTALL"; fi
    exit 1
fi

# The installer recreates this ignored package directory; preserve its tracked placeholder.
git show HEAD:vllm-ascend/vllm_ascend/_cann_ops_custom/.gitkeep > "$INSTALL/.gitkeep"
