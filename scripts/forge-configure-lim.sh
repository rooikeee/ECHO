#!/usr/bin/env bash
# Configure only the two existing fused operators in this checkout.
set -eo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
source scripts/vllm-ascend-env.sh
set -u
export CMAKE_BUILD_PARALLEL_LEVEL=4 MAX_JOBS=4
BUILD="$ROOT/vllm-ascend/csrc/build"
mkdir -p "$ROOT/build/forge-lim"
# This pinned build.sh does not convert comma-separated names into CMake lists.
# Its first build may fail because it assumes nonempty public/excluded INIs;
# finish generation explicitly below for this inner-only pair.
if ! (cd vllm-ascend/csrc && bash build.sh --pkg \
    '--ops=fused_lightning_indexer_manage;fused_scatter_copy_sparse_flash_attention' \
    --soc=ascend910_93 -j4); then
    [[ -f "$BUILD/build.ninja" ]] || exit 1
fi
cmake --build "$BUILD" --target op_host_aclnnInner -j4
OPS_PROTO_SEPARATE=1 OPS_ACLNN_GEN=1 OPS_PROJECT_NAME=aclnnInner \
OPS_PRODUCT_NAME='ascend910_93;' "$ASCEND_HOME_PATH/tools/opbuild/op_build" \
    "$BUILD/libop_host_aclnnInner.so" "$BUILD/autogen/inner"
mkdir -p "$BUILD/autogen/exc"
touch "$BUILD/autogen/aic-ascend910_93-ops-info.ini" \
    "$BUILD/autogen/exc/aic-ascend910_93-ops-info.ini"
cmake -S vllm-ascend/csrc -B "$BUILD"
# Work around empty categories only in the generated Ninja command, not source.
python - "$BUILD" <<'PYCODE'
from pathlib import Path
import sys
root = Path(sys.argv[1])
path = root / "build.ninja"
lines = []
for line in path.read_text().splitlines(keepends=True):
    if line.lstrip().startswith("COMMAND =") and "/parse_ini_to_json.py " in line:
        for name in ("autogen/aic-ascend910_93-ops-info.ini",
                     "autogen/exc/aic-ascend910_93-ops-info.ini"):
            if not (root / name).read_text().strip():
                line = line.replace(" " + str(root / name), "")
    lines.append(line)
path.write_text("".join(lines))
PYCODE
cmake --build "$BUILD" -j4
cmake --build "$BUILD" --target package -j4
# Installation is performed by forge-build-lim.sh after source hash validation.
