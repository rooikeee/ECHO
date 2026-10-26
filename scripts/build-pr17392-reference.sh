#!/usr/bin/env bash
# Build an isolated, unoptimized PR reference; no package registration changes.
set -eo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT/scripts/vllm-ascend-env.sh"
set -u
SOURCE="$ROOT/build/lim-copysfa-optimization/versions/pr"
BUILD="$ROOT/build/lim-copysfa-optimization/pr-native"
test -f "$SOURCE/CMakeLists.txt"
cp -a --no-clobber "$ROOT/vllm-ascend/csrc/third_party/catlass" "$SOURCE/csrc/third_party/"
cmake -S "$SOURCE" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$SOURCE/vllm_ascend" \
  -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" \
  -DPYTHON_EXECUTABLE="$(command -v python)" \
  -DPYTHON_INCLUDE_PATH="$(python -c 'import sysconfig; print(sysconfig.get_path("include"))')" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DTORCH_NPU_PATH="$VIRTUAL_ENV/lib/python3.11/site-packages/torch_npu" \
  -DSOC_VERSION="$SOC_VERSION"
cmake --build "$BUILD" -j "$MAX_JOBS"
cmake --install "$BUILD"
# Build in the isolated source; retain the upstream-selected operator list.
# Skip only the archive-inapplicable git safe.directory/submodule setup.
export CPATH="$SOURCE/csrc/third_party/catlass/include${CPATH:+:$CPATH}"
chmod u+w "$SOURCE/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/scripts" 2>/dev/null || true
bash <(sed '/git config --global --add safe.directory/d' "$SOURCE/csrc/build_aclnn.sh") \
  "$SOURCE" "$SOC_VERSION"
