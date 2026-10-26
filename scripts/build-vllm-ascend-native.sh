#!/usr/bin/env bash
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT/scripts/vllm-ascend-env.sh"
cd "$ROOT"

cmake -S vllm-ascend -B build/vllm-native \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$ROOT/vllm-ascend/vllm_ascend" \
  -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" \
  -DPYTHON_EXECUTABLE="$(command -v python)" \
  -DPYTHON_INCLUDE_PATH="$(python -c 'import sysconfig; print(sysconfig.get_path("include"))')" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DTORCH_NPU_PATH="$VIRTUAL_ENV/lib/python3.11/site-packages/torch_npu" \
  -DSOC_VERSION="$SOC_VERSION"
cmake --build build/vllm-native -j "$MAX_JOBS"
cmake --install build/vllm-native
(
  cd vllm-ascend
  bash csrc/build_aclnn.sh "$PWD" "$SOC_VERSION"
)

# Native binaries above are already installed into the editable source tree.
COMPILE_CUSTOM_KERNELS=0 \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+pr17392.fcc49fe \
  python -m pip install --no-build-isolation --no-deps -e ./vllm-ascend
