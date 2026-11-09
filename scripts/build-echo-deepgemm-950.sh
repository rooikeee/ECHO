#!/usr/bin/env bash
# Run only in a prepared Ascend 950 / CANN 9.20 / torch_npu environment.
set -euo pipefail
ECHO_DG_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ECHO_DG_PYTHON=${ECHO_DG_PYTHON:-python3}
: "${ASCEND_HOME_PATH:?Activate the 950 CANN 9.20 environment before building}"
"$ECHO_DG_PYTHON" - <<'PY'
import torch, torch_npu
name = torch.npu.get_device_name()
if "950" not in name:
    raise SystemExit(f"Requires Ascend 950; found {name}. Do not override architecture to build/run on 910.")
print("Building for", name)
PY
cd "$ECHO_DG_ROOT/deepgemm-ascend"
DG_USE_LOCAL_VERSION=0 "$ECHO_DG_PYTHON" -m pip install --no-build-isolation --no-deps .
cd "$ECHO_DG_ROOT"
"$ECHO_DG_PYTHON" - <<'PY'
import deep_gemm
from deep_gemm.echo import EchoPrefetchBuffers
assert hasattr(deep_gemm._C, "echo_paged_mqa_prefetch_out"), "Installed extension lacks ECHO entry"
assert hasattr(deep_gemm._C, "echo_clear_prefetch_mappings"), "Rebuild: installed extension lacks tagged-map cleanup"
print("Installed", deep_gemm.__file__)
PY
