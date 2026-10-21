#!/usr/bin/env bash

ECHO_NPU_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "${ECHO_CANN_ENV:-${ECHO_NPU_ROOT}/build/cann/ascend-toolkit/set_env.sh}" || return 1
source "${ECHO_VENV:-${ECHO_NPU_ROOT}/.venv-npu}/bin/activate" || return 1
# The pinned PyTorch 2.8 ARM64 wheel uses the C++11 ABI.
source "${ECHO_ATB_ENV:-${ECHO_NPU_ROOT}/build/cann/nnal/atb/set_env.sh}" --cxx_abi=1 || return 1

export ASCEND_RT_VISIBLE_DEVICES="${ECHO_NPU_DEVICE:-${ASCEND_RT_VISIBLE_DEVICES:-14}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_HOME="${ECHO_NPU_ROOT}/build/npu-cache/huggingface"
export TORCH_HOME="${ECHO_NPU_ROOT}/build/npu-cache/torch"
export TRITON_CACHE_DIR="${ECHO_NPU_ROOT}/build/npu-cache/triton"
export XDG_CACHE_HOME="${ECHO_NPU_ROOT}/build/npu-cache"
export ASCEND_CACHE_PATH="${ECHO_NPU_ROOT}/build/npu-cache/ascend"
export ASCEND_WORK_PATH="${ECHO_NPU_ROOT}/build/npu-cache/ascend-work"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$ASCEND_CACHE_PATH" "$ASCEND_WORK_PATH"
