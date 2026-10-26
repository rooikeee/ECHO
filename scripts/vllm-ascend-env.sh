#!/usr/bin/env bash

ECHO_VLLM_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "${ECHO_VLLM_CANN_ENV:-${ECHO_VLLM_ROOT}/build/cann-vllm/cann/set_env.sh}" || return 1
if [[ -f "${ECHO_VLLM_ROOT}/build/cann-vllm/nnal/atb/set_env.sh" ]]; then
    source "${ECHO_VLLM_ROOT}/build/cann-vllm/nnal/atb/set_env.sh" || return 1
fi
source "${ECHO_VLLM_VENV:-${ECHO_VLLM_ROOT}/.venv-vllm-ascend}/bin/activate" || return 1
export PATH="${ECHO_VLLM_ROOT}/build/vllm-tools/usr/bin:${PATH}"
export PYTHONPATH="${ECHO_VLLM_ROOT}/vllm:${ECHO_VLLM_ROOT}/vllm-ascend${PYTHONPATH:+:${PYTHONPATH}}"

export ASCEND_RT_VISIBLE_DEVICES="${ECHO_NPU_DEVICE:-${ASCEND_RT_VISIBLE_DEVICES:-0}}"
export SOC_VERSION="${SOC_VERSION:-ascend910_9391}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MAX_JOBS="${MAX_JOBS:-16}"
export HF_HOME="${ECHO_VLLM_ROOT}/build/vllm-cache/huggingface"
export TRITON_CACHE_DIR="${ECHO_VLLM_ROOT}/build/vllm-cache/triton"
export ASCEND_CACHE_PATH="${ECHO_VLLM_ROOT}/build/vllm-cache/ascend"
export ASCEND_WORK_PATH="${ECHO_VLLM_ROOT}/build/vllm-cache/ascend-work"
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$ASCEND_CACHE_PATH" "$ASCEND_WORK_PATH"
