#!/usr/bin/env bash
set -eo pipefail

ECHO_SERVE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export ECHO_NPU_DEVICE="${ECHO_NPU_DEVICE:-8,9,10,11,12,13,14,15}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-200}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"

exec "${ECHO_SERVE_ROOT}/scripts/npu-run.sh" python -m sglang.launch_server \
    --model-path "${ECHO_SERVE_ROOT}/build/models/DeepSeek-V3.2-AWQ" \
    --served-model-name deepseek-v3.2-awq \
    --trust-remote-code \
    --device npu --attention-backend ascend \
    --quantization awq --dtype bfloat16 \
    --tp-size 8 \
    --host 127.0.0.1 --port "${ECHO_V32_PORT:-30032}" \
    --mem-fraction-static 0.80 \
    --context-length 4096 --max-total-tokens 4096 \
    --max-running-requests 2 \
    --chunked-prefill-size -1 --max-prefill-tokens 4096 \
    --disable-radix-cache --disable-shared-experts-fusion \
    --cuda-graph-bs 1 2 \
    --decode-log-interval 1 --watchdog-timeout 9000
