#!/usr/bin/env bash
set -euo pipefail

export DSV32W4="${DSV32W4:-/data1/dsv32awq}"

CURRENT_MMDD="$(date +%m%d)"
TASK_DATASET_DIR="${TASK_DATASET_DIR:-/workspace/infinibench_task}"
REQUEST_RATES_INPUT="${REQUEST_RATES:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,inf}"
REQUEST_RATES=()
NUM_PROMPTS="${NUM_PROMPTS:-318}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-0}"
NO_OFFLOAD_RES_DIR="${NO_OFFLOAD_RES_DIR:-infini-task/no-offload-${CURRENT_MMDD}}"
OFFLOAD_RES_DIR="${OFFLOAD_RES_DIR:-infini-task/offload-${CURRENT_MMDD}}"

usage() {
  cat <<'EOF'
Usage: e2e_test_pd_infini_task.sh [options]

Options:
  --request-rates VALUE
      Use VALUE as benchmark request rates for both runs.
      Accepts comma-separated values, e.g. 0.1,0.5,inf.
  --no-offload-res-dir VALUE
      Use VALUE as the no-offload benchmark result directory.
  --offload-res-dir VALUE
      Use VALUE as the offload benchmark result directory.
  -h, --help
      Show this help message.
EOF
}

parse_request_rates() {
  local raw_value="$1"
  local normalized_value

  normalized_value="${raw_value//,/ }"
  read -r -a REQUEST_RATES <<< "$normalized_value"

  if [[ ${#REQUEST_RATES[@]} -eq 0 ]]; then
    echo "Request rates cannot be empty" >&2
    exit 1
  fi
}

parse_request_rates "$REQUEST_RATES_INPUT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --request-rates)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage
        exit 1
      fi
      parse_request_rates "$2"
      shift 2
      ;;
    --no-offload-res-dir)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage
        exit 1
      fi
      NO_OFFLOAD_RES_DIR="$2"
      shift 2
      ;;
    --offload-res-dir)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage
        exit 1
      fi
      OFFLOAD_RES_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

wait_for_ready() {
  local name="$1"
  local log_file="$2"
  local ready_text="$3"

  until [[ -f "$log_file" ]] && grep -Fq "$ready_text" "$log_file"; do
    echo "Waiting for ${name} to be ready ..."
    sleep 5
  done
}

require_task_dir() {
  if [[ ! -d "$TASK_DATASET_DIR" ]]; then
    echo "Task dataset directory not found: $TASK_DATASET_DIR" >&2
    exit 1
  fi
}

bench_all_tasks() {
  local res_root="$1"
  local task_file task_name task_dir request_rate

  mkdir -p "$res_root"
  shopt -s nullglob
  for task_file in "$TASK_DATASET_DIR"/*.jsonl; do
    task_name="$(basename "$task_file" .jsonl)"
    task_dir="${res_root}/${task_name}"
    mkdir -p "$task_dir"

    for request_rate in "${REQUEST_RATES[@]}"; do
      python3 -m sglang.bench_serving \
        --backend sglang \
        --dataset-name infinitebench \
        --dataset-path "$TASK_DATASET_DIR" \
        --infinitebench-task "$task_name" \
        --sort-len \
        --num-prompts "$NUM_PROMPTS" \
        --warmup-requests "$WARMUP_REQUESTS" \
        --host localhost \
        --port 30000 \
        --model "$DSV32W4" \
        --request-rate "$request_rate" \
        --output-file "${task_dir}/req_rate_${request_rate}.jsonl" \
        > "${task_dir}/req_rate_${request_rate}.log"
    done
  done
  shopt -u nullglob
}

require_task_dir

FAKE_P_NODE=1 DS_DEBUG_LAYERS=1 python -m sglang.launch_server --model-path "$DSV32W4" --disaggregation-mode prefill --host localhost --port 1111 --tp 8 --dp 8 --enable-dp-attention --disaggregation-bootstrap-port 8998 --quantization moe_wna16 --mem-fraction-static 0.03 --chunked-prefill-size 16384 --disaggregation-transfer-backend fake --load-balance-method round_robin &> prefill_no_offload.log &
wait_for_ready "prefill(no-offload)" "prefill_no_offload.log" "The server is fired up and ready to roll!"

python -m sglang.launch_server --model-path "$DSV32W4" --disaggregation-mode decode --host localhost --port 2233 --tp 8 --dp 8 --enable-dp-attention --speculative-num-steps 0 --disable-radix-cache --disable-overlap-schedule --quantization moe_wna16 --mem-fraction-static 0.9 --prefill-round-robin-balance --cuda-graph-max-bs 64 --disaggregation-transfer-backend fake &> decode_no_offload.log &
wait_for_ready "decode(no-offload)" "decode_no_offload.log" "The server is fired up and ready to roll!"

python -m sglang_router.launch_router --pd-disaggregation --prefill http://127.0.0.1:1111 8998 --decode http://127.0.0.1:2233 --host 127.0.0.1 --port 30000 &> router_no_offload.log &
wait_for_ready "router(no-offload)" "router_no_offload.log" "Router ready | workers:"

bench_all_tasks "$NO_OFFLOAD_RES_DIR"

bash scripts/killall_sglang.sh
bash scripts/killall_sglang.sh

FAKE_P_NODE=1 DS_DEBUG_LAYERS=1 python -m sglang.launch_server --model-path "$DSV32W4" --disaggregation-mode prefill --host localhost --port 1111 --tp 8 --dp 8 --enable-dp-attention --disaggregation-bootstrap-port 8998 --quantization moe_wna16 --mem-fraction-static 0.03 --chunked-prefill-size 16384 --disaggregation-transfer-backend fake --load-balance-method round_robin &> prefill.log &
wait_for_ready "prefill" "prefill.log" "The server is fired up and ready to roll!"

NSA_DEV_CACHE_SIZE=260_000 NSA_KV_OFFLOAD=1 python -m sglang.launch_server --model-path "$DSV32W4" --disaggregation-mode decode --host localhost --port 2233 --tp 8 --dp 8 --enable-dp-attention --speculative-num-steps 0 --disable-radix-cache --disable-overlap-schedule --quantization moe_wna16 --prefill-round-robin-balance --cuda-graph-max-bs 48 --max-total-tokens 1_800_000 --disaggregation-transfer-backend fake &> decode.log &
wait_for_ready "decode" "decode.log" "The server is fired up and ready to roll!"

python -m sglang_router.launch_router --pd-disaggregation --prefill http://127.0.0.1:1111 8998 --decode http://127.0.0.1:2233 --host 127.0.0.1 --port 30000 &> router.log &
wait_for_ready "router" "router.log" "Router ready | workers:"

bench_all_tasks "$OFFLOAD_RES_DIR"

bash scripts/killall_sglang.sh
