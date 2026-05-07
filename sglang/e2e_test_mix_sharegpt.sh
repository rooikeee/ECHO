#!/usr/bin/env bash
set -euo pipefail

bash scripts/killall_sglang.sh

export DSV32W4="${DSV32W4:-/data1/dsv32awq}"

CURRENT_MMDD="$(date +%m%d)"
REQUEST_RATES_INPUT="${REQUEST_RATES:-inf,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1}"
REQUEST_RATES=()
NUM_PROMPTS="${NUM_PROMPTS:-100}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-0}"
NO_OFFLOAD_RES_DIR="${NO_OFFLOAD_RES_DIR:-sharegpt-100/no-offload-${CURRENT_MMDD}}"
OFFLOAD_RES_DIR="${OFFLOAD_RES_DIR:-sharegpt-100/offload-${CURRENT_MMDD}}"
OFFLOAD_NSA_DEV_CACHE_SIZE="${OFFLOAD_NSA_DEV_CACHE_SIZE:-200_000}"
OFFLOAD_MAX_TOTAL_TOKENS="${OFFLOAD_MAX_TOTAL_TOKENS:-1000_000}"

usage() {
  cat <<'EOF'
Usage: e2e_test_mix_sharegpt.sh [options]

Options:
  --request-rates VALUE
      Use VALUE as benchmark request rates for both runs.
      Accepts comma-separated values, e.g. inf,0.9,0.5,0.1.
  --num-prompts VALUE
      Use VALUE as the number of ShareGPT prompts for each benchmark.
  --warmup-requests VALUE
      Use VALUE as the warmup request count for each benchmark.
  --no-offload-res-dir VALUE
      Use VALUE as the no-offload benchmark result directory.
  --offload-res-dir VALUE
      Use VALUE as the offload benchmark result directory.
  --offload-nsa-dev-cache-size VALUE
      Pass VALUE to the offload run as NSA_DEV_CACHE_SIZE.
  --offload-max-total-tokens VALUE
      Pass VALUE to the offload run as --max-total-tokens.
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
    --num-prompts)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage
        exit 1
      fi
      NUM_PROMPTS="$2"
      shift 2
      ;;
    --warmup-requests)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage
        exit 1
      fi
      WARMUP_REQUESTS="$2"
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
    --offload-nsa-dev-cache-size)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage
        exit 1
      fi
      OFFLOAD_NSA_DEV_CACHE_SIZE="$2"
      shift 2
      ;;
    --offload-max-total-tokens)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage
        exit 1
      fi
      OFFLOAD_MAX_TOTAL_TOKENS="$2"
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

bench_sharegpt() {
  local res_dir="$1"
  local request_rate

  export RES_DIR="$res_dir"
  mkdir -p "$RES_DIR"
  for request_rate in "${REQUEST_RATES[@]}"; do
    python3 -m sglang.bench_serving \
      --backend sglang \
      --dataset-name sharegpt \
      --num-prompts "$NUM_PROMPTS" \
      --warmup-requests "$WARMUP_REQUESTS" \
      --host localhost \
      --port 30000 \
      --model "$DSV32W4" \
      --request-rate "$request_rate" \
      --output-file "${RES_DIR}/req_rate_${request_rate}.jsonl" \
      > "${RES_DIR}/req_rate_${request_rate}.log"
  done
}

python -m sglang.launch_server \
  --model "$DSV32W4" \
  --host localhost \
  --port 30000 \
  --mem-fraction-static 0.8 \
  --speculative-num-steps 0 \
  --disable-radix-cache \
  --disable-overlap-schedule \
  --chunked-prefill-size -1 \
  --cuda-graph-max-bs 32 \
  --tp 8 \
  --dp 8 \
  --enable-dp-attention \
  --quantization moe_wna16 \
  &> mix_sharegpt_no_offload.log &
wait_for_ready "mix sharegpt(no-offload)" "mix_sharegpt_no_offload.log" "The server is fired up and ready to roll!"

bench_sharegpt "$NO_OFFLOAD_RES_DIR"

bash scripts/killall_sglang.sh
bash scripts/killall_sglang.sh

NSA_DEV_CACHE_SIZE="$OFFLOAD_NSA_DEV_CACHE_SIZE" \
NSA_KV_OFFLOAD=1 \
python -m sglang.launch_server \
  --model "$DSV32W4" \
  --host localhost \
  --port 30000 \
  --mem-fraction-static 0.8 \
  --speculative-num-steps 0 \
  --disable-radix-cache \
  --disable-overlap-schedule \
  --chunked-prefill-size -1 \
  --cuda-graph-max-bs 32 \
  --tp 8 \
  --dp 8 \
  --enable-dp-attention \
  --quantization moe_wna16 \
  --max-total-tokens "$OFFLOAD_MAX_TOTAL_TOKENS" \
  &> mix_sharegpt_offload.log &
wait_for_ready "mix sharegpt(offload)" "mix_sharegpt_offload.log" "The server is fired up and ready to roll!"

bench_sharegpt "$OFFLOAD_RES_DIR"

bash scripts/killall_sglang.sh
