#!/usr/bin/env bash
set -eo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/vllm-ascend-env.sh"
exec "$@"
