#!/usr/bin/env bash
set -eo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/npu-env.sh"
exec "$@"
