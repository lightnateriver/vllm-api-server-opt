#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VLLM_ASCEND_API_OPT_PHASE=0
exec "${BUNDLE_DIR}/start_phase_server.sh"
