#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VLLM_HOST="${VLLM_HOST:-${VLLM_BENCHMARK_HOST:-http://127.0.0.1:8000}}"
DATA_DIR="${VLLM_BENCHMARK_DATA_DIR:-${ROOT_DIR}/data}"
RESULTS_DIR="${VLLM_BENCHMARK_RESULTS_DIR:-${ROOT_DIR}/results}"
REQUEST_MODEL="${VLLM_REQUEST_MODEL:-${MODEL_DIR:-}}"

mkdir -p "$RESULTS_DIR"

# Function to measure E2E latency
measure_request() {
    local payload_file=$1
    local output_file=$2
    local round_num=$3
    
    echo "Testing round $round_num..."
    
    # Measure time with curl
    start_time=$(date +%s%N)
    
    payload_json=$(python3 - "$payload_file" "$REQUEST_MODEL" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
if sys.argv[2]:
    payload["model"] = sys.argv[2]
print(json.dumps(payload, ensure_ascii=False))
PY
)

    response=$(curl -s -X POST "$VLLM_HOST/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$payload_json")
    
    end_time=$(date +%s%N)
    e2e_latency=$(( (end_time - start_time) / 1000000 ))  # Convert to ms
    
    # Extract token counts from response
    prompt_tokens=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['prompt_tokens'])")
    completion_tokens=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['completion_tokens'])")
    total_tokens=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['total_tokens'])")
    
    # Calculate TTFT and TPOT (simplified - actual implementation needs more precise timing)
    ttft=$e2e_latency  # Simplified - first token time approx = total for this test
    tpot=$(( e2e_latency / completion_tokens ))
    
    # Save results
    cat >> "$output_file" << EOF
Round $round_num:
  E2E Latency: ${e2e_latency}ms
  TTFT: ${ttft}ms
  TPOT: ${tpot}ms/token
  Prompt Tokens: $prompt_tokens
  Completion Tokens: $completion_tokens
  Total Tokens: $total_tokens

EOF

    echo "  E2E: ${e2e_latency}ms, Tokens: $total_tokens"
}

# Main test execution
echo "Starting Performance Benchmark Test"
echo "====================================="
echo ""

# Warmup rounds (3 rounds)
echo "WARMUP PHASE (3 rounds)"
echo "-----------------------"
for i in 0 1 2; do
    payload_file="$DATA_DIR/round_$i/payload.json"
    if [ -f "$payload_file" ]; then
        measure_request "$payload_file" "$RESULTS_DIR/warmup_results.txt" $i
    fi
done
echo ""

# Test rounds (10 rounds)
echo "TEST PHASE (10 rounds)"
echo "----------------------"
for i in 3 4 5 6 7 8 9 10 11 12; do
    payload_file="$DATA_DIR/round_$i/payload.json"
    if [ -f "$payload_file" ]; then
        measure_request "$payload_file" "$RESULTS_DIR/test_results.txt" $i
    fi
done
echo ""

echo "====================================="
echo "Test Complete!"
echo "Results saved to: $RESULTS_DIR"
