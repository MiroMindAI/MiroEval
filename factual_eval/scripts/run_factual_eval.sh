#!/bin/bash

# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

# Run factual evaluation on Deep Research results
#
# Usage:
#   bash scripts/run_factual_eval.sh
#   bash scripts/run_factual_eval.sh --max-tasks 1          # test with 1 task
#   bash scripts/run_factual_eval.sh --config config/benchmark_factual-eval_mirothinker.yaml
#   bash scripts/run_factual_eval.sh --config ... --result-dir logs/factual-eval/prev_run  # resume

# Configuration (override via environment variables)
CONFIG=${CONFIG:-"config/benchmark_factual-eval_gpt5-mini.yaml"}
MAX_CONCURRENT=${MAX_CONCURRENT:-10}
MAX_CONCURRENT_CHUNKS=${MAX_CONCURRENT_CHUNKS:-10}

# Parse command line arguments
EXTRA_OVERRIDES=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --max-tasks)
            EXTRA_OVERRIDES+=("benchmark.execution.max_tasks=$2")
            shift 2
            ;;
        --max-concurrent)
            MAX_CONCURRENT="$2"
            shift 2
            ;;
        --max-concurrent-chunks)
            MAX_CONCURRENT_CHUNKS="$2"
            shift 2
            ;;
        --result-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        *)
            EXTRA_OVERRIDES+=("$1")
            shift
            ;;
    esac
done

# Set results directory with config name + timestamp
CONFIG_NAME=$(basename "$CONFIG" .yaml)
TIMESTAMP=$(date +%Y%m%d_%H%M)
RESULTS_DIR=${RESULTS_DIR:-"logs/factual-eval/${CONFIG_NAME}_${TIMESTAMP}"}

cleanup() {
    echo ""
    echo "Received interrupt signal, terminating..."
    pkill -TERM -f "run_factual_eval.py" 2>/dev/null
    sleep 2
    pkill -KILL -f "run_factual_eval.py" 2>/dev/null
    pkill -TERM -P $$ 2>/dev/null
    echo "All processes terminated."
    exit 130
}

trap cleanup SIGINT SIGTERM

echo "=========================================="
echo "Factual Evaluation"
echo "=========================================="
echo "Config:               $CONFIG"
echo "Output dir:           $RESULTS_DIR"
echo "Max concurrent tasks: $MAX_CONCURRENT"
echo "Max concurrent chunks: $MAX_CONCURRENT_CHUNKS"
echo "=========================================="

mkdir -p "$RESULTS_DIR"

uv run python miroflow/benchmark/run_factual_eval.py \
    --config-path "$CONFIG" \
    benchmark.execution.max_concurrent=$MAX_CONCURRENT \
    benchmark.execution.max_concurrent_chunks=$MAX_CONCURRENT_CHUNKS \
    output_dir="$RESULTS_DIR" \
    "${EXTRA_OVERRIDES[@]}" \
    2>&1 | tee "$RESULTS_DIR/run.log"

# Summary
echo ""
echo "=========================================="
echo "Factual evaluation completed!"
echo "Results: $RESULTS_DIR"
RESULT_COUNT=$(find "$RESULTS_DIR" -name "*.json" -not -name "factual_eval_summary.jsonl" 2>/dev/null | wc -l)
echo "Generated $RESULT_COUNT result file(s)"
echo "=========================================="
