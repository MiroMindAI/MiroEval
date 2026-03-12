#!/bin/bash

# Test a single multimodal factual evaluation task.
#
# Usage:
#   ./scripts/test_single_multimodal_factual_eval.sh                          # default: mirothinker_108
#   ./scripts/test_single_multimodal_factual_eval.sh --file data/factual-eval/mirothinker-multimodal-demo/mirothinker_132.json
#   ./scripts/test_single_multimodal_factual_eval.sh --task-index 0
#   ./scripts/test_single_multimodal_factual_eval.sh --config config/benchmark_factual-eval_mirothinker-multimodal-demo.yaml --file data.json

set -e

# Default configuration
CONFIG_PATH="config/benchmark_factual-eval_mirothinker-multimodal-demo.yaml"
DEFAULT_FILE="data/factual-eval/mirothinker-multimodal-demo/mirothinker_108.json"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo "=================================================="
echo "Multimodal Factual Eval - Single Task Test"
echo "=================================================="

# Build command arguments
CMD_ARGS=()

# Parse arguments
if [ $# -eq 0 ]; then
    # No arguments: use default file
    echo -e "${CYAN}No arguments provided, using default task: ${DEFAULT_FILE}${NC}"
    CMD_ARGS+=(--file "$DEFAULT_FILE")
else
    while [[ $# -gt 0 ]]; do
        case $1 in
            --config|--config-path)
                CONFIG_PATH="$2"
                shift 2
                ;;
            --file)
                CMD_ARGS+=(--file "$2")
                shift 2
                ;;
            --task-index)
                CMD_ARGS+=(--task-index "$2")
                shift 2
                ;;
            --output-dir)
                CMD_ARGS+=(--output-dir "$2")
                shift 2
                ;;
            -h|--help)
                echo ""
                echo "Usage:"
                echo "  $0                                    # Test default task (mirothinker_108)"
                echo "  $0 --file <path>                      # Test specific JSON file"
                echo "  $0 --task-index <n>                   # Test task by index (0-based)"
                echo ""
                echo "Options:"
                echo "  --config <path>       Config file (default: $CONFIG_PATH)"
                echo "  --output-dir <path>   Output directory"
                echo ""
                echo "Available test files:"
                ls data/factual-eval/mirothinker-multimodal-demo/*.json 2>/dev/null | sed 's/^/  /'
                exit 0
                ;;
            *)
                echo -e "${RED}Unknown option: $1${NC}"
                exit 1
                ;;
        esac
    done
fi

# Display configuration
echo -e "${YELLOW}Config:${NC}  $CONFIG_PATH"

# Check config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo -e "${RED}Error: Config not found: $CONFIG_PATH${NC}"
    exit 1
fi

# Show which task file will be used
for i in "${!CMD_ARGS[@]}"; do
    if [ "${CMD_ARGS[$i]}" = "--file" ]; then
        FILE_PATH="${CMD_ARGS[$((i+1))]}"
        echo -e "${YELLOW}File:${NC}    $FILE_PATH"
        if [ ! -f "$FILE_PATH" ]; then
            echo -e "${RED}Error: Data file not found: $FILE_PATH${NC}"
            exit 1
        fi
        # Show attachment info
        ATTACHMENTS=$(python -c "
import json
with open('$FILE_PATH') as f:
    d = json.load(f)
fps = d.get('file_paths', d.get('file_path', []))
if isinstance(fps, str): fps = [fps]
for fp in (fps or []):
    print(fp)
" 2>/dev/null)
        if [ -n "$ATTACHMENTS" ]; then
            echo -e "${YELLOW}Attachments:${NC}"
            echo "$ATTACHMENTS" | while read -r att; do
                if [ -f "$att" ]; then
                    SIZE=$(du -h "$att" | cut -f1)
                    echo -e "  ${GREEN}✓${NC} $att ($SIZE)"
                else
                    echo -e "  ${RED}✗${NC} $att (NOT FOUND)"
                fi
            done
        fi
    fi
done

echo ""
echo -e "${GREEN}Running factual eval...${NC}"
echo ""

uv run python scripts/run_single_factual_eval_task.py \
    --config "$CONFIG_PATH" \
    "${CMD_ARGS[@]}"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Test completed successfully!${NC}"
else
    echo -e "${RED}✗ Test failed with exit code $EXIT_CODE${NC}"
fi

exit $EXIT_CODE
