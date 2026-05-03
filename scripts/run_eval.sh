#!/bin/bash
set -e

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: bash scripts/run_eval.sh <adapter_path_or_'baseline'> <output_dir>"
    echo "Examples:"
    echo "  bash scripts/run_eval.sh baseline results/baseline"
    echo "  bash scripts/run_eval.sh outputs/sft_base/final results/sft_base"
    exit 1
fi

ADAPTER="$1"
OUTPUT_DIR="$2"
MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

if [ "$ADAPTER" = "baseline" ]; then
    ADAPTER_FLAG=""
else
    ADAPTER_FLAG="--adapter-path $ADAPTER"
fi

python -m evaluation.generate_samples \
    --model-name "$MODEL" \
    $ADAPTER_FLAG \
    --test-path data/processed/splits/test.jsonl \
    --output-path "$OUTPUT_DIR/task_eval/generations.jsonl" \
    --max-samples 200 \
    --max-new-tokens 256

python -m evaluation.run_task_eval \
    --input "$OUTPUT_DIR/task_eval/generations.jsonl" \
    --output-dir "$OUTPUT_DIR/task_eval"

python -m evaluation.run_humaneval \
    --model-name "$MODEL" \
    $ADAPTER_FLAG \
    --output-dir "$OUTPUT_DIR/humaneval" \
    --n-samples 1 \
    --batch-size 1
