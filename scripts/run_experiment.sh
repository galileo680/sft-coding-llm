#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: bash scripts/run_experiment.sh configs/sft_base.yaml"
    exit 1
fi

CONFIG="$1"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

python -m training.train_sft --config "$CONFIG"
