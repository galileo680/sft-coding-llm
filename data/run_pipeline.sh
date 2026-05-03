#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

python data/download_data.py --output-dir data/raw

python data/filter_quality.py \
    --input data/raw/train.jsonl \
    --output data/processed/filtered.jsonl \
    --min-doc-words 10 \
    --min-code-lines 4 \
    --max-code-lines 80

python data/format_dataset.py \
    --input data/processed/filtered.jsonl \
    --output data/processed/formatted.jsonl

python data/split_dataset.py \
    --input data/processed/formatted.jsonl \
    --output-dir data/processed/splits \
    --max-samples 25000 \
    --val-ratio 0.05 \
    --test-ratio 0.05 \
    --seed 42

python data/dataset_statistics.py \
    --input data/processed/formatted.jsonl \
    --output-dir data/processed/stats
