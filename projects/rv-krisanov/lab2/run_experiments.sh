#!/bin/bash
set -e

# Create results directory
mkdir -p results

# Experiment 1: Baseline
uv run python ./scripts/build_index.py --config=./scripts/configs/config_baseline.yaml
uv run python ./scripts/load_to_vector_store.py --config=./scripts/configs/config_baseline.yaml
uv run python ./scripts/evaluate.py --config=./scripts/configs/config_baseline.yaml --output results/baseline.json

# Experiment 2: Large Chunks
uv run python ./scripts/build_index.py --config=./scripts/configs/config_large_chunks.yaml
uv run python ./scripts/load_to_vector_store.py --config=./scripts/configs/config_large_chunks.yaml
uv run python ./scripts/evaluate.py --config=./scripts/configs/config_large_chunks.yaml --output results/large_chunks.json

# Experiment 3: Better HNSW
uv run python ./scripts/build_index.py --config=./scripts/configs/config_better_hnsw.yaml
uv run python ./scripts/load_to_vector_store.py --config=./scripts/configs/config_better_hnsw.yaml
uv run python ./scripts/evaluate.py --config=./scripts/configs/config_better_hnsw.yaml --output results/better_hnsw.json

