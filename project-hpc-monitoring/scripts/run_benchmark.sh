#!/bin/bash
# run_benchmark.sh - Run NCCL benchmarks simulating real LLM training communication
# Usage: bash scripts/run_benchmark.sh [NUM_GPUS]
#
# This script runs the benchmark, starts the metrics exporter, and
# optionally runs the scheduling recommender.

set -e

NUM_GPUS=${1:-$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)}

echo "========================================="
echo "  NCCL Benchmark + Monitoring Pipeline"
echo "  GPUs: $NUM_GPUS"
echo "========================================="

# Step 1: Run benchmark
echo ""
echo "[1/3] Running NCCL benchmark..."
torchrun \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    -m src.nccl_profiler.benchmark \
    --warmup 10 \
    --iters 50 \
    --output nccl_benchmark_results.json

# Step 2: Start metrics exporter in background
echo ""
echo "[2/3] Starting NCCL metrics exporter on :9091..."
python3 -m src.nccl_profiler.export_metrics \
    --port 9091 \
    --results-file nccl_benchmark_results.json &
EXPORTER_PID=$!
echo "  Exporter PID: $EXPORTER_PID"

# Wait for exporter to ingest data
sleep 10

# Step 3: Run recommender
echo ""
echo "[3/3] Running scheduling recommender..."
python3 -m src.recommender.recommender \
    --prometheus-url http://localhost:9090 \
    --window 1h

# Cleanup
echo ""
echo "Stopping metrics exporter..."
kill $EXPORTER_PID 2>/dev/null || true

echo ""
echo "========================================="
echo "  Pipeline complete!"
echo "========================================="
