#!/bin/bash
# run_profiler.sh - Run the NCCL profiler across available GPUs
# Usage: bash scripts/run_profiler.sh [NUM_GPUS]
#
# Examples:
#   bash scripts/run_profiler.sh        # auto-detect GPU count
#   bash scripts/run_profiler.sh 2      # use 2 GPUs

set -e

NUM_GPUS=${1:-$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)}

echo "========================================="
echo "  NCCL Profiler"
echo "  GPUs: $NUM_GPUS"
echo "========================================="

# Enable NCCL debug logging for topology info
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL

echo ""
echo "[1/2] Running NCCL profiler (quick test)..."
torchrun \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    -m src.nccl_profiler.profiler

echo ""
echo "[2/2] Running NCCL benchmark (full sweep)..."
torchrun \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    -m src.nccl_profiler.benchmark \
    --warmup 5 \
    --iters 20 \
    --output nccl_benchmark_results.json

echo ""
echo "========================================="
echo "  Profiling complete!"
echo "  Results: nccl_profile_results.json"
echo "           nccl_benchmark_results.json"
echo "========================================="
