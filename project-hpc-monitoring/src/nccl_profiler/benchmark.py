"""
NCCL Benchmark Suite

Runs a systematic sweep of collective operations across different tensor
sizes and reports throughput / latency. Simulates the communication patterns
seen in distributed LLM training and inference.

Usage:
    torchrun --nproc_per_node=2 -m src.nccl_profiler.benchmark
    torchrun --nproc_per_node=4 -m src.nccl_profiler.benchmark --warmup 10 --iters 50
"""

import argparse
import json
import time

import torch
import torch.distributed as dist


class NCCLBenchmark:
    """Systematic benchmark for NCCL collective operations."""

    OPERATIONS = ["allreduce", "broadcast", "allgather", "reduce_scatter"]

    # Sizes representative of real workloads:
    #   4 KB   ~ small gradient (single layer bias)
    #   256 KB ~ medium gradient
    #   4 MB   ~ large gradient (attention projection)
    #  32 MB   ~ full layer gradient
    # 256 MB   ~ aggregated model update
    DEFAULT_SIZES = [
        4 * 1024,          # 4 KB
        64 * 1024,         # 64 KB
        256 * 1024,        # 256 KB
        1024 * 1024,       # 1 MB
        4 * 1024 * 1024,   # 4 MB
        32 * 1024 * 1024,  # 32 MB
        256 * 1024 * 1024, # 256 MB
    ]

    def __init__(self, warmup_iters: int = 5, bench_iters: int = 20):
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        torch.cuda.set_device(self.device)
        self.warmup_iters = warmup_iters
        self.bench_iters = bench_iters
        self.results: list[dict] = []

    def _run_op(self, op_name: str, tensor: torch.Tensor):
        """Execute a single collective operation."""
        if op_name == "allreduce":
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        elif op_name == "broadcast":
            dist.broadcast(tensor, src=0)
        elif op_name == "allgather":
            out = [torch.zeros_like(tensor) for _ in range(self.world_size)]
            dist.all_gather(out, tensor)
        elif op_name == "reduce_scatter":
            chunk = tensor.shape[0] // self.world_size
            output = torch.zeros(chunk, dtype=tensor.dtype, device=self.device)
            input_list = list(tensor.chunk(self.world_size, dim=0))
            dist.reduce_scatter(output, input_list)

    def bench_operation(self, op_name: str, size_bytes: int) -> dict:
        """Benchmark one operation at one size."""
        num_elements = size_bytes // 4  # float32 = 4 bytes
        # Ensure divisible by world_size for reduce_scatter
        num_elements = (num_elements // self.world_size) * self.world_size
        tensor = torch.randn(num_elements, device=self.device)

        # Warmup
        for _ in range(self.warmup_iters):
            self._run_op(op_name, tensor.clone())
        torch.cuda.synchronize()

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(self.bench_iters):
            self._run_op(op_name, tensor.clone())
        torch.cuda.synchronize()
        total_s = time.perf_counter() - start

        avg_ms = (total_s / self.bench_iters) * 1000
        actual_bytes = num_elements * 4
        # Algorithm bandwidth: accounts for the NCCL algorithm's data movement
        # For AllReduce: 2 * (N-1)/N * data_size (ring algorithm)
        if op_name == "allreduce":
            algo_bytes = 2 * (self.world_size - 1) / self.world_size * actual_bytes
        elif op_name == "broadcast":
            algo_bytes = actual_bytes
        elif op_name == "allgather":
            algo_bytes = (self.world_size - 1) / self.world_size * actual_bytes
        elif op_name == "reduce_scatter":
            algo_bytes = (self.world_size - 1) / self.world_size * actual_bytes
        else:
            algo_bytes = actual_bytes

        bus_bw_gbps = (algo_bytes / 1e9) / (avg_ms / 1000) if avg_ms > 0 else 0

        return {
            "operation": op_name,
            "size_bytes": actual_bytes,
            "avg_latency_ms": round(avg_ms, 4),
            "bus_bandwidth_gbps": round(bus_bw_gbps, 2),
            "world_size": self.world_size,
            "iters": self.bench_iters,
        }

    def run_all(self, sizes: list[int] | None = None, operations: list[str] | None = None):
        """Run benchmark across all sizes and operations."""
        sizes = sizes or self.DEFAULT_SIZES
        operations = operations or self.OPERATIONS
        self.results = []

        for op in operations:
            for size in sizes:
                result = self.bench_operation(op, size)
                self.results.append(result)
                if self.rank == 0:
                    _print_result(result)

        if self.rank == 0:
            self._print_summary()

    def _print_summary(self):
        print(f"\n{'='*78}")
        print(f"  NCCL Benchmark Summary  |  GPUs: {self.world_size}  |  "
              f"Warmup: {self.warmup_iters}  |  Iters: {self.bench_iters}")
        print(f"{'='*78}")
        print(f"  {'Operation':<16} {'Size':>10} {'Latency(ms)':>14} {'BusBW(GB/s)':>14}")
        print(f"  {'-'*16} {'-'*10} {'-'*14} {'-'*14}")
        for r in self.results:
            size_str = _fmt(r["size_bytes"])
            print(f"  {r['operation']:<16} {size_str:>10} "
                  f"{r['avg_latency_ms']:>14.4f} {r['bus_bandwidth_gbps']:>14.2f}")
        print(f"{'='*78}\n")

    def export_json(self, path: str):
        if self.rank != 0:
            return
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Results exported to {path}")


def _fmt(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    elif n < 1024**2:
        return f"{n // 1024} KB"
    elif n < 1024**3:
        return f"{n // 1024**2} MB"
    return f"{n / 1024**3:.1f} GB"


def _print_result(r: dict):
    size_str = _fmt(r["size_bytes"])
    print(f"  [{r['operation']:<16}] {size_str:>10}  "
          f"lat={r['avg_latency_ms']:.4f} ms  bw={r['bus_bandwidth_gbps']:.2f} GB/s")


def main():
    parser = argparse.ArgumentParser(description="NCCL Benchmark")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--output", type=str, default="nccl_benchmark_results.json")
    args = parser.parse_args()

    bench = NCCLBenchmark(warmup_iters=args.warmup, bench_iters=args.iters)
    bench.run_all()
    bench.export_json(args.output)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
