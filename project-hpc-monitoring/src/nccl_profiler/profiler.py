"""
NCCL Profiling Wrappers

Wraps PyTorch distributed (NCCL backend) collective operations and records
wall-clock timings. Uses torch.cuda.synchronize() before and after each
call to get accurate measurements, since NCCL operations are asynchronous.

Usage:
    # Launch with torchrun:
    # torchrun --nproc_per_node=2 -m src.nccl_profiler.profiler

    profiler = NCCLProfiler()
    tensor = torch.randn(1024, device=profiler.device)
    result = profiler.allreduce(tensor)
    profiler.print_summary()
"""

import json
import time
from dataclasses import dataclass, field

import torch
import torch.distributed as dist


@dataclass
class ProfileRecord:
    operation: str
    data_size_bytes: int
    time_ms: float
    rank: int


class NCCLProfiler:
    """Profiles NCCL collective operations via PyTorch distributed."""

    def __init__(self):
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        torch.cuda.set_device(self.device)
        self.records: list[ProfileRecord] = []

    def _sync_and_time(self, fn):
        """Run fn between two cuda synchronization barriers and return elapsed ms."""
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        return result, elapsed_ms

    def _record(self, op: str, tensor: torch.Tensor, time_ms: float):
        self.records.append(ProfileRecord(
            operation=op,
            data_size_bytes=tensor.nelement() * tensor.element_size(),
            time_ms=time_ms,
            rank=self.rank,
        ))

    # ---- Collective operations ----

    def allreduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """Profile NCCL AllReduce: every GPU contributes, all receive the sum."""
        _, elapsed = self._sync_and_time(
            lambda: dist.all_reduce(tensor, op=op)
        )
        self._record("AllReduce", tensor, elapsed)
        return tensor

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Profile NCCL Broadcast: src GPU sends data to all others."""
        _, elapsed = self._sync_and_time(
            lambda: dist.broadcast(tensor, src=src)
        )
        self._record("Broadcast", tensor, elapsed)
        return tensor

    def allgather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Profile NCCL AllGather: collect shards from every GPU into a full tensor."""
        gather_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        _, elapsed = self._sync_and_time(
            lambda: dist.all_gather(gather_list, tensor)
        )
        self._record("AllGather", tensor, elapsed)
        return torch.cat(gather_list, dim=0)

    def reduce_scatter(self, tensor: torch.Tensor) -> torch.Tensor:
        """Profile NCCL ReduceScatter: reduce then scatter result shards."""
        chunk_size = tensor.shape[0] // self.world_size
        output = torch.zeros(chunk_size, *tensor.shape[1:],
                             dtype=tensor.dtype, device=tensor.device)
        input_list = list(tensor.chunk(self.world_size, dim=0))
        _, elapsed = self._sync_and_time(
            lambda: dist.reduce_scatter(output, input_list)
        )
        self._record("ReduceScatter", tensor, elapsed)
        return output

    # ---- Reporting ----

    def print_summary(self):
        """Print a summary table of all recorded operations (rank 0 only)."""
        if self.rank != 0:
            return
        print(f"\n{'='*70}")
        print(f"  NCCL Profile Summary  |  World Size: {self.world_size}")
        print(f"{'='*70}")
        print(f"  {'Operation':<18} {'Size':>12} {'Time (ms)':>12} {'BW (GB/s)':>12}")
        print(f"  {'-'*18} {'-'*12} {'-'*12} {'-'*12}")
        for r in self.records:
            size_str = _format_bytes(r.data_size_bytes)
            bw = (r.data_size_bytes / 1e9) / (r.time_ms / 1000) if r.time_ms > 0 else 0
            print(f"  {r.operation:<18} {size_str:>12} {r.time_ms:>12.3f} {bw:>12.2f}")
        print(f"{'='*70}\n")

    def export_json(self, path: str):
        """Export all records to a JSON file."""
        if self.rank != 0:
            return
        data = [
            {
                "operation": r.operation,
                "data_size_bytes": r.data_size_bytes,
                "time_ms": r.time_ms,
                "rank": r.rank,
            }
            for r in self.records
        ]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Profile exported to {path}")

    def reset(self):
        self.records.clear()


def _format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    elif n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    elif n < 1024 ** 3:
        return f"{n / 1024**2:.1f} MB"
    else:
        return f"{n / 1024**3:.2f} GB"


# ---- CLI entrypoint ----

def main():
    """Quick self-test: run all collective operations and print results."""
    profiler = NCCLProfiler()

    sizes = [1024, 256 * 1024, 1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024]

    for size in sizes:
        t = torch.randn(size, device=profiler.device)
        profiler.allreduce(t)

    for size in sizes:
        t = torch.randn(size, device=profiler.device)
        profiler.broadcast(t, src=0)

    for size in sizes:
        t = torch.randn(size, device=profiler.device)
        profiler.allgather(t)

    profiler.print_summary()
    profiler.export_json("nccl_profile_results.json")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
