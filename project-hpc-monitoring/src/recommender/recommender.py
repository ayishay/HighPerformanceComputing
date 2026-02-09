"""
GPU Scheduling Recommender

Queries Prometheus for GPU utilization, memory, temperature, and NCCL
communication metrics, then produces actionable recommendations for
improving cluster resource usage.

Usage:
    python -m src.recommender.recommender
    python -m src.recommender.recommender --prometheus-url http://prometheus:9090 --window 2h
"""

import argparse
from dataclasses import dataclass

from .prometheus_client import PrometheusClient, MetricResult


@dataclass
class Recommendation:
    gpu_id: str
    severity: str   # "info", "warning", "critical"
    category: str   # "utilization", "memory", "temperature", "communication"
    message: str


class SchedulingRecommender:
    """Analyzes GPU cluster metrics and suggests resource allocation improvements."""

    # Thresholds (configurable)
    GPU_UTIL_LOW = 30.0       # below this -> underutilized
    GPU_UTIL_HIGH = 95.0      # above this -> saturated
    GPU_TEMP_WARN = 80.0      # Celsius
    GPU_TEMP_CRITICAL = 90.0
    MEM_UTIL_HIGH = 90.0      # % of total memory used
    NCCL_LATENCY_WARN = 10.0  # ms - high latency for AllReduce

    def __init__(self, prometheus_url: str = "http://localhost:9090", window: str = "1h"):
        self.client = PrometheusClient(prometheus_url)
        self.window = window

    def analyze(self) -> list[Recommendation]:
        """Run all analyses and return a list of recommendations."""
        recommendations = []
        recommendations.extend(self._analyze_utilization())
        recommendations.extend(self._analyze_memory())
        recommendations.extend(self._analyze_temperature())
        recommendations.extend(self._analyze_nccl())
        return recommendations

    def _analyze_utilization(self) -> list[Recommendation]:
        recs = []
        try:
            results = self.client.get_gpu_utilization(self.window)
        except Exception as e:
            return [Recommendation("N/A", "warning", "utilization",
                                   f"Could not query GPU utilization: {e}")]

        for r in results:
            gpu_id = r.labels.get("gpu", r.labels.get("GPU_I_ID", "unknown"))
            util = r.value

            if util < self.GPU_UTIL_LOW:
                recs.append(Recommendation(
                    gpu_id, "warning", "utilization",
                    f"GPU {gpu_id} avg utilization is {util:.1f}% (< {self.GPU_UTIL_LOW}%). "
                    f"Consider consolidating workloads onto fewer GPUs, or check if "
                    f"the job is I/O-bound (data loading bottleneck)."
                ))
            elif util > self.GPU_UTIL_HIGH:
                recs.append(Recommendation(
                    gpu_id, "info", "utilization",
                    f"GPU {gpu_id} avg utilization is {util:.1f}% (> {self.GPU_UTIL_HIGH}%). "
                    f"GPU is near saturation. Consider distributing across more GPUs "
                    f"or enabling pipeline parallelism."
                ))
            else:
                recs.append(Recommendation(
                    gpu_id, "info", "utilization",
                    f"GPU {gpu_id} avg utilization is {util:.1f}% - healthy range."
                ))
        return recs

    def _analyze_memory(self) -> list[Recommendation]:
        recs = []
        try:
            used = self.client.get_gpu_memory_used()
            free = self.client.get_gpu_memory_free()
        except Exception as e:
            return [Recommendation("N/A", "warning", "memory",
                                   f"Could not query GPU memory: {e}")]

        used_map = {r.labels.get("gpu", "?"): r.value for r in used}
        free_map = {r.labels.get("gpu", "?"): r.value for r in free}

        for gpu_id in used_map:
            u = used_map[gpu_id]
            f = free_map.get(gpu_id, 0)
            total = u + f
            if total == 0:
                continue
            pct = (u / total) * 100

            if pct > self.MEM_UTIL_HIGH:
                recs.append(Recommendation(
                    gpu_id, "warning", "memory",
                    f"GPU {gpu_id} memory usage is {pct:.1f}% ({u:.0f}/{total:.0f} MiB). "
                    f"Risk of OOM. Consider enabling gradient checkpointing, reducing "
                    f"batch size, or using DeepSpeed ZeRO to shard optimizer states."
                ))
            elif pct > 70:
                recs.append(Recommendation(
                    gpu_id, "info", "memory",
                    f"GPU {gpu_id} memory usage is {pct:.1f}% ({u:.0f}/{total:.0f} MiB)."
                ))
        return recs

    def _analyze_temperature(self) -> list[Recommendation]:
        recs = []
        try:
            results = self.client.get_gpu_temperature()
        except Exception as e:
            return [Recommendation("N/A", "warning", "temperature",
                                   f"Could not query GPU temperature: {e}")]

        for r in results:
            gpu_id = r.labels.get("gpu", "unknown")
            temp = r.value

            if temp > self.GPU_TEMP_CRITICAL:
                recs.append(Recommendation(
                    gpu_id, "critical", "temperature",
                    f"GPU {gpu_id} temperature is {temp:.0f}°C (> {self.GPU_TEMP_CRITICAL}°C). "
                    f"CRITICAL: Risk of thermal throttling or hardware damage. "
                    f"Check cooling system and consider reducing workload."
                ))
            elif temp > self.GPU_TEMP_WARN:
                recs.append(Recommendation(
                    gpu_id, "warning", "temperature",
                    f"GPU {gpu_id} temperature is {temp:.0f}°C (> {self.GPU_TEMP_WARN}°C). "
                    f"Approaching thermal limits. Monitor closely."
                ))
        return recs

    def _analyze_nccl(self) -> list[Recommendation]:
        recs = []
        for op in ["AllReduce", "AllGather", "Broadcast"]:
            try:
                results = self.client.get_nccl_latency(op, self.window)
            except Exception:
                continue

            for r in results:
                latency = r.value
                if latency > self.NCCL_LATENCY_WARN:
                    recs.append(Recommendation(
                        "cluster", "warning", "communication",
                        f"NCCL {op} avg latency is {latency:.2f} ms "
                        f"(> {self.NCCL_LATENCY_WARN} ms). "
                        f"Communication may be a bottleneck. Consider: "
                        f"(1) overlapping communication with computation, "
                        f"(2) checking if NVLink is being used (NCCL_P2P_LEVEL=NVL), "
                        f"(3) reducing AllReduce frequency via gradient accumulation."
                    ))
        return recs

    def print_report(self, recommendations: list[Recommendation] | None = None):
        """Print a formatted report of all recommendations."""
        if recommendations is None:
            recommendations = self.analyze()

        severity_icons = {"info": "[OK]", "warning": "[!!]", "critical": "[XX]"}

        print(f"\n{'='*72}")
        print(f"  HPC Cluster Scheduling Recommendations")
        print(f"  Analysis window: {self.window}")
        print(f"{'='*72}")

        if not recommendations:
            print("  No recommendations - cluster looks healthy.")
        else:
            for cat in ["critical", "warning", "info"]:
                cat_recs = [r for r in recommendations if r.severity == cat]
                if not cat_recs:
                    continue
                print(f"\n  --- {cat.upper()} ---")
                for r in cat_recs:
                    icon = severity_icons[r.severity]
                    print(f"  {icon} [{r.category}] {r.message}")

        print(f"\n{'='*72}\n")


def main():
    parser = argparse.ArgumentParser(description="GPU Scheduling Recommender")
    parser.add_argument("--prometheus-url", type=str, default="http://localhost:9090")
    parser.add_argument("--window", type=str, default="1h",
                        help="PromQL time window for averaging (e.g., 1h, 30m, 2h)")
    args = parser.parse_args()

    recommender = SchedulingRecommender(args.prometheus_url, args.window)
    recommender.print_report()


if __name__ == "__main__":
    main()
