"""
Prometheus Query Client

Provides a thin wrapper around the Prometheus HTTP API for querying GPU and
NCCL metrics. Used by the SchedulingRecommender to pull historical data.

Prometheus Query Language (PromQL) quick reference:
    DCGM_FI_DEV_GPU_UTIL          -> GPU utilization percentage (0-100)
    DCGM_FI_DEV_FB_USED           -> GPU framebuffer memory used (MiB)
    DCGM_FI_DEV_FB_FREE           -> GPU framebuffer memory free (MiB)
    DCGM_FI_DEV_GPU_TEMP          -> GPU temperature (Celsius)
    DCGM_FI_PROF_DRAM_ACTIVE      -> Memory bandwidth utilization (0-1)
    nccl_operation_latency_ms      -> NCCL operation latency (from our exporter)
"""

from dataclasses import dataclass

import requests


@dataclass
class MetricResult:
    """A single metric data point from Prometheus."""
    labels: dict[str, str]
    value: float
    timestamp: float


class PrometheusClient:
    """Client for querying the Prometheus HTTP API."""

    def __init__(self, base_url: str = "http://localhost:9090"):
        self.base_url = base_url.rstrip("/")

    def instant_query(self, query: str) -> list[MetricResult]:
        """Run an instant PromQL query and return results."""
        resp = requests.get(
            f"{self.base_url}/api/v1/query",
            params={"query": query},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if data["status"] != "success":
            raise RuntimeError(f"Prometheus query failed: {data.get('error', 'unknown')}")

        results = []
        for item in data["data"]["result"]:
            results.append(MetricResult(
                labels=item["metric"],
                value=float(item["value"][1]),
                timestamp=float(item["value"][0]),
            ))
        return results

    def range_query(self, query: str, start: str, end: str, step: str = "60s") -> list[dict]:
        """Run a range PromQL query (returns time series)."""
        resp = requests.get(
            f"{self.base_url}/api/v1/query_range",
            params={"query": query, "start": start, "end": end, "step": step},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        if data["status"] != "success":
            raise RuntimeError(f"Prometheus query failed: {data.get('error', 'unknown')}")

        return data["data"]["result"]

    # ---- Convenience methods for common GPU metrics ----

    def get_gpu_utilization(self, window: str = "1h") -> list[MetricResult]:
        """Average GPU utilization over a time window."""
        return self.instant_query(f"avg_over_time(DCGM_FI_DEV_GPU_UTIL[{window}])")

    def get_gpu_memory_used(self) -> list[MetricResult]:
        """Current GPU memory usage in MiB."""
        return self.instant_query("DCGM_FI_DEV_FB_USED")

    def get_gpu_memory_free(self) -> list[MetricResult]:
        """Current GPU memory free in MiB."""
        return self.instant_query("DCGM_FI_DEV_FB_FREE")

    def get_gpu_temperature(self) -> list[MetricResult]:
        """Current GPU temperature in Celsius."""
        return self.instant_query("DCGM_FI_DEV_GPU_TEMP")

    def get_memory_bandwidth_utilization(self, window: str = "1h") -> list[MetricResult]:
        """Average memory bandwidth utilization (0-1) over a time window."""
        return self.instant_query(f"avg_over_time(DCGM_FI_PROF_DRAM_ACTIVE[{window}])")

    def get_nccl_latency(self, operation: str, window: str = "1h") -> list[MetricResult]:
        """Average NCCL operation latency from our custom exporter."""
        return self.instant_query(
            f'avg_over_time(nccl_operation_latency_ms{{operation="{operation}"}}[{window}])'
        )

    def health_check(self) -> bool:
        """Check if Prometheus is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/-/healthy", timeout=5)
            return resp.status_code == 200
        except requests.ConnectionError:
            return False
