"""
NCCL Metrics Exporter for Prometheus

Exposes NCCL profiling results as Prometheus-compatible metrics on an HTTP
endpoint (/metrics). This allows Grafana to visualize NCCL communication
performance alongside GPU hardware metrics from DCGM Exporter.

Metrics exported:
    nccl_operation_latency_ms      - Histogram of operation latencies
    nccl_operation_bandwidth_gbps  - Gauge of last measured bandwidth
    nccl_operation_count_total     - Counter of operations performed

Usage:
    python -m src.nccl_profiler.export_metrics --port 9091
"""

import argparse
import json
import threading
import time
from pathlib import Path

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

# --- Prometheus metrics ---

LATENCY_HISTOGRAM = Histogram(
    "nccl_operation_latency_ms",
    "Latency of NCCL collective operations in milliseconds",
    ["operation", "size_bucket"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0],
)

BANDWIDTH_GAUGE = Gauge(
    "nccl_operation_bandwidth_gbps",
    "Bus bandwidth of NCCL collective operations in GB/s",
    ["operation", "size_bucket"],
)

OP_COUNTER = Counter(
    "nccl_operation_count_total",
    "Total number of NCCL collective operations",
    ["operation"],
)


def _size_bucket(size_bytes: int) -> str:
    """Categorize data size into a human-readable bucket label."""
    if size_bytes < 1024:
        return "<1KB"
    elif size_bytes < 64 * 1024:
        return "1KB-64KB"
    elif size_bytes < 1024 * 1024:
        return "64KB-1MB"
    elif size_bytes < 32 * 1024 * 1024:
        return "1MB-32MB"
    else:
        return ">32MB"


def record_operation(operation: str, size_bytes: int, time_ms: float):
    """Record a single NCCL operation into Prometheus metrics."""
    bucket = _size_bucket(size_bytes)
    LATENCY_HISTOGRAM.labels(operation=operation, size_bucket=bucket).observe(time_ms)
    bw = (size_bytes / 1e9) / (time_ms / 1000) if time_ms > 0 else 0
    BANDWIDTH_GAUGE.labels(operation=operation, size_bucket=bucket).set(bw)
    OP_COUNTER.labels(operation=operation).inc()


class MetricsFileWatcher(threading.Thread):
    """
    Watches a JSON results file (produced by profiler.py or benchmark.py)
    and feeds new entries into Prometheus metrics.
    """

    def __init__(self, results_path: str, poll_interval: float = 5.0):
        super().__init__(daemon=True)
        self.results_path = Path(results_path)
        self.poll_interval = poll_interval
        self._last_count = 0

    def run(self):
        while True:
            try:
                if self.results_path.exists():
                    with open(self.results_path) as f:
                        data = json.load(f)
                    # Only process new records
                    new_records = data[self._last_count:]
                    for rec in new_records:
                        record_operation(
                            rec["operation"],
                            rec.get("data_size_bytes", rec.get("size_bytes", 0)),
                            rec.get("time_ms", rec.get("avg_latency_ms", 0)),
                        )
                    self._last_count = len(data)
            except (json.JSONDecodeError, KeyError):
                pass
            time.sleep(self.poll_interval)


def main():
    parser = argparse.ArgumentParser(description="NCCL Metrics Exporter")
    parser.add_argument("--port", type=int, default=9091,
                        help="Port to expose /metrics on")
    parser.add_argument("--results-file", type=str,
                        default="nccl_profile_results.json",
                        help="JSON file to watch for new profiling results")
    parser.add_argument("--poll-interval", type=float, default=5.0,
                        help="Seconds between file polls")
    args = parser.parse_args()

    print(f"Starting NCCL metrics exporter on :{args.port}/metrics")
    print(f"Watching results file: {args.results_file}")
    start_http_server(args.port)

    watcher = MetricsFileWatcher(args.results_file, args.poll_interval)
    watcher.start()

    # Keep main thread alive
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Shutting down.")


if __name__ == "__main__":
    main()
