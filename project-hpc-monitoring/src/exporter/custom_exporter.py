"""
Custom GPU + NCCL Metrics Exporter (FastAPI)

Combines GPU hardware metrics (via nvidia-smi / pynvml) with NCCL profiling
metrics into a single /metrics endpoint that Prometheus can scrape.

This is a lightweight alternative when DCGM Exporter is not available
(e.g., on a local workstation without the DCGM driver).

Endpoints:
    GET /metrics     - Prometheus-compatible metrics
    GET /health      - Health check
    GET /gpu/status  - JSON summary of all GPUs

Usage:
    uvicorn src.exporter.custom_exporter:app --host 0.0.0.0 --port 9090
"""

import subprocess
import json
from io import StringIO

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, JSONResponse

app = FastAPI(title="HPC Monitoring Exporter")


def _query_nvidia_smi() -> list[dict]:
    """Query nvidia-smi for GPU metrics. Returns list of dicts per GPU."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,temperature.gpu,utilization.gpu,"
                "utilization.memory,memory.used,memory.free,memory.total,"
                "power.draw,power.limit,clocks.sm,clocks.mem",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    gpus = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 12:
            continue
        gpus.append({
            "index": parts[0],
            "name": parts[1],
            "temperature_celsius": _safe_float(parts[2]),
            "gpu_utilization_pct": _safe_float(parts[3]),
            "mem_utilization_pct": _safe_float(parts[4]),
            "memory_used_mib": _safe_float(parts[5]),
            "memory_free_mib": _safe_float(parts[6]),
            "memory_total_mib": _safe_float(parts[7]),
            "power_draw_w": _safe_float(parts[8]),
            "power_limit_w": _safe_float(parts[9]),
            "sm_clock_mhz": _safe_float(parts[10]),
            "mem_clock_mhz": _safe_float(parts[11]),
        })
    return gpus


def _safe_float(s: str) -> float:
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def _format_prometheus(gpus: list[dict]) -> str:
    """Format GPU metrics as Prometheus exposition text."""
    lines = StringIO()

    metrics = [
        ("gpu_temperature_celsius", "gauge", "GPU temperature in Celsius",
         "temperature_celsius"),
        ("gpu_utilization_percent", "gauge", "GPU compute utilization percentage",
         "gpu_utilization_pct"),
        ("gpu_memory_utilization_percent", "gauge", "GPU memory utilization percentage",
         "mem_utilization_pct"),
        ("gpu_memory_used_mib", "gauge", "GPU memory used in MiB",
         "memory_used_mib"),
        ("gpu_memory_free_mib", "gauge", "GPU memory free in MiB",
         "memory_free_mib"),
        ("gpu_memory_total_mib", "gauge", "GPU total memory in MiB",
         "memory_total_mib"),
        ("gpu_power_draw_watts", "gauge", "GPU power draw in watts",
         "power_draw_w"),
        ("gpu_power_limit_watts", "gauge", "GPU power limit in watts",
         "power_limit_w"),
        ("gpu_sm_clock_mhz", "gauge", "GPU SM clock speed in MHz",
         "sm_clock_mhz"),
        ("gpu_memory_clock_mhz", "gauge", "GPU memory clock speed in MHz",
         "mem_clock_mhz"),
    ]

    for metric_name, metric_type, help_text, field in metrics:
        lines.write(f"# HELP {metric_name} {help_text}\n")
        lines.write(f"# TYPE {metric_name} {metric_type}\n")
        for gpu in gpus:
            gpu_idx = gpu["index"]
            gpu_name = gpu["name"].replace('"', '\\"')
            val = gpu[field]
            lines.write(
                f'{metric_name}{{gpu="{gpu_idx}",gpu_name="{gpu_name}"}} {val}\n'
            )
        lines.write("\n")

    return lines.getvalue()


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    gpus = _query_nvidia_smi()
    return _format_prometheus(gpus)


@app.get("/health")
async def health():
    """Health check endpoint."""
    gpus = _query_nvidia_smi()
    return {"status": "healthy", "gpu_count": len(gpus)}


@app.get("/gpu/status", response_class=JSONResponse)
async def gpu_status():
    """JSON summary of all GPUs."""
    gpus = _query_nvidia_smi()
    return {"gpus": gpus, "count": len(gpus)}
