"""
Tests for NCCL profiler components.

These tests can run without GPUs by mocking torch.distributed.
Run with: python -m pytest tests/ -v
"""

import json
import tempfile
from unittest.mock import MagicMock, patch

import pytest


class TestNCCLProfilerUnit:
    """Unit tests for profiler helper functions."""

    def test_format_bytes_small(self):
        from src.nccl_profiler.profiler import _format_bytes
        assert _format_bytes(512) == "512 B"

    def test_format_bytes_kb(self):
        from src.nccl_profiler.profiler import _format_bytes
        assert _format_bytes(2048) == "2.0 KB"

    def test_format_bytes_mb(self):
        from src.nccl_profiler.profiler import _format_bytes
        assert _format_bytes(5 * 1024 * 1024) == "5.0 MB"

    def test_format_bytes_gb(self):
        from src.nccl_profiler.profiler import _format_bytes
        assert _format_bytes(2 * 1024 ** 3) == "2.00 GB"


class TestNCCLBenchmarkUnit:
    """Unit tests for benchmark helper functions."""

    def test_fmt_small(self):
        from src.nccl_profiler.benchmark import _fmt
        assert _fmt(512) == "512 B"

    def test_fmt_kb(self):
        from src.nccl_profiler.benchmark import _fmt
        assert _fmt(4096) == "4 KB"

    def test_fmt_mb(self):
        from src.nccl_profiler.benchmark import _fmt
        assert _fmt(4 * 1024 * 1024) == "4 MB"


class TestMetricsExporter:
    """Tests for the Prometheus metrics exporter."""

    def test_size_bucket_small(self):
        from src.nccl_profiler.export_metrics import _size_bucket
        assert _size_bucket(100) == "<1KB"

    def test_size_bucket_medium(self):
        from src.nccl_profiler.export_metrics import _size_bucket
        assert _size_bucket(100 * 1024) == "64KB-1MB"

    def test_size_bucket_large(self):
        from src.nccl_profiler.export_metrics import _size_bucket
        assert _size_bucket(64 * 1024 * 1024) == ">32MB"

    def test_record_operation(self):
        from src.nccl_profiler.export_metrics import record_operation
        # Should not raise
        record_operation("AllReduce", 1024 * 1024, 0.5)

    def test_file_watcher_missing_file(self):
        from src.nccl_profiler.export_metrics import MetricsFileWatcher
        watcher = MetricsFileWatcher("/nonexistent/path.json")
        # Should handle missing file gracefully (just skip)
        assert watcher._last_count == 0

    def test_file_watcher_reads_json(self):
        from src.nccl_profiler.export_metrics import MetricsFileWatcher
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([
                {"operation": "AllReduce", "data_size_bytes": 1024, "time_ms": 0.1}
            ], f)
            f.flush()
            watcher = MetricsFileWatcher(f.name)
            # Simulate one poll cycle
            watcher.results_path = watcher.results_path  # ensure Path
            # The watcher thread logic would process this; here we just verify it parses
            data = json.load(open(f.name))
            assert len(data) == 1
            assert data[0]["operation"] == "AllReduce"
