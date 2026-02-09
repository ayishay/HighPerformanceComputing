"""
Tests for the scheduling recommender.

Run with: python -m pytest tests/ -v
"""

from unittest.mock import MagicMock, patch

import pytest

from src.recommender.prometheus_client import PrometheusClient, MetricResult
from src.recommender.recommender import SchedulingRecommender, Recommendation


class TestPrometheusClient:
    """Tests for the Prometheus query client."""

    def test_health_check_unreachable(self):
        client = PrometheusClient("http://localhost:99999")
        assert client.health_check() is False

    @patch("src.recommender.prometheus_client.requests.get")
    def test_instant_query_success(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "status": "success",
                "data": {
                    "resultType": "vector",
                    "result": [
                        {
                            "metric": {"gpu": "0", "instance": "node1:9400"},
                            "value": [1700000000, "85.5"],
                        }
                    ],
                },
            },
        )
        mock_get.return_value.raise_for_status = MagicMock()

        client = PrometheusClient("http://fake:9090")
        results = client.instant_query("DCGM_FI_DEV_GPU_UTIL")

        assert len(results) == 1
        assert results[0].value == 85.5
        assert results[0].labels["gpu"] == "0"

    @patch("src.recommender.prometheus_client.requests.get")
    def test_instant_query_error(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"status": "error", "error": "bad query"},
        )
        mock_get.return_value.raise_for_status = MagicMock()

        client = PrometheusClient("http://fake:9090")
        with pytest.raises(RuntimeError, match="bad query"):
            client.instant_query("invalid{")


class TestSchedulingRecommender:
    """Tests for the recommender logic."""

    def _make_recommender(self):
        recommender = SchedulingRecommender.__new__(SchedulingRecommender)
        recommender.client = MagicMock()
        recommender.window = "1h"
        return recommender

    def test_low_utilization_warning(self):
        recommender = self._make_recommender()
        recommender.client.get_gpu_utilization.return_value = [
            MetricResult(labels={"gpu": "0"}, value=15.0, timestamp=0),
        ]
        recs = recommender._analyze_utilization()
        assert len(recs) == 1
        assert recs[0].severity == "warning"
        assert "underutilized" in recs[0].message.lower() or "consolidat" in recs[0].message.lower()

    def test_high_utilization_info(self):
        recommender = self._make_recommender()
        recommender.client.get_gpu_utilization.return_value = [
            MetricResult(labels={"gpu": "0"}, value=97.0, timestamp=0),
        ]
        recs = recommender._analyze_utilization()
        assert len(recs) == 1
        assert "saturation" in recs[0].message.lower() or "distribut" in recs[0].message.lower()

    def test_healthy_utilization(self):
        recommender = self._make_recommender()
        recommender.client.get_gpu_utilization.return_value = [
            MetricResult(labels={"gpu": "0"}, value=75.0, timestamp=0),
        ]
        recs = recommender._analyze_utilization()
        assert len(recs) == 1
        assert "healthy" in recs[0].message.lower()

    def test_high_memory_warning(self):
        recommender = self._make_recommender()
        recommender.client.get_gpu_memory_used.return_value = [
            MetricResult(labels={"gpu": "0"}, value=72000, timestamp=0),
        ]
        recommender.client.get_gpu_memory_free.return_value = [
            MetricResult(labels={"gpu": "0"}, value=8000, timestamp=0),
        ]
        recs = recommender._analyze_memory()
        assert len(recs) == 1
        assert recs[0].severity == "warning"
        assert "OOM" in recs[0].message or "memory" in recs[0].message.lower()

    def test_critical_temperature(self):
        recommender = self._make_recommender()
        recommender.client.get_gpu_temperature.return_value = [
            MetricResult(labels={"gpu": "0"}, value=92.0, timestamp=0),
        ]
        recs = recommender._analyze_temperature()
        assert len(recs) == 1
        assert recs[0].severity == "critical"

    def test_nccl_high_latency(self):
        recommender = self._make_recommender()
        recommender.client.get_nccl_latency.return_value = [
            MetricResult(labels={"operation": "AllReduce"}, value=25.0, timestamp=0),
        ]
        recs = recommender._analyze_nccl()
        assert any("bottleneck" in r.message.lower() for r in recs)

    def test_query_failure_handled(self):
        recommender = self._make_recommender()
        recommender.client.get_gpu_utilization.side_effect = Exception("connection refused")
        recs = recommender._analyze_utilization()
        assert len(recs) == 1
        assert "Could not query" in recs[0].message
