# Project 4: HPC Cluster Performance Monitoring & Profiling Toolkit (2026)

**Technologies:** Kubernetes, Prometheus, Grafana, NCCL, Python, C

---

## 4.1 What This Project Does

When you run HPC workloads on a cluster of machines with GPUs, you need to know: Are all GPUs being used? Is there a bottleneck in communication between nodes? Is one node slower than others? This project builds a monitoring dashboard that collects GPU metrics, profiles NCCL communication between GPUs, and suggests where to improve resource usage.

**Analogy:** think of a factory floor with 20 machines. A monitoring system shows you which machines are idle, which are overheating, and where the assembly line is getting backed up. That is what this project does for a GPU cluster.

---

## 4.2 Key Concepts Explained Simply

### What is Prometheus?

Prometheus is a monitoring tool that "scrapes" (collects) metrics from your applications every few seconds. Your app exposes an endpoint (e.g., `/metrics`) that returns numbers like `gpu_utilization=85%`, `memory_used=12GB`. Prometheus stores this time-series data so you can query things like "what was the average GPU utilization over the last hour?"

### What is Grafana?

Grafana is a visualization tool that connects to Prometheus (and other data sources) and displays dashboards with graphs, gauges, and alerts. You build dashboards by writing queries like "show me gpu_utilization for each node over time."

### What is NCCL Profiling?

NCCL (NVIDIA Collective Communication Library) provides GPU-to-GPU communication operations like AllReduce, Broadcast, and AllGather. Profiling means measuring how long each operation takes. If AllReduce takes 40% of total runtime, that is a communication bottleneck — the GPUs are sitting idle waiting for data. Profiling helps you find these bottlenecks. You can enable NCCL debug logging with the environment variable `NCCL_DEBUG=INFO`, or use NVIDIA Nsight Systems to visualize NCCL calls on a timeline.

### What is a Helm Chart?

Helm is a "package manager for Kubernetes" — like apt or pip but for K8s deployments. A Helm chart is a bundle of YAML files that describes how to deploy an application (Prometheus, Grafana, your custom exporter, etc.). Instead of applying 10 YAML files manually, you run `helm install monitoring ./my-chart` and it sets up everything.

---

## 4.3 Architecture Diagram

```
GPU Cluster (Kubernetes)
  +-------------------+    +-------------------+
  |  Node A           |    |  Node B           |
  |  GPU 0, GPU 1     |    |  GPU 2, GPU 3     |
  |                   |    |                   |
  | [DCGM Exporter]   |    | [DCGM Exporter]   |
  | (exposes GPU      |    | (exposes GPU      |
  |  metrics on :9400)|    |  metrics on :9400)|
  |                   |    |                   |
  | [NCCL App + Prof.]|    | [NCCL App + Prof.]|
  | (logs comm times) |    | (logs comm times) |
  +--------+----------+    +--------+----------+
           |                        |
           v                        v
  +--------------------------------------------+
  |         Prometheus Server                   |
  |  Scrapes /metrics from all nodes every 15s  |
  |  Stores time-series data                    |
  +---------------------+----------------------+
                        |
                        v
  +--------------------------------------------+
  |            Grafana Dashboard                |
  |  - GPU Utilization per node (line chart)    |
  |  - Memory Usage per GPU (gauge)             |
  |  - NCCL Communication Time (bar chart)      |
  |  - Temperature alerts                       |
  +--------------------------------------------+
                        |
                        v
  +--------------------------------------------+
  |       Scheduling Recommender (Python)       |
  | Reads Prometheus data via API               |
  | "GPU 2 is idle 60% of the time,             |
  |  consider consolidating workloads"          |
  +--------------------------------------------+
```

---

## 4.4 Step-by-Step Walkthrough

### Step 1: Set Up Kubernetes with GPU Support

```bash
# Start minikube with GPU support (or use a cloud cluster)
minikube start --driver=docker --gpus all

# Install NVIDIA device plugin (allows K8s to see GPUs)
kubectl create -f https://raw.githubusercontent.com/NVIDIA/\
  k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Verify GPUs are visible
kubectl get nodes -o json | jq ".items[].status.capacity"
```

### Step 2: Deploy Prometheus + Grafana via Helm

```bash
# Add Helm repos
helm repo add prometheus-community \
  https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts

# Install Prometheus
helm install prometheus prometheus-community/prometheus \
  --set server.persistentVolume.enabled=false

# Install Grafana
helm install grafana grafana/grafana \
  --set adminPassword=admin \
  --set service.type=NodePort
```

### Step 3: Deploy DCGM Exporter for GPU Metrics

```bash
# DCGM (Data Center GPU Manager) exports GPU metrics to Prometheus
helm install dcgm-exporter \
  https://nvidia.github.io/dcgm-exporter/helm-charts/dcgm-exporter

# This exposes metrics like:
#   DCGM_FI_DEV_GPU_UTIL      -> GPU utilization %
#   DCGM_FI_DEV_FB_USED       -> GPU memory used (MB)
#   DCGM_FI_DEV_GPU_TEMP      -> GPU temperature
#   DCGM_FI_PROF_DRAM_ACTIVE  -> memory bandwidth utilization
```

### Step 4: Write NCCL Profiling Wrappers

```python
# nccl_profiler.py - wraps NCCL calls via PyTorch distributed and measures timing
import torch
import torch.distributed as dist
import time
import json
import os

# Initialize NCCL process group
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device(f"cuda:{rank}")

def profiled_allreduce(tensor):
    """Wraps NCCL AllReduce and logs the time taken."""
    torch.cuda.synchronize()  # ensure GPU ops are done before timing
    start = time.perf_counter()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()  # wait for NCCL op to finish
    elapsed = time.perf_counter() - start
    if rank == 0:
        print(json.dumps({
            "operation": "AllReduce",
            "time_ms": elapsed * 1000,
            "data_size_bytes": tensor.nelement() * tensor.element_size()
        }))
    return tensor

def profiled_broadcast(tensor, src=0):
    """Wraps NCCL Broadcast and logs the time taken."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    dist.broadcast(tensor, src=src)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if rank == 0:
        print(json.dumps({
            "operation": "Broadcast",
            "time_ms": elapsed * 1000
        }))
    return tensor

def profiled_allgather(tensor):
    """Wraps NCCL AllGather and logs the time taken."""
    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    torch.cuda.synchronize()
    start = time.perf_counter()
    dist.all_gather(gather_list, tensor)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if rank == 0:
        print(json.dumps({
            "operation": "AllGather",
            "time_ms": elapsed * 1000
        }))
    return torch.cat(gather_list, dim=0)

# Example: profile AllReduce with different tensor sizes
for size in [1024, 1024*1024, 64*1024*1024]:
    t = torch.randn(size, device=device)
    profiled_allreduce(t)

# Tip: also set NCCL_DEBUG=INFO to see NCCL internal logs
# export NCCL_DEBUG=INFO
# torchrun --nproc_per_node=2 nccl_profiler.py
```

### Step 5: Build the Scheduling Recommender

```python
# recommender.py - queries Prometheus and gives suggestions
import requests

PROMETHEUS_URL = "http://prometheus-server:9090"

def get_avg_gpu_util(hours=1):
    """Query average GPU utilization over the past N hours."""
    query = f'avg_over_time(DCGM_FI_DEV_GPU_UTIL[{hours}h])'
    resp = requests.get(f"{PROMETHEUS_URL}/api/v1/query",
                        params={"query": query})
    results = resp.json()["data"]["result"]
    for r in results:
        gpu = r["metric"]["gpu"]
        util = float(r["value"][1])
        print(f"GPU {gpu}: avg utilization = {util:.1f}%")
        if util < 30:
            print(f"  -> RECOMMENDATION: GPU {gpu} is underutilized. "
                  f"Consider consolidating workloads.")
        elif util > 95:
            print(f"  -> RECOMMENDATION: GPU {gpu} is saturated. "
                  f"Consider distributing load.")

get_avg_gpu_util(hours=1)
```

---

## 4.5 Key Points

- "I built this because in HPC, understanding where time is spent — computation vs. communication — is critical for optimization."
- "The NCCL profiling wrappers helped me discover that AllReduce was taking 35% of total training time in one workload, which led me to try overlapping communication with computation using CUDA streams."
- "I used torch.cuda.synchronize() before and after each NCCL call to get accurate wall-clock timings, since NCCL operations are asynchronous by default."
- "Prometheus + Grafana is the industry standard for monitoring Kubernetes workloads. DCGM Exporter gives GPU-specific metrics that are essential for HPC."
- "Using Helm charts for deployment means anyone can reproduce my monitoring setup with a single command."

