# High Performance Computing System Architecture for AI/LLM

## What is HPC?

High Performance Computing (HPC) refers to the practice of aggregating computing power to deliver much higher performance than a single workstation, solving large-scale computational problems. For AI/LLM workloads, HPC systems are specifically designed to handle the massive compute, memory, and data movement demands of training and serving large language models.

## Key Architectural Components

### 1. Compute Nodes
  - Each node is essentially a powerful server containing CPUs, GPUs/accelerators, RAM, and local storage
- Modern AI clusters use nodes with 4-8 GPUs each (e.g., NVIDIA DGX systems with 8x H100 GPUs)
- CPUs handle orchestration, data preprocessing, and scheduling; GPUs handle the actual tensor math

### 2. Accelerators (GPUs/TPUs)
- The primary workhorse for AI workloads
- GPUs (NVIDIA H100, A100, B200) provide thousands of cores optimized for matrix operations
- Google TPUs are custom ASICs designed specifically for tensor computations
- AMD MI300X is an emerging competitor

### 3. High-Speed Interconnects
- Intra-node: NVLink/NVSwitch connects GPUs within a single node (900 GB/s on H100)
- Inter-node: InfiniBand (400 Gb/s NDR) or RoCE connects nodes across the cluster
- Low latency and high bandwidth are critical because LLM training requires constant synchronization of gradients across GPUs

### 4. High-Bandwidth Memory
- GPUs use HBM (High Bandwidth Memory) — H100 has 80GB HBM3 at 3.35 TB/s
- LLMs are often memory-bound, not compute-bound, making memory bandwidth a bottleneck

### 5. Storage System
- Parallel file systems (Lustre, GPFS/Spectrum Scale) provide shared storage across all nodes
- NVMe SSDs for fast local scratch space
- Training datasets for LLMs can be terabytes to petabytes in size

### 6. Network Fabric
- Fat-tree or dragonfly topologies minimize congestion
- Non-blocking switches ensure full bisection bandwidth
- Critical for all-reduce operations during distributed training

## Why LLMs Need HPC

| Challenge | HPC Solution |
|-----------|-------------|
| Models have billions of parameters (e.g., LLaMA 3 — 70B+) | Distribute model across many GPUs (model parallelism) |
| Training data is massive (trillions of tokens) | Parallel data loading from high-speed storage |
| Single GPU lacks enough memory | Tensor parallelism splits layers across GPUs |
| Training takes weeks even on fast hardware | Hundreds/thousands of GPUs working in parallel |
| Gradient synchronization across GPUs | High-speed interconnects (NVLink, InfiniBand) |

## Parallelism Strategies

- **Data Parallelism**: Each GPU holds a full model copy, processes different data batches, then synchronizes gradients
- **Tensor Parallelism**: Individual layers are split across GPUs within a node (requires fast NVLink)
- **Pipeline Parallelism**: Different layers are assigned to different GPUs, data flows through them like a pipeline
- **Expert Parallelism**: For Mixture-of-Experts models, different experts reside on different GPUs

## Typical HPC Cluster Layout for LLM Training

```
┌─────────────────────────────────────────────┐
│              Head/Login Nodes                │
│         (job submission, compilation)        │
├─────────────────────────────────────────────┤
│            High-Speed Network               │
│         (InfiniBand / RoCE Fabric)          │
├──────┬──────┬──────┬──────┬──────┬──────────┤
│Node 1│Node 2│Node 3│Node 4│ ...  │ Node N   │
│8xGPU │8xGPU │8xGPU │8xGPU │      │ 8xGPU   │
│NVLink│NVLink│NVLink│NVLink│      │ NVLink   │
├──────┴──────┴──────┴──────┴──────┴──────────┤
│         Parallel File System (Lustre)       │
│         High-Speed Shared Storage           │
└─────────────────────────────────────────────┘
```