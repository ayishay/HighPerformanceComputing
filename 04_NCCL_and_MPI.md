# NCCL and MPI: Communication Libraries for HPC in AI/LLM

## Why Communication Libraries Matter for LLMs

Training a large language model is a **distributed computing problem**. When a model is spread across dozens or hundreds of GPUs, those GPUs must constantly exchange data:

- **Gradients** must be synchronized after each backward pass (data parallelism)
- **Activations** must be passed between pipeline stages (pipeline parallelism)
- **Partial results** must be combined within a layer (tensor parallelism)

The communication library determines **how fast** these exchanges happen. A slow communication layer means GPUs sit idle waiting for data — wasting expensive hardware. For frontier LLM training, communication can consume **30-50% of total training time** if not optimized.

---

## MPI (Message Passing Interface)

### What Is MPI?

MPI is a **standardized specification** (not a single implementation) for message passing between processes in distributed computing. It has been the backbone of HPC communication since the 1990s.

MPI defines **how processes send and receive data** across a network. It is language-agnostic (C, C++, Fortran, Python bindings) and hardware-agnostic (runs on Ethernet, InfiniBand, shared memory).

### Key Implementations

| Implementation | Maintainer | Notes |
|---------------|------------|-------|
| **OpenMPI** | Open source consortium | Most widely used, broad hardware support |
| **MPICH** | Argonne National Lab | Reference implementation, basis for many derivatives |
| **Intel MPI** | Intel | Optimized for Intel hardware (Xeon, OPA) |
| **MVAPICH2** | Ohio State University | Optimized for InfiniBand and GPU-aware MPI |
| **NVIDIA HPC-X** | NVIDIA | Based on OpenMPI, optimized for NVIDIA networks |

### MPI Core Concepts

**Communicator and Ranks**

Every MPI program starts with a communicator — a group of processes that can talk to each other. Each process has a unique **rank** (ID number).

```
MPI_COMM_WORLD (default communicator)
┌────────┬────────┬────────┬────────┐
│ Rank 0 │ Rank 1 │ Rank 2 │ Rank 3 │
│ GPU 0  │ GPU 1  │ GPU 2  │ GPU 3  │
│ Node 0 │ Node 0 │ Node 1 │ Node 1 │
└────────┴────────┴────────┴────────┘
```

**Point-to-Point Communication**

Direct data transfer between two specific processes.

```c
// Rank 0 sends data to Rank 1
if (rank == 0) {
    MPI_Send(buffer, count, MPI_FLOAT, 1, tag, MPI_COMM_WORLD);
} else if (rank == 1) {
    MPI_Recv(buffer, count, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &status);
}
```

```
Rank 0 ──────[data]──────► Rank 1
         MPI_Send           MPI_Recv
```

**Collective Communication**

Operations that involve **all processes** in a communicator. These are the operations that matter most for LLM training.

| Operation | Description | Use in LLM Training |
|-----------|-------------|---------------------|
| **Broadcast** | One process sends data to all others | Distribute model weights at initialization |
| **Reduce** | Combine data from all processes to one | Aggregate loss values |
| **All-Reduce** | Combine data from all, result goes to all | **Gradient synchronization** (most critical) |
| **All-Gather** | Each process sends data, all receive everything | Gather model shards (ZeRO-3 forward pass) |
| **Reduce-Scatter** | Reduce + scatter result chunks | ZeRO gradient reduction |
| **All-to-All** | Each process sends unique data to every other | Expert parallelism routing (MoE) |
| **Scatter** | One process distributes chunks to all | Distribute data batches |
| **Gather** | All processes send data to one | Collect results |
| **Barrier** | Synchronization — all wait until everyone arrives | Checkpoint synchronization |

### Collective Operations Visualized

**Broadcast:**
```
Before:                After:
Rank 0: [A B C D]     Rank 0: [A B C D]
Rank 1: [. . . .]     Rank 1: [A B C D]
Rank 2: [. . . .]     Rank 2: [A B C D]
Rank 3: [. . . .]     Rank 3: [A B C D]
```

**Reduce (sum to Rank 0):**
```
Before:                After:
Rank 0: [1 2 3 4]     Rank 0: [10 20 30 40]  ← sum of all
Rank 1: [2 4 6 8]     Rank 1: [. . . .]
Rank 2: [3 6 9 12]    Rank 2: [. . . .]
Rank 3: [4 8 12 16]   Rank 3: [. . . .]
```

**All-Reduce (sum, result to all):**
```
Before:                After:
Rank 0: [1 2 3 4]     Rank 0: [10 20 30 40]
Rank 1: [2 4 6 8]     Rank 1: [10 20 30 40]
Rank 2: [3 6 9 12]    Rank 2: [10 20 30 40]
Rank 3: [4 8 12 16]   Rank 3: [10 20 30 40]
```

**All-Gather:**
```
Before:                After:
Rank 0: [A]            Rank 0: [A B C D]
Rank 1: [B]            Rank 1: [A B C D]
Rank 2: [C]            Rank 2: [A B C D]
Rank 3: [D]            Rank 3: [A B C D]
```

**Reduce-Scatter (sum, then scatter):**
```
Before:                        After:
Rank 0: [1 2 3 4]             Rank 0: [10]      ← chunk 0 of sum
Rank 1: [2 4 6 8]             Rank 1: [20]      ← chunk 1 of sum
Rank 2: [3 6 9 12]            Rank 2: [30]      ← chunk 2 of sum
Rank 3: [4 8 12 16]           Rank 3: [40]      ← chunk 3 of sum
```

**All-to-All:**
```
Before:                        After:
Rank 0: [A0 A1 A2 A3]         Rank 0: [A0 B0 C0 D0]
Rank 1: [B0 B1 B2 B3]         Rank 1: [A1 B1 C1 D1]
Rank 2: [C0 C1 C2 C3]         Rank 2: [A2 B2 C2 D2]
Rank 3: [D0 D1 D2 D3]         Rank 3: [A3 B3 C3 D3]
```

### MPI in LLM Training

MPI is used as the **process launcher and management layer** in most distributed training frameworks:

```bash
# Launch 64 processes across 8 nodes using MPI
mpirun -np 64 \
    --hostfile hosts.txt \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_SOCKET_IFNAME=ib0 \
    --map-by ppr:8:node \
    --bind-to numa \
    python train.py --model llama-70b
```

```
# hosts.txt
node001 slots=8
node002 slots=8
node003 slots=8
node004 slots=8
node005 slots=8
node006 slots=8
node007 slots=8
node008 slots=8
```

**GPU-Aware MPI:**
Traditional MPI copies data: GPU → CPU → Network → CPU → GPU. GPU-aware MPI (MVAPICH2-GDR, HPC-X) enables **GPUDirect RDMA** — data goes directly from GPU memory to the network card, bypassing CPU entirely.

```
Traditional MPI:
GPU 0 → CPU RAM → NIC ═══network═══ NIC → CPU RAM → GPU 1
         copy         send    recv         copy
         ↑ slow: 2 extra copies via CPU ↑

GPU-Aware MPI (GPUDirect RDMA):
GPU 0 → NIC ═══════network═══════ NIC → GPU 1
     direct          send    recv      direct
     ↑ fast: zero CPU copies ↑
```

---

## NCCL (NVIDIA Collective Communications Library)

### What Is NCCL?

NCCL (pronounced "nickel") is NVIDIA's **GPU-optimized collective communication library**. While MPI is general-purpose, NCCL is specifically designed to move data between NVIDIA GPUs as fast as physically possible.

NCCL understands the **GPU topology** — which GPUs are connected by NVLink, which are on the same PCIe switch, which nodes are connected by InfiniBand — and automatically selects the optimal communication path.

### Why NCCL Exists

MPI was designed for CPUs. Even GPU-aware MPI treats GPU data transfer as an add-on. NCCL was built from the ground up for GPU-to-GPU communication:

| Aspect | MPI (GPU-aware) | NCCL |
|--------|-----------------|------|
| Designed for | CPU processes | GPU-to-GPU |
| Topology awareness | Basic | Deep (NVLink, NVSwitch, PCIe, IB) |
| Kernel fusion | No | Yes (overlaps compute + comm) |
| CUDA stream integration | Limited | Native |
| Multi-GPU per process | Awkward | Native |
| Performance on NVIDIA HW | Good | Optimal |

### NCCL Architecture

```
┌──────────────────────────────────────────────┐
│                 NCCL Library                  │
│                                              │
│  ┌──────────────────────────────────────┐    │
│  │        Topology Detection            │    │
│  │  - NVLink graph                      │    │
│  │  - PCIe topology                     │    │
│  │  - InfiniBand fabric                 │    │
│  │  - Network interface mapping         │    │
│  └──────────────┬───────────────────────┘    │
│                 ▼                             │
│  ┌──────────────────────────────────────┐    │
│  │        Algorithm Selection           │    │
│  │  - Ring (bandwidth-optimal)          │    │
│  │  - Tree (latency-optimal)            │    │
│  │  - CollNet (switch-assisted)         │    │
│  └──────────────┬───────────────────────┘    │
│                 ▼                             │
│  ┌──────────────────────────────────────┐    │
│  │        Protocol Selection            │    │
│  │  - LL (Low Latency) — small msgs     │    │
│  │  - LL128 — medium msgs               │    │
│  │  - Simple — large msgs (bulk xfer)   │    │
│  └──────────────┬───────────────────────┘    │
│                 ▼                             │
│  ┌──────────────────────────────────────┐    │
│  │        Transport Layer               │    │
│  │  - P2P (NVLink, PCIe)               │    │
│  │  - SHM (shared memory)              │    │
│  │  - NET (InfiniBand, RoCE, TCP)      │    │
│  └──────────────────────────────────────┘    │
└──────────────────────────────────────────────┘
```

### NCCL Collective Operations

NCCL implements the same collective operations as MPI, but optimized for GPUs:

```c
// NCCL All-Reduce example (C API)
ncclAllReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum, comm, stream);

// All NCCL operations:
ncclAllReduce()      // Most used — gradient sync
ncclBroadcast()      // Weight distribution
ncclReduce()         // Loss aggregation
ncclAllGather()      // ZeRO-3 parameter gather
ncclReduceScatter()  // ZeRO gradient scatter
ncclSend()           // Point-to-point send
ncclRecv()           // Point-to-point receive
```

In PyTorch, NCCL is accessed through the distributed backend:
```python
import torch.distributed as dist

# Initialize NCCL backend
dist.init_process_group(backend="nccl")

# All-reduce gradient tensor across all GPUs
dist.all_reduce(gradient_tensor, op=dist.ReduceOp.SUM)

# Broadcast model weights from rank 0 to all
dist.broadcast(model.state_dict(), src=0)
```

### NCCL Algorithms

NCCL automatically selects the best algorithm based on message size and topology:

**Ring Algorithm (bandwidth-optimal)**

GPUs form a logical ring. Data is chunked and passed around the ring. After 2×(N-1) steps, every GPU has the full reduced result.

```
Ring All-Reduce with 4 GPUs:

Step 1: Each GPU sends a chunk to the next GPU in the ring

  GPU 0 ──chunk──► GPU 1
    ▲                │
    │              chunk
  chunk              │
    │                ▼
  GPU 3 ◄──chunk── GPU 2

Step 2-6: Continue passing and accumulating around the ring
Step 7:   All GPUs have the complete reduced result

Bandwidth utilization: 2(N-1)/N ≈ 100% for large N
Best for: Large messages (gradients in data parallelism)
```

**Tree Algorithm (latency-optimal)**

GPUs form a binary tree. Data is reduced up the tree and broadcast down. Requires only 2×log₂(N) steps.

```
Tree All-Reduce with 8 GPUs:

        ┌───[GPU 0]───┐         Reduce phase (up):
      [GPU 1]       [GPU 2]     Leaves send to parents
     /      \      /      \     Parents reduce + forward
  [GPU3] [GPU4] [GPU5] [GPU6]
    |
  [GPU7]                        Broadcast phase (down):
                                Root sends result down

Steps: 2 × log₂(8) = 6 steps (vs 14 for ring)
Best for: Small messages (latency-dominated)
```

**CollNet (Switch-assisted reduction)**

Uses InfiniBand switch hardware (SHARP — Scalable Hierarchical Aggregation and Reduction Protocol) to perform reductions in the network switch itself.

```
Without SHARP:
GPU → NIC → Switch → NIC → GPU → reduce → NIC → Switch → ...

With SHARP:
GPU → NIC → Switch (reduces in-switch) → NIC → GPU
             ↑ fewer hops, lower latency ↑
```

### NCCL Topology Awareness

NCCL auto-detects the GPU interconnect topology and builds optimal communication paths:

```
Example: DGX H100 (8 GPUs with NVSwitch)

Intra-node topology (detected by NCCL):
┌──────────────────────────────────────┐
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐   │
│  │GPU 0│ │GPU 1│ │GPU 2│ │GPU 3│   │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘   │
│     │       │       │       │       │
│  ═══╪═══════╪═══════╪═══════╪═══   │ ← NVSwitch (900 GB/s all-to-all)
│     │       │       │       │       │
│  ┌──┴──┐ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐   │
│  │GPU 4│ │GPU 5│ │GPU 6│ │GPU 7│   │
│  └─────┘ └─────┘ └─────┘ └─────┘   │
│                                      │
│  NICs: 8x ConnectX-7 (400Gb IB each)│
└──────────────────────────────────────┘

NCCL decision:
- Intra-node: Use NVLink/NVSwitch (900 GB/s) → Tensor Parallelism
- Inter-node: Use InfiniBand NICs (400 Gb/s) → Data/Pipeline Parallelism
```

### NCCL Environment Variables

Key tuning knobs for LLM training:

```bash
# Network interface selection
export NCCL_SOCKET_IFNAME=ib0          # Use InfiniBand interface
export NCCL_IB_DISABLE=0               # Enable InfiniBand (default)
export NCCL_IB_HCA=mlx5                # Select specific IB HCA

# Algorithm selection
export NCCL_ALGO=Ring                   # Force ring algorithm
export NCCL_ALGO=Tree                   # Force tree algorithm
export NCCL_ALGO=CollNet               # Force switch-assisted (SHARP)

# Protocol selection
export NCCL_PROTO=Simple               # Best for large messages

# Debugging
export NCCL_DEBUG=INFO                 # Print topology and algorithm choices
export NCCL_DEBUG_SUBSYS=ALL           # Detailed subsystem logging

# Performance tuning
export NCCL_BUFFSIZE=16777216          # 16MB communication buffer
export NCCL_NTHREADS=512               # Threads per NCCL kernel
export NCCL_NSOCKS_PERTHREAD=4         # Sockets per thread (TCP)
export NCCL_CROSS_NIC=1                # Allow cross-NIC communication

# Multi-NIC (use all InfiniBand ports)
export NCCL_IB_GID_INDEX=3             # RoCE GID index
export NCCL_NET_GDR_LEVEL=5            # GPUDirect RDMA level
```

---

## MPI vs NCCL: How They Work Together

In modern LLM training, MPI and NCCL are **not competitors — they complement each other**:

```
┌──────────────────────────────────────────────┐
│           Distributed Training Stack          │
│                                              │
│  ┌──────────────────────────────────────┐    │
│  │  Training Framework                   │    │
│  │  (PyTorch, DeepSpeed, Megatron-LM)   │    │
│  └──────────┬───────────────────────────┘    │
│             │                                │
│  ┌──────────▼───────────────────────────┐    │
│  │  torch.distributed                    │    │
│  │                                      │    │
│  │  backend="nccl"  ← GPU collectives   │    │
│  │  backend="gloo"  ← CPU collectives   │    │
│  │  backend="mpi"   ← MPI backend       │    │
│  └──────────┬───────────────────────────┘    │
│             │                                │
│  ┌──────────▼───────────────────────────┐    │
│  │  Process Management Layer             │    │
│  │                                      │    │
│  │  Option A: mpirun (MPI launcher)     │    │
│  │  Option B: torchrun (PyTorch native) │    │
│  │  Option C: srun (SLURM)             │    │
│  └──────────┬───────────────────────────┘    │
│             │                                │
│  ┌──────────▼───────────────────────────┐    │
│  │  Communication Libraries              │    │
│  │                                      │    │
│  │  NCCL: GPU↔GPU data transfer         │    │
│  │  MPI:  Process launch + CPU comms    │    │
│  │  Gloo: CPU tensors + fallback        │    │
│  └──────────────────────────────────────┘    │
└──────────────────────────────────────────────┘
```

**The typical division of labor:**

| Role | Library |
|------|---------|
| Launch processes across nodes | MPI (`mpirun`) or `torchrun` or SLURM (`srun`) |
| GPU-to-GPU gradient sync | NCCL |
| CPU-based metadata exchange | MPI or Gloo |
| GPU collective operations | NCCL |
| Process discovery/rendezvous | MPI or c10d (PyTorch) |

### Practical Example: Multi-Node LLM Training

```python
# train.py — runs on each GPU process
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize: NCCL for GPU ops, Gloo for CPU ops
dist.init_process_group(backend="nccl")

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# Model setup
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
model = model.to(f"cuda:{local_rank}")
model = DDP(model, device_ids=[local_rank])

# Training loop
for batch in dataloader:
    loss = model(batch)       # Forward pass (local compute)
    loss.backward()           # Backward pass (NCCL all-reduce happens here)
    optimizer.step()          # Update weights (local compute)
```

**Launch with MPI:**
```bash
mpirun -np 16 --hostfile hosts.txt \
    -x MASTER_ADDR=node001 \
    -x MASTER_PORT=29500 \
    -x NCCL_IB_DISABLE=0 \
    --map-by ppr:8:node \
    python train.py
```

**Launch with torchrun (no MPI needed):**
```bash
# On node 0:
torchrun --nnodes=2 --nproc_per_node=8 \
    --rdzv_id=job1 --rdzv_backend=c10d \
    --rdzv_endpoint=node001:29500 \
    train.py

# On node 1:
torchrun --nnodes=2 --nproc_per_node=8 \
    --rdzv_id=job1 --rdzv_backend=c10d \
    --rdzv_endpoint=node001:29500 \
    train.py
```

---

## MPI Operator for Kubernetes

For running MPI-based distributed training on Kubernetes:

```yaml
apiVersion: kubeflow.org/v2beta1
kind: MPIJob
metadata:
  name: llm-training-mpi
spec:
  slotsPerWorker: 8         # 8 GPUs per worker
  runPolicy:
    cleanPodPolicy: Running
  mpiReplicaSpecs:
    Launcher:
      replicas: 1
      template:
        spec:
          containers:
          - name: launcher
            image: llm-trainer:v1
            command:
            - mpirun
            - -np
            - "64"
            - --allow-run-as-root
            - -bind-to
            - none
            - -x NCCL_DEBUG=INFO
            - -x NCCL_IB_DISABLE=0
            - python
            - train.py
    Worker:
      replicas: 8
      template:
        spec:
          containers:
          - name: worker
            image: llm-trainer:v1
            resources:
              limits:
                nvidia.com/gpu: 8
```

---

## Communication Patterns in LLM Training

### Data Parallelism: All-Reduce Dominates

```
Forward Pass    Backward Pass    All-Reduce      Update
(local)         (local)          (NCCL)          (local)

GPU 0: compute → gradients ─┐
GPU 1: compute → gradients ─┼─► NCCL AllReduce ─► averaged gradients → optimizer step
GPU 2: compute → gradients ─┤   (Ring or Tree)
GPU 3: compute → gradients ─┘

Communication volume per step: 2 × model_size × (N-1)/N
For 70B model, 4 GPUs: ~2 × 140GB × 0.75 = 210 GB
```

### Tensor Parallelism: All-Reduce per Layer

```
Each transformer layer requires 2 all-reduce operations:

Input
  │
  ▼
┌─────────────────────────────┐
│ Column-Parallel Linear (QKV)│
│ GPU 0: W[:, :d/2]          │
│ GPU 1: W[:, d/2:]          │
└──────────┬──────────────────┘
           ▼
    NCCL All-Reduce ← (1st all-reduce)
           │
           ▼
┌─────────────────────────────┐
│ Row-Parallel Linear (Output)│
│ GPU 0: W[:d/2, :]          │
│ GPU 1: W[d/2:, :]          │
└──────────┬──────────────────┘
           ▼
    NCCL All-Reduce ← (2nd all-reduce)
           │
           ▼
     Next Layer

Communication per layer: 2 × batch × seq_len × hidden_dim × dtype_size
Requires NVLink bandwidth (intra-node only)
```

### Pipeline Parallelism: Point-to-Point Send/Recv

```
Stage 0 (GPU 0-3)              Stage 1 (GPU 4-7)
┌──────────────┐               ┌──────────────┐
│ Layers 0-39  │──NCCL Send──►│ Layers 40-79 │
│              │               │              │
│              │◄──NCCL Recv──│              │  (backward pass)
└──────────────┘               └──────────────┘

Communication: activations tensor between stages
Relatively small compared to all-reduce
Can use InfiniBand (inter-node) since less bandwidth needed
```

### ZeRO-3: All-Gather + Reduce-Scatter

```
Forward Pass:
  Before each layer → NCCL All-Gather (reconstruct full parameters)
  After each layer  → discard non-owned parameters (free memory)

Backward Pass:
  Before each layer → NCCL All-Gather (need full params for gradient)
  After each layer  → NCCL Reduce-Scatter (each GPU keeps its gradient shard)

┌──────┐  All-Gather   ┌──────────┐  Compute   ┌──────┐  Reduce-Scatter  ┌──────┐
│Shard │──────────────►│Full Param│──────────►│Grads │────────────────►│Shard │
│1/N   │               │Temporary │           │Full  │                 │1/N   │
└──────┘               └──────────┘           └──────┘                 └──────┘
```

---

## Performance Benchmarking

### NCCL Tests

NVIDIA provides `nccl-tests` for benchmarking communication performance:

```bash
# Build nccl-tests
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=1 CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr/local/nccl

# Run all-reduce benchmark (most important for LLM training)
mpirun -np 8 ./build/all_reduce_perf \
    -b 1M -e 4G -f 2 -g 1

# Example output:
#       size    count   type   redop    time   algbw   busbw
#       (B)             float    sum    (us)  (GB/s)  (GB/s)
     1048576   262144   float     sum   45.2   23.2    40.6
     2097152   524288   float     sum   52.1   40.3    70.5
     4194304  1048576   float     sum   66.8   62.8   109.9
   134217728 33554432   float     sum  1024.5  131.0   229.3
  4294967296 1.07e+09   float     sum 24891.2  172.5   301.9

# busbw = bus bandwidth (effective bandwidth accounting for algorithm)
# Target: close to theoretical max
#   NVLink H100: ~900 GB/s
#   InfiniBand NDR: ~400 Gb/s = ~50 GB/s per port
```

### OSU Micro-Benchmarks (MPI)

```bash
# Point-to-point latency
mpirun -np 2 ./osu_latency

# Point-to-point bandwidth
mpirun -np 2 ./osu_bw

# All-reduce latency
mpirun -np 8 ./osu_allreduce -m 4096:1073741824
```

---

## Communication Overlap: Hiding Latency

The key optimization in modern LLM training is **overlapping communication with computation** so GPUs never idle:

```
Without overlap:
GPU: [  Compute  ][  Wait  ][  Compute  ][  Wait  ]
NET:              [ Comm   ]             [ Comm   ]
     ↑ GPU idle during communication ↑

With overlap (gradient bucketing in DDP):
GPU: [  Compute layer N  ][  Compute layer N-1  ][  Compute layer N-2  ]
NET:                      [  AllReduce layer N   ][  AllReduce layer N-1]
     ↑ Communication of completed layers overlaps with compute of next layers ↑
```

PyTorch DDP implements this automatically:
- Gradients are bucketed (grouped) by layer
- As soon as a bucket's gradients are ready, NCCL all-reduce starts
- Next layer's backward pass runs concurrently on the GPU
- By the time backward pass finishes, most gradients are already synchronized

---

## Key Takeaways

1. **MPI is the process management standard** — it launches processes, manages ranks, and provides CPU-side communication. It's been the HPC backbone for 30+ years.

2. **NCCL is the GPU communication standard** — purpose-built for NVIDIA GPUs, it understands NVLink/NVSwitch/InfiniBand topology and selects optimal algorithms automatically.

3. **They work together, not against each other** — MPI (or torchrun/srun) launches processes; NCCL handles all GPU data movement.

4. **All-Reduce is the most critical operation** for LLM training — it synchronizes gradients across all GPUs in data parallelism and combines partial results in tensor parallelism.

5. **Ring algorithm** is bandwidth-optimal for large messages (gradients); **Tree algorithm** is latency-optimal for small messages.

6. **Communication can consume 30-50% of training time** — optimizations like gradient bucketing, communication overlap, and topology-aware placement are essential.

7. **NCCL environment variables** are the primary tuning mechanism — `NCCL_ALGO`, `NCCL_IB_DISABLE`, `NCCL_SOCKET_IFNAME` are commonly adjusted per cluster.

8. **Always benchmark with nccl-tests** before running LLM training to verify network performance matches hardware specs.
