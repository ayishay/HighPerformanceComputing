# Building a High Performance Computing Cluster for LLM Training and Inference

## What Are We Building?

We are building a **GPU cluster** — a group of powerful servers connected together — specifically designed to train and serve large language models (like LLaMA, GPT, etc.). Here is the technology stack we will use:

| Component | Technology | Role |
|-----------|-----------|------|
| **Compute** | NVIDIA GPUs (H100) | The "brains" — do all the math for training/inference |
| **Intra-node connection** | NVLink / NVSwitch | Super-fast cable connecting GPUs **inside** a single server |
| **Inter-node connection** | InfiniBand (IB) | Super-fast network connecting **different servers** together |
| **Communication library** | NCCL | Software that moves data between GPUs efficiently |
| **Scheduler** | Kubernetes (K8s) | Software that decides which jobs run on which GPUs |

**Important clarification:** NCCL is **not** an accelerator — it is a **communication library**. The GPUs are the accelerators (they accelerate the math). NCCL is the software that helps GPUs talk to each other. Think of it this way:
- **GPUs** = workers doing the heavy lifting
- **NVLink / InfiniBand** = roads between workers
- **NCCL** = the traffic management system that moves data on those roads as efficiently as possible
- **Kubernetes** = the manager who assigns work to workers

---

## The Big Picture: What Does the Cluster Look Like?

Before diving into details, let's see the full picture of what we're building:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        OUR HPC CLUSTER                              │
│                                                                     │
│  ┌─────────── Kubernetes Control Plane ──────────┐                  │
│  │  API Server │ Scheduler │ etcd │ Controllers  │                  │
│  └──────────────────────┬────────────────────────┘                  │
│                         │ (manages everything)                      │
│                         │                                           │
│  ═══════════════════════╪═══════════════════════════════            │
│  ║           InfiniBand Network Fabric (400 Gb/s)     ║            │
│  ║    (super-fast network connecting all servers)      ║            │
│  ═══╤══════════╤══════════╤══════════╤════════════════             │
│     │          │          │          │                              │
│  ┌──▼────┐ ┌──▼────┐ ┌──▼────┐ ┌──▼────┐                         │
│  │Server │ │Server │ │Server │ │Server │  ... more servers         │
│  │Node 0 │ │Node 1 │ │Node 2 │ │Node 3 │                          │
│  │       │ │       │ │       │ │       │                           │
│  │8 GPUs │ │8 GPUs │ │8 GPUs │ │8 GPUs │  (8 GPUs per server)     │
│  │  ↕↕↕  │ │  ↕↕↕  │ │  ↕↕↕  │ │  ↕↕↕  │                         │
│  │NVLink │ │NVLink │ │NVLink │ │NVLink │  (GPUs talk inside)      │
│  └───────┘ └───────┘ └───────┘ └───────┘                          │
│     │          │          │          │                              │
│  ═══╧══════════╧══════════╧══════════╧════════════════             │
│  ║         Shared Storage (Lustre / NFS / Ceph)       ║            │
│  ║    (training data + model checkpoints live here)    ║            │
│  ═════════════════════════════════════════════════════              │
│                                                                     │
│  NCCL runs on every GPU, managing all data movement                │
│  between GPUs (both inside servers via NVLink and                  │
│  across servers via InfiniBand)                                    │
└─────────────────────────────────────────────────────────────────────┘
```

**In plain English:** We have multiple powerful servers (called "nodes"), each stuffed with 8 GPUs. Inside each server, the GPUs are connected to each other with ultra-fast NVLink cables. The servers themselves are connected to each other with InfiniBand networking. NCCL is software running on every GPU that figures out the fastest way to move data around. Kubernetes sits on top and decides which training/inference jobs run where.

---

## Part 1: The Hardware Layer

### 1.1 Choosing the GPU Servers (Nodes)

Each server in our cluster is a powerful machine packed with GPUs. The most common choice for LLM training is the **NVIDIA DGX H100** or an equivalent server from companies like Supermicro, Dell, or Lenovo.

**What's inside ONE server node:**

```
┌─────────────────────────────────────────────────────────┐
│                    One Server Node                       │
│                                                          │
│  CPUs:  2× AMD EPYC 9654 (96 cores each = 192 total)   │
│         → Handle data loading, preprocessing, and        │
│           orchestrating the GPUs                         │
│                                                          │
│  RAM:   2 TB DDR5 system memory                          │
│         → Holds training data batches before sending      │
│           to GPUs                                        │
│                                                          │
│  GPUs:  8× NVIDIA H100 SXM (80 GB HBM3 each)           │
│         → The actual workers that do the matrix math     │
│         → 640 GB total GPU memory per server             │
│                                                          │
│  NVLink: NVSwitch connecting all 8 GPUs                  │
│         → 900 GB/s between any two GPUs in this server   │
│                                                          │
│  NICs:  8× ConnectX-7 InfiniBand adapters (400 Gb/s)    │
│         → One NIC per GPU for direct GPU-to-network      │
│           communication (GPUDirect RDMA)                 │
│                                                          │
│  Storage: 8× 3.84 TB NVMe SSDs (local scratch)          │
│         → Fast temporary storage for data during training │
│                                                          │
│  Power: ~10 kW per server (these are power-hungry!)      │
└─────────────────────────────────────────────────────────┘
```

**Why 8 GPUs per server?** This is the sweet spot because:
- NVLink/NVSwitch can connect up to 8 GPUs with full bandwidth
- It matches the physical design of GPU baseboards (SXM form factor)
- Most distributed training frameworks assume 8 GPUs per node

**How many servers do we need?** That depends on the model size:

| Model Size | Minimum GPUs | Servers (8 GPUs each) | Why |
|-----------|-------------|----------------------|-----|
| 7B params | 1-8 | 1 | Fits on 1-2 GPUs (FP16: 14 GB) |
| 13B params | 2-8 | 1 | Fits on 2 GPUs (FP16: 26 GB) |
| 70B params | 16-64 | 2-8 | Parameters alone = 140 GB, plus optimizer states |
| 405B params | 128-512 | 16-64 | Massive model, needs many GPUs |
| Training 70B from scratch | 256-2048 | 32-256 | Need high throughput for trillions of tokens |

For this guide, let's build a **moderate cluster: 8 servers × 8 GPUs = 64 GPUs total**. This is enough to train a 70B model or serve multiple smaller models.

---

### 1.2 NVLink: The Fast Lane Inside Each Server

**What is NVLink?**

NVLink is a proprietary high-speed connection created by NVIDIA that directly links GPUs together **inside a single server**. Think of it as a private highway between GPUs — much wider and faster than the regular PCIe "road" that connects GPUs to the CPU.

**Why do we need it?** During LLM training, GPUs inside the same server constantly exchange data (partial results, activations, weights). The faster this happens, the less time GPUs spend waiting, and the faster training goes.

**Speed comparison:**

```
Connection type:          Bandwidth:        Analogy:
─────────────────────────────────────────────────────
PCIe Gen5 x16             64 GB/s           → Country road
NVLink 4.0 (H100)         900 GB/s          → 14-lane superhighway
                          (14× faster!)

That means: Data moves between GPUs 14× faster on NVLink
compared to PCIe. For LLM training where GPUs are constantly
exchanging gradients, this is a MASSIVE difference.
```

**How NVSwitch works:**

Without NVSwitch, NVLink connects GPUs in pairs — GPU 0 can talk fast to GPU 1, but talking to GPU 7 requires hopping through intermediate GPUs (slow). NVSwitch fixes this by acting as a central hub:

```
WITHOUT NVSwitch (old way — daisy chain):
  GPU 0 ↔ GPU 1 ↔ GPU 2 ↔ GPU 3
  Problem: GPU 0 talking to GPU 3 must go through GPU 1 and GPU 2
           (slow, adds latency)

WITH NVSwitch (our way — full mesh):
              ┌─────────────────────────────┐
              │         NVSwitch             │
              │  (central hub — connects     │
              │   ALL GPUs to ALL GPUs)      │
              └─┬───┬───┬───┬───┬───┬───┬─┘
                │   │   │   │   │   │   │
              GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7

  Now: GPU 0 can talk to GPU 7 directly at 900 GB/s
       GPU 3 can talk to GPU 5 directly at 900 GB/s
       ANY GPU can talk to ANY GPU at full speed simultaneously!
```

**What uses NVLink in LLM training?**

Tensor Parallelism — the technique where a single transformer layer is split across multiple GPUs. Each GPU computes part of the result, then they must combine their partial results. This combine step (called "all-reduce") happens **every single layer, every single step**. It must be ultra-fast, which is why it runs on NVLink.

```
Example: One transformer layer with Tensor Parallelism across 8 GPUs

  Input tokens
       │
       ▼
  ┌──────────────────────────────────────────┐
  │  Attention layer: split across 8 GPUs     │
  │                                           │
  │  GPU 0: computes attention heads 0-7      │
  │  GPU 1: computes attention heads 8-15     │
  │  GPU 2: computes attention heads 16-23    │
  │  ...                                      │
  │  GPU 7: computes attention heads 56-63    │
  │                                           │
  │  Now they must COMBINE results:           │
  │  ← All-Reduce over NVLink (900 GB/s) →   │
  │  This takes ~0.1 ms because NVLink is     │
  │  so fast                                  │
  └──────────────────────────────────────────┘
       │
       ▼
  Combined result → next layer
```

---

### 1.3 InfiniBand: The Fast Lane Between Servers

**What is InfiniBand?**

InfiniBand (IB) is a high-speed networking technology designed for HPC clusters. While your home network uses Ethernet (1-10 Gbps), InfiniBand runs at **400 Gbps per port** — 40 to 400 times faster than home Ethernet.

**Why not just use regular Ethernet?** Three reasons:

| Feature | Regular Ethernet | InfiniBand |
|---------|-----------------|------------|
| Bandwidth | 1-100 Gbps | 400 Gbps (NDR) |
| Latency | ~50-500 microseconds | ~1 microsecond |
| CPU overhead | High (CPU processes packets) | Near zero (RDMA bypasses CPU) |
| GPU-direct | No | Yes (GPUDirect RDMA) |

**RDMA (Remote Direct Memory Access)** — this is the killer feature. With regular Ethernet, data goes: GPU → CPU → RAM → NIC → Network → NIC → RAM → CPU → GPU. With InfiniBand RDMA, data goes: GPU → NIC → Network → NIC → GPU. The CPU is completely bypassed, saving massive overhead.

```
Regular Ethernet (slow, many copies):
  GPU memory                              GPU memory
      │                                       ▲
      ▼ copy to CPU                           │ copy from CPU
  CPU + RAM                               CPU + RAM
      │                                       ▲
      ▼ send                                  │ receive
  Ethernet NIC ═══ network ═══ Ethernet NIC

  Total: 4 memory copies, CPU involved at every step
  Latency: ~100+ microseconds

InfiniBand with GPUDirect RDMA (fast, zero copies through CPU):
  GPU memory                              GPU memory
      │                                       ▲
      └──► IB NIC ═══ network ═══ IB NIC ────┘

  Total: 0 CPU copies, data goes directly from GPU to network
  Latency: ~1-2 microseconds
```

**InfiniBand Network Topology**

Our cluster needs an InfiniBand "fabric" — the network of switches connecting all servers. The most common topology is a **fat-tree**, which ensures every server can talk to every other server at full speed:

```
                    ┌──────────┐ ┌──────────┐
                    │Core SW 0 │ │Core SW 1 │    ← Core switches
                    └────┬┬────┘ └────┬┬────┘      (top level)
                     ┌───┘└───┐  ┌───┘└───┐
                     │        │  │        │
                ┌────▼──┐ ┌──▼──▼─┐ ┌────▼──┐
                │Leaf 0 │ │Leaf 1 │ │Leaf 2 │    ← Leaf switches
                └┬┬┬┬───┘ └┬┬┬┬──┘ └┬┬┬┬───┘      (connect to servers)
                 ││││      ││││     ││││
              ┌──┘│││   ┌──┘│││  ┌──┘│││
              │ ┌─┘││   │ ┌─┘││  │ ┌─┘││
              │ │┌─┘│   │ │┌─┘│  │ │┌─┘│
              │ ││ ┌┘   │ ││ ┌┘  │ ││ ┌┘
              ▼ ▼▼ ▼    ▼ ▼▼ ▼   ▼ ▼▼ ▼
             Servers   Servers   Servers

Fat-tree ensures:
- Every server can reach every other server
- Full bisection bandwidth (no bottleneck at the top)
- If one switch fails, traffic re-routes automatically
```

**Hardware shopping list for InfiniBand:**

For our 8-server cluster with 8 NICs per server (one per GPU):

| Component | Quantity | Purpose |
|-----------|----------|---------|
| NVIDIA ConnectX-7 NICs (400 Gb/s) | 64 (8 per server) | Connect each GPU to the network |
| NVIDIA QM9700 leaf switches (64-port 400 Gb/s) | 2 | Connect servers to network |
| NVIDIA QM9700 core switches | 2 | Connect leaf switches (for full bisection BW) |
| InfiniBand cables (400 Gb/s) | ~80 | Connect everything together |

**Why one NIC per GPU?** This enables GPUDirect RDMA — each GPU has its own dedicated network port. When GPU 3 on Server 0 needs to send data to GPU 3 on Server 5, it goes directly through its own NIC without competing for bandwidth with other GPUs.

```
Server 0:                              Server 1:
┌────────────────────────┐             ┌────────────────────────┐
│ GPU 0 → NIC 0 ─────────╫─── IB ────╫── NIC 0 ← GPU 0       │
│ GPU 1 → NIC 1 ─────────╫─── IB ────╫── NIC 1 ← GPU 1       │
│ GPU 2 → NIC 2 ─────────╫─── IB ────╫── NIC 2 ← GPU 2       │
│ ...                     │           │                  ...    │
│ GPU 7 → NIC 7 ─────────╫─── IB ────╫── NIC 7 ← GPU 7       │
└────────────────────────┘             └────────────────────────┘

Each GPU has its own private 400 Gb/s (50 GB/s) link to the network.
8 GPUs × 50 GB/s = 400 GB/s total bandwidth per server.
```

---

### 1.4 Shared Storage

All servers need access to the same training data and model checkpoints. We use a **parallel file system** that all nodes can read/write simultaneously.

```
Options:

┌───────────────────────────────────────────────────────────┐
│  Storage Type       │ Speed       │ Best For              │
├───────────────────────────────────────────────────────────┤
│  Lustre             │ Very fast   │ Large-scale training  │
│  (parallel FS)      │ 100+ GB/s   │ data, checkpoints     │
│                     │ aggregate   │                       │
├───────────────────────────────────────────────────────────┤
│  Ceph / CephFS      │ Fast        │ K8s-native storage,   │
│  (distributed)      │ 10-50 GB/s  │ persistent volumes    │
├───────────────────────────────────────────────────────────┤
│  NFS                │ Moderate    │ Config files, small   │
│  (simple)           │ 1-10 GB/s   │ datasets, home dirs   │
├───────────────────────────────────────────────────────────┤
│  Local NVMe SSDs    │ Fastest     │ Temporary scratch,    │
│  (per-node)         │ 7+ GB/s     │ cached datasets       │
│                     │ per node    │                       │
└───────────────────────────────────────────────────────────┘
```

**Practical approach:** Use Lustre or Ceph for shared data, and local NVMe for caching:

```
Training data flow:

  ┌───────────┐    copy once     ┌─────────────┐   fast read    ┌──────┐
  │  Lustre   │ ──────────────► │ Local NVMe  │ ─────────────► │ GPU  │
  │  (shared) │   at job start   │ (per-node)  │  during train  │      │
  └───────────┘                  └─────────────┘                └──────┘

Why? Reading from shared storage for every batch is slow.
Copy the dataset to fast local NVMe once, then read from there.
This is called "data staging" or "caching."
```

---

## Part 2: NCCL — The Communication Engine

### 2.1 What Does NCCL Actually Do?

NCCL (NVIDIA Collective Communications Library, pronounced "nickel") is the software library that **moves data between GPUs**. Every time GPUs need to share information during training, NCCL handles it.

**Analogy:** If GPUs are workers in different offices, and NVLink/InfiniBand are the hallways between offices, then NCCL is the **mail delivery system**. It figures out the fastest route, the best schedule, and the most efficient way to deliver packages (data) between workers (GPUs).

**What happens without NCCL?** You would have to write your own code to:
1. Figure out the GPU topology (which GPUs are connected by NVLink vs. InfiniBand)
2. Choose the best algorithm (ring, tree, etc.)
3. Break data into chunks and send them in the right order
4. Handle errors and retries
5. Overlap communication with computation

This is incredibly complex. NCCL does it all automatically.

### 2.2 When Does NCCL Run During LLM Training?

Let's walk through **one training step** and see every point where NCCL is needed:

```
ONE TRAINING STEP (what happens when the model learns from one batch of data):

Step 1: DATA LOADING (CPU)
  CPU reads a batch of text from storage
  CPU tokenizes text into numbers
  CPU sends tokens to GPU memory
  → NCCL not needed (this is CPU → GPU, handled by CUDA)

Step 2: FORWARD PASS (GPU compute + NCCL)
  The model processes the input tokens through all its layers.

  For each transformer layer:
    a) Each GPU computes its portion of the attention layer
       → Local GPU compute, no communication needed

    b) GPUs must combine their partial attention results
       → NCCL All-Reduce over NVLink (intra-node, ~0.1 ms)

    c) Each GPU computes its portion of the MLP (feed-forward) layer
       → Local GPU compute, no communication needed

    d) GPUs must combine their partial MLP results
       → NCCL All-Reduce over NVLink (intra-node, ~0.1 ms)

  Repeat for all 80 layers (for a 70B model)

  At the end: the model produces predicted next tokens
  We compare predictions to actual tokens → loss value

Step 3: BACKWARD PASS (GPU compute + NCCL)
  Same as forward, but in reverse — computing gradients.

  For each transformer layer (in reverse order):
    a) Compute local gradients
       → Local GPU compute

    b) Synchronize gradients across ALL GPUs (not just within a server)
       → NCCL All-Reduce over NVLink (within server)
       → NCCL All-Reduce over InfiniBand (across servers)

       This is the BIGGEST communication step.
       For a 70B model: ~140 GB of gradient data must be
       synchronized across all 64 GPUs.

Step 4: OPTIMIZER UPDATE (GPU compute)
  Each GPU updates its portion of the model weights
  → Local GPU compute, no communication needed

Step 5: REPEAT from Step 1 with the next batch
```

**Summary of NCCL usage in one step:**

| Phase | NCCL Operation | Network Used | Data Volume |
|-------|---------------|-------------|-------------|
| Forward: tensor parallel | All-Reduce (per layer) | NVLink | ~small (activations) |
| Forward: pipeline parallel | Send/Recv (between stages) | InfiniBand | ~medium (activations) |
| Backward: gradient sync | All-Reduce (global) | NVLink + InfiniBand | ~large (all gradients) |
| ZeRO: parameter gather | All-Gather | NVLink + InfiniBand | ~large (model weights) |

### 2.3 How NCCL Chooses the Best Path

NCCL is smart — it automatically detects the network topology and picks the fastest algorithm. Here's what happens when NCCL initializes:

```
NCCL Initialization (happens once when training starts):

1. DISCOVER TOPOLOGY
   NCCL scans the system and builds a map:

   "I see 64 GPUs total:
    - GPUs 0-7 are on Node 0, connected by NVSwitch (900 GB/s)
    - GPUs 8-15 are on Node 1, connected by NVSwitch (900 GB/s)
    - ...
    - Nodes are connected by InfiniBand (50 GB/s per link, 8 links per node)
    - Each GPU has a dedicated InfiniBand NIC"

2. CHOOSE ALGORITHM per operation size:

   Small messages (< 256 KB):
   → Use TREE algorithm (low latency, fewer steps)
   → Like sending a text message — fast for small data

   Large messages (> 256 KB):
   → Use RING algorithm (high bandwidth, uses all links)
   → Like shipping a container — efficient for bulk data

   If SHARP-capable switches are available:
   → Use CollNet (reduction happens IN the network switch)
   → Like sorting mail at the post office instead of at each house

3. CHOOSE PROTOCOL:

   Tiny messages: LL (Low Latency) protocol
   → Prioritizes speed over throughput

   Medium messages: LL128 protocol
   → Balance of latency and throughput

   Large messages: Simple protocol
   → Maximum throughput, higher latency OK

4. BUILD COMMUNICATION CHANNELS
   NCCL creates multiple parallel "channels" to use all
   available NVLink and InfiniBand links simultaneously.

   With 8 InfiniBand NICs per node:
   → NCCL creates 8 channels, each using a different NIC
   → All 8 transfer data simultaneously = 8× more bandwidth
```

### 2.4 NCCL Configuration for Our Cluster

These environment variables tell NCCL how to behave on our cluster:

```bash
# ─── Network Configuration ───
# Tell NCCL to use InfiniBand (not Ethernet)
export NCCL_IB_DISABLE=0              # 0 = InfiniBand enabled (yes, use it!)

# Tell NCCL which network interface to use for control messages
export NCCL_SOCKET_IFNAME=ib0         # Use the InfiniBand interface

# Tell NCCL which InfiniBand cards to use
export NCCL_IB_HCA=mlx5               # Mellanox/NVIDIA ConnectX cards

# ─── GPUDirect RDMA ───
# Enable direct GPU-to-network transfers (bypassing CPU)
export NCCL_NET_GDR_LEVEL=5           # Maximum GPUDirect RDMA level
export NCCL_IB_GID_INDEX=3            # RoCE GID index for IB

# ─── Performance Tuning ───
# Allow NCCL to use all InfiniBand NICs
export NCCL_CROSS_NIC=1               # GPUs can use any NIC, not just their closest one

# Buffer size for communication (larger = more throughput for big messages)
export NCCL_BUFFSIZE=16777216          # 16 MB buffer

# Number of communication channels (more = higher bandwidth)
# Default is auto-detected, but you can force it:
# export NCCL_MIN_NCHANNELS=8
# export NCCL_MAX_NCHANNELS=16

# ─── Debugging (use during setup, disable in production) ───
export NCCL_DEBUG=INFO                 # Print what NCCL is doing
export NCCL_DEBUG_SUBSYS=ALL           # Print everything
# Change to WARN in production to reduce log noise
```

### 2.5 Verifying NCCL Performance

Before running any LLM training, **always test NCCL** to make sure the network is working at full speed. This is like test-driving a car before a road trip.

```bash
# Build the NCCL test suite
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=1 CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr/local/nccl

# ─── Test 1: Intra-node (8 GPUs within one server) ───
# This tests NVLink performance
mpirun -np 8 ./build/all_reduce_perf -b 1M -e 4G -f 2 -g 1

# Expected output (H100 NVSwitch):
#   size         busbw (GB/s)
#   1 MB         ~40 GB/s
#   1 GB         ~400 GB/s      ← Should be close to 450 GB/s theoretical
#   4 GB         ~430 GB/s

# If you see much less than 400 GB/s → NVLink problem!

# ─── Test 2: Inter-node (64 GPUs across 8 servers) ───
# This tests InfiniBand performance
mpirun -np 64 --hostfile hosts.txt \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_DEBUG=INFO \
    ./build/all_reduce_perf -b 1M -e 4G -f 2 -g 1

# Expected output (8× 400 Gb/s InfiniBand per node):
#   size         busbw (GB/s)
#   1 MB         ~10 GB/s
#   1 GB         ~150 GB/s
#   4 GB         ~180 GB/s     ← Should approach 200 GB/s (8 × 50 GB/s × efficiency)

# If you see much less than 150 GB/s → InfiniBand problem!
# Common issues: wrong NIC selected, GPUDirect RDMA not working,
# cable problem, switch misconfiguration
```

**How to read the results:**

```
# nccl-tests output columns explained:
#
#  size     = message size being tested
#  count    = number of elements
#  type     = data type (float)
#  redop    = reduction operation (sum)
#  time     = how long the operation took (microseconds)
#  algbw    = algorithm bandwidth (raw data / time)
#  busbw    = bus bandwidth (what you should compare against hardware specs)
#
# busbw is the KEY metric. It accounts for the algorithm overhead
# and tells you how much of the hardware bandwidth you're actually using.
#
# Good busbw / theoretical max:
#   > 80% = excellent
#   > 60% = acceptable
#   < 50% = something is wrong, investigate
```

---

## Part 3: Kubernetes — The Scheduler

### 3.1 Why Kubernetes for GPU Clusters?

Kubernetes manages our cluster. It decides:
- Which server runs which training job
- How many GPUs each job gets
- What happens when a job fails (restart it)
- How to scale inference up/down based on demand

**Why not SLURM?** Both work. We chose Kubernetes because:
- We want **both training AND inference** on the same cluster
- We want **autoscaling** for inference (add/remove copies based on traffic)
- We want **containerized workloads** (reproducible environments)
- We want **self-healing** (if a pod crashes, K8s restarts it automatically)
- Our team is familiar with cloud-native tooling

**Trade-off:** Kubernetes adds slightly more network overhead than bare-metal SLURM, and multi-node GPU training requires additional operators (Kubeflow, Volcano). For maximum training performance, some organizations use SLURM for training and Kubernetes for inference.

### 3.2 Setting Up Kubernetes for GPUs

Here is the step-by-step process to configure Kubernetes on our GPU cluster:

**Step 1: Install Kubernetes on all nodes**

```bash
# On every server node, install kubeadm, kubelet, and kubectl
# (the three core Kubernetes components)

# One node becomes the "control plane" (the brain)
# The other 7 become "worker nodes" (where GPUs jobs run)

# Initialize the control plane (run on the master node)
kubeadm init --pod-network-cidr=10.244.0.0/16

# Join worker nodes to the cluster (run on each worker)
kubeadm join <master-ip>:6443 --token <token> --discovery-token-ca-cert-hash <hash>
```

**Step 2: Install the NVIDIA GPU Operator**

The GPU Operator automatically installs everything Kubernetes needs to use GPUs:

```bash
# Add NVIDIA Helm repository
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

# Install the GPU Operator
# This single command installs: GPU drivers, container toolkit,
# device plugin, DCGM monitoring, and more
helm install gpu-operator nvidia/gpu-operator \
    --namespace gpu-operator \
    --create-namespace \
    --set driver.enabled=true \
    --set toolkit.enabled=true \
    --set devicePlugin.enabled=true \
    --set dcgmExporter.enabled=true \
    --set migManager.enabled=false \
    --set gds.enabled=true
```

What the GPU Operator installs on every node:

```
┌─────────────────────────────────────────────────┐
│            GPU Operator Components                │
│                                                   │
│  ┌─────────────────────────────────────────────┐ │
│  │ NVIDIA Driver (kernel module)                │ │
│  │ → Lets the OS talk to the GPU hardware       │ │
│  └─────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────┐ │
│  │ NVIDIA Container Toolkit                     │ │
│  │ → Lets containers (Docker/containerd) see    │ │
│  │   and use GPUs inside the container          │ │
│  └─────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────┐ │
│  │ NVIDIA Device Plugin                         │ │
│  │ → Tells Kubernetes "this node has 8 GPUs"    │ │
│  │ → Allocates specific GPUs to pods            │ │
│  └─────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────┐ │
│  │ DCGM Exporter                                │ │
│  │ → Monitors GPU health (temp, power, errors)  │ │
│  │ → Exports metrics to Prometheus/Grafana       │ │
│  └─────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────┐ │
│  │ GDS (GPUDirect Storage)                      │ │
│  │ → Allows direct NVMe-to-GPU data transfer    │ │
│  └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

**Step 3: Configure InfiniBand networking for Kubernetes**

By default, Kubernetes uses overlay networks (like Flannel or Calico) which add overhead. For GPU training, we need pods to use InfiniBand directly. We use **Multus** — a plugin that lets pods have multiple network interfaces:

```bash
# Install Multus (allows pods to use InfiniBand)
kubectl apply -f https://raw.githubusercontent.com/k8snetworkplumbingwg/multus-cni/master/deployments/multus-daemonset.yml

# Install RDMA shared device plugin (exposes InfiniBand to pods)
kubectl apply -f rdma-shared-device-plugin.yaml
```

Then create a network attachment definition for InfiniBand:

```yaml
# ib-network.yaml
# This tells Kubernetes: "There is an InfiniBand network available
# that pods can attach to"
apiVersion: k8s.cni.cncf.io/v1
kind: NetworkAttachmentDefinition
metadata:
  name: ib-sriov-network
  annotations:
    k8s.v1.cni.cncf.io/resourceName: rdma/rdma_shared_device_a
spec:
  config: |
    {
      "cniVersion": "0.3.1",
      "type": "host-device",
      "device": "ib0",
      "ipam": {
        "type": "whereabouts",
        "range": "192.168.1.0/24"
      }
    }
```

**Step 4: Install Volcano for batch scheduling**

Standard Kubernetes scheduler doesn't understand "I need all 64 GPUs to start at the same time." Volcano adds gang scheduling:

```bash
# Install Volcano scheduler
kubectl apply -f https://raw.githubusercontent.com/volcano-sh/volcano/master/installer/volcano-development.yaml
```

**Step 5: Install Kubeflow Training Operator**

For distributed training (PyTorchJob, MPIJob):

```bash
# Install Kubeflow Training Operator
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone"
```

**Step 6: Set up shared storage**

```yaml
# Create a PersistentVolume for training data
# This connects Kubernetes to our Lustre file system
apiVersion: v1
kind: PersistentVolume
metadata:
  name: training-data-pv
spec:
  capacity:
    storage: 100Ti          # 100 TB of training data
  accessModes:
    - ReadOnlyMany          # Many pods can read simultaneously
  mountOptions:
    - noatime               # Don't track access times (faster)
  csi:
    driver: lustre.csi.io
    volumeHandle: training-data
    volumeAttributes:
      mgs: lustre-mgs.cluster.local@tcp
      fs: training

---
# Claim the volume so pods can use it
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: training-data-pvc
spec:
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 100Ti
  volumeName: training-data-pv
```

### 3.3 Verifying the Kubernetes GPU Setup

After installation, verify everything works:

```bash
# Check that Kubernetes sees all GPUs
kubectl get nodes -o custom-columns=\
NAME:.metadata.name,\
GPUs:.status.capacity.nvidia\.com/gpu

# Expected output:
# NAME       GPUs
# node-00    8
# node-01    8
# node-02    8
# node-03    8
# node-04    8
# node-05    8
# node-06    8
# node-07    8

# Run a quick GPU test pod
kubectl run gpu-test --image=nvidia/cuda:12.2.0-base-ubuntu22.04 \
    --limits=nvidia.com/gpu=1 \
    --command -- nvidia-smi

# Check the output
kubectl logs gpu-test
# Should show GPU info (name, memory, temperature, etc.)
```

---

## Part 4: Running LLM Training

### 4.1 Distributed Training Job (PyTorchJob)

Here is a complete example that trains LLaMA-70B across all 64 GPUs in our cluster:

```yaml
# llama-70b-training.yaml
#
# This creates a distributed training job that:
# - Uses all 8 servers (64 GPUs total)
# - Uses Tensor Parallelism within each server (over NVLink)
# - Uses Data Parallelism across servers (over InfiniBand)
# - NCCL handles all GPU-to-GPU communication automatically

apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: llama-70b-pretrain
  namespace: training
spec:
  elasticPolicy:
    rdzvBackend: c10d           # PyTorch's built-in rendezvous
    minReplicas: 8              # Need at least 8 nodes
    maxReplicas: 8              # Use exactly 8 nodes
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        metadata:
          annotations:
            # Attach InfiniBand network to this pod
            k8s.v1.cni.cncf.io/networks: ib-sriov-network
        spec:
          schedulerName: volcano    # Use Volcano for gang scheduling
          # Make sure this runs on H100 nodes only
          nodeSelector:
            nvidia.com/gpu.product: NVIDIA-H100-SXM
          tolerations:
          - key: "nvidia.com/gpu"
            operator: "Exists"
            effect: "NoSchedule"
          containers:
          - name: trainer
            image: my-registry/llm-trainer:v2.0
            command:
            - torchrun
            - --nproc_per_node=8         # 8 GPUs per server
            - --nnodes=8                 # 8 servers total
            - train.py
            - --model=meta-llama/Llama-3-70B
            - --tensor-parallel-size=8   # Split layers across 8 GPUs (NVLink)
            - --data-parallel-size=8     # 8 copies of model across servers (IB)
            - --batch-size-per-gpu=4     # 4 sequences per GPU
            - --sequence-length=4096     # 4096 tokens per sequence
            - --bf16                     # Use BF16 mixed precision
            - --flash-attention          # Use Flash Attention (faster + less memory)
            - --gradient-checkpointing   # Trade compute for memory
            - --dataset=/data/pile       # Training data location
            - --output=/checkpoints/llama-70b
            env:
            # NCCL configuration
            - name: NCCL_IB_DISABLE
              value: "0"                 # Use InfiniBand
            - name: NCCL_SOCKET_IFNAME
              value: "ib0"               # InfiniBand interface
            - name: NCCL_IB_HCA
              value: "mlx5"              # NVIDIA InfiniBand cards
            - name: NCCL_NET_GDR_LEVEL
              value: "5"                 # Enable GPUDirect RDMA
            - name: NCCL_DEBUG
              value: "WARN"              # Minimal logging in production
            resources:
              limits:
                nvidia.com/gpu: 8        # All 8 GPUs
                memory: "512Gi"          # 512 GB RAM
                cpu: "128"               # 128 CPU cores
                rdma/rdma_shared_device_a: 1  # InfiniBand access
              requests:
                nvidia.com/gpu: 8
                memory: "512Gi"
                cpu: "128"
            volumeMounts:
            - name: training-data
              mountPath: /data
              readOnly: true
            - name: checkpoints
              mountPath: /checkpoints
            - name: dev-shm
              mountPath: /dev/shm        # Shared memory for NCCL
          volumes:
          - name: training-data
            persistentVolumeClaim:
              claimName: training-data-pvc
          - name: checkpoints
            persistentVolumeClaim:
              claimName: checkpoints-pvc
          - name: dev-shm
            emptyDir:
              medium: Memory
              sizeLimit: "128Gi"         # 128 GB shared memory for NCCL buffers
    Worker:
      replicas: 7                        # 7 workers + 1 master = 8 total
      template:
        # ... identical to Master template above ...
        metadata:
          annotations:
            k8s.v1.cni.cncf.io/networks: ib-sriov-network
        spec:
          schedulerName: volcano
          nodeSelector:
            nvidia.com/gpu.product: NVIDIA-H100-SXM
          tolerations:
          - key: "nvidia.com/gpu"
            operator: "Exists"
            effect: "NoSchedule"
          containers:
          - name: trainer
            image: my-registry/llm-trainer:v2.0
            command:
            - torchrun
            - --nproc_per_node=8
            - --nnodes=8
            - train.py
            - --model=meta-llama/Llama-3-70B
            - --tensor-parallel-size=8
            - --data-parallel-size=8
            - --batch-size-per-gpu=4
            - --sequence-length=4096
            - --bf16
            - --flash-attention
            - --gradient-checkpointing
            - --dataset=/data/pile
            - --output=/checkpoints/llama-70b
            env:
            - name: NCCL_IB_DISABLE
              value: "0"
            - name: NCCL_SOCKET_IFNAME
              value: "ib0"
            - name: NCCL_IB_HCA
              value: "mlx5"
            - name: NCCL_NET_GDR_LEVEL
              value: "5"
            - name: NCCL_DEBUG
              value: "WARN"
            resources:
              limits:
                nvidia.com/gpu: 8
                memory: "512Gi"
                cpu: "128"
                rdma/rdma_shared_device_a: 1
              requests:
                nvidia.com/gpu: 8
                memory: "512Gi"
                cpu: "128"
            volumeMounts:
            - name: training-data
              mountPath: /data
              readOnly: true
            - name: checkpoints
              mountPath: /checkpoints
            - name: dev-shm
              mountPath: /dev/shm
          volumes:
          - name: training-data
            persistentVolumeClaim:
              claimName: training-data-pvc
          - name: checkpoints
            persistentVolumeClaim:
              claimName: checkpoints-pvc
          - name: dev-shm
            emptyDir:
              medium: Memory
              sizeLimit: "128Gi"
```

**What happens when you submit this job:**

```
You run: kubectl apply -f llama-70b-training.yaml

1. Kubernetes API Server receives the request

2. Volcano scheduler checks:
   "This job needs 8 nodes with 8 GPUs each.
    Do I have 64 free GPUs? Let me check...
    Yes! Nodes 0-7 all have 8 free H100 GPUs."

3. Volcano uses GANG SCHEDULING:
   "I will start ALL 8 pods at the same time.
    If only 6 nodes are free, I wait until all 8 are free.
    (Because distributed training doesn't work with missing nodes)"

4. Kubelet on each node:
   "I received a pod. Let me:
    - Pull the container image
    - Assign 8 GPUs to this container
    - Mount the training data volume
    - Set up InfiniBand networking
    - Start the training command"

5. torchrun on each node:
   "I'm starting 8 training processes (one per GPU).
    I need to find the master node to coordinate...
    Found master at node-00:29500. Connecting..."

6. NCCL initializes:
   "I see 64 GPUs across 8 nodes.
    Intra-node: NVSwitch detected, 900 GB/s available
    Inter-node: InfiniBand ConnectX-7, 8 × 50 GB/s = 400 GB/s
    GPUDirect RDMA: enabled
    Using Ring algorithm for large all-reduces
    Using Tree algorithm for small broadcasts
    Creating 8 communication channels per node"

7. Training begins:
   - Each GPU processes 4 sequences of 4096 tokens per step
   - Global batch = 4 × 64 GPUs = 256 sequences per step
   - NCCL synchronizes gradients after each backward pass
   - Model checkpoint saved every N steps to /checkpoints
```

### 4.2 Monitoring Training

```bash
# Watch pod status
kubectl get pods -n training -w

# Expected output:
# NAME                          READY   STATUS    RESTARTS   AGE
# llama-70b-pretrain-master-0   1/1     Running   0          2m
# llama-70b-pretrain-worker-0   1/1     Running   0          2m
# llama-70b-pretrain-worker-1   1/1     Running   0          2m
# llama-70b-pretrain-worker-2   1/1     Running   0          2m
# llama-70b-pretrain-worker-3   1/1     Running   0          2m
# llama-70b-pretrain-worker-4   1/1     Running   0          2m
# llama-70b-pretrain-worker-5   1/1     Running   0          2m
# llama-70b-pretrain-worker-6   1/1     Running   0          2m

# View training logs (loss, throughput, etc.)
kubectl logs -n training llama-70b-pretrain-master-0 -f

# Expected log output:
# Step 100 | Loss: 8.42 | LR: 1.0e-04 | Tokens/s: 380,000 | MFU: 52%
# Step 200 | Loss: 7.15 | LR: 2.0e-04 | Tokens/s: 395,000 | MFU: 54%
# Step 300 | Loss: 6.33 | LR: 3.0e-04 | Tokens/s: 398,000 | MFU: 55%

# Check GPU utilization across all nodes
kubectl exec -n training llama-70b-pretrain-master-0 -- nvidia-smi
```

---

## Part 5: Running LLM Inference

### 5.1 Serving a Trained Model

After training, we deploy the model as an always-on API service. Inference has very different requirements from training:

```
Training vs Inference — different needs:

Training:                          Inference:
- Uses ALL GPUs for ONE job        - Uses a FEW GPUs per model copy
- Runs for days/weeks              - Runs forever (always-on service)
- Maximum throughput needed        - Low latency per request needed
- Batch size: large                - Batch size: varies (1 to hundreds)
- No autoscaling needed            - Autoscaling crucial (traffic varies)
- Failure = restart from checkpoint- Failure = users see errors
```

```yaml
# llm-inference.yaml
# Deploys LLaMA-70B as an API service using vLLM
# Autoscales from 2 to 6 replicas based on GPU utilization

apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-70b-inference
  namespace: inference
spec:
  replicas: 2                    # Start with 2 copies of the model
  selector:
    matchLabels:
      app: llama-70b
  template:
    metadata:
      labels:
        app: llama-70b
    spec:
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-H100-SXM
      tolerations:
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
        - --model=/models/llama-70b         # Path to trained model
        - --tensor-parallel-size=4          # Split across 4 GPUs (NVLink)
        - --max-model-len=8192              # Max context length
        - --gpu-memory-utilization=0.90     # Use 90% of GPU memory
        - --dtype=bfloat16                  # BF16 precision
        - --port=8000                       # API port
        ports:
        - containerPort: 8000
        env:
        - name: NCCL_IB_DISABLE
          value: "0"
        - name: NCCL_SOCKET_IFNAME
          value: "ib0"
        resources:
          limits:
            nvidia.com/gpu: 4               # 4 GPUs per replica
            memory: "256Gi"
            cpu: "32"
          requests:
            nvidia.com/gpu: 4
            memory: "256Gi"
            cpu: "32"
        # Health check: make sure the model is loaded and responding
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 120          # Model takes ~2 min to load
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 180
          periodSeconds: 30
        volumeMounts:
        - name: model-weights
          mountPath: /models
          readOnly: true
        - name: dev-shm
          mountPath: /dev/shm
      volumes:
      - name: model-weights
        persistentVolumeClaim:
          claimName: model-weights-pvc      # Trained model weights
      - name: dev-shm
        emptyDir:
          medium: Memory
          sizeLimit: "64Gi"

---
# Service: Makes the model accessible within the cluster
apiVersion: v1
kind: Service
metadata:
  name: llama-70b-service
  namespace: inference
spec:
  selector:
    app: llama-70b
  ports:
  - port: 8000
    targetPort: 8000
    protocol: TCP
  type: ClusterIP                 # Internal access only

---
# Ingress: Makes the model accessible from outside the cluster
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: llama-70b-ingress
  namespace: inference
  annotations:
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
spec:
  rules:
  - host: llm-api.mycompany.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: llama-70b-service
            port:
              number: 8000

---
# Autoscaler: Automatically add/remove replicas based on demand
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llama-70b-hpa
  namespace: inference
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llama-70b-inference
  minReplicas: 2                  # Always keep at least 2 copies running
  maxReplicas: 6                  # Scale up to 6 copies during peak traffic
  metrics:
  - type: Pods
    pods:
      metric:
        name: gpu_utilization     # Scale based on GPU usage
      target:
        type: AverageValue
        averageValue: "80"        # Add more replicas when GPUs > 80% busy
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 120   # Wait 2 min before scaling up
      policies:
      - type: Pods
        value: 1                  # Add 1 replica at a time
        periodSeconds: 120
    scaleDown:
      stabilizationWindowSeconds: 300   # Wait 5 min before scaling down
      policies:
      - type: Pods
        value: 1                  # Remove 1 replica at a time
        periodSeconds: 300
```

**What this setup does:**

```
Normal traffic (2 replicas):
  llama-70b-inference-pod-0  →  4 GPUs on Node 0  →  handles requests
  llama-70b-inference-pod-1  →  4 GPUs on Node 1  →  handles requests
  (remaining 48 GPUs free for training or other models)

Peak traffic (autoscales to 6 replicas):
  llama-70b-inference-pod-0  →  4 GPUs on Node 0
  llama-70b-inference-pod-1  →  4 GPUs on Node 1
  llama-70b-inference-pod-2  →  4 GPUs on Node 2  ← added automatically
  llama-70b-inference-pod-3  →  4 GPUs on Node 3  ← added automatically
  llama-70b-inference-pod-4  →  4 GPUs on Node 4  ← added automatically
  llama-70b-inference-pod-5  →  4 GPUs on Node 5  ← added automatically

Traffic drops back (autoscales down):
  Extra replicas are removed after 5 minutes of low usage
  GPUs are freed for other workloads
```

### 5.2 Using the Inference API

Once deployed, users can call the model:

```bash
# Send a request to the LLM API
curl https://llm-api.mycompany.com/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-70b",
    "prompt": "Explain quantum computing in simple terms:",
    "max_tokens": 256,
    "temperature": 0.7
  }'

# Response:
# {
#   "choices": [{
#     "text": "Quantum computing uses quantum bits (qubits) that can be
#              in multiple states at once, unlike classical bits that are
#              either 0 or 1. This allows quantum computers to explore
#              many solutions simultaneously..."
#   }]
# }
```

---

## Part 6: Putting It All Together — How Data Flows

Let's trace exactly what happens during **one training step** across our entire cluster:

```
STEP-BY-STEP: What happens when our 64 GPUs process one training batch

═══════════════════════════════════════════════════════════════
PHASE 1: DATA LOADING
═══════════════════════════════════════════════════════════════

Lustre Storage → NVMe Cache → CPU RAM → GPU Memory

Each server's CPU loads 4 sequences (4096 tokens each) from storage.
8 servers × 4 sequences = 32 sequences total (global batch).
CPUs tokenize and transfer data to their GPUs via PCIe.

Hardware used: CPU, NVMe SSD, PCIe
NCCL: not involved

═══════════════════════════════════════════════════════════════
PHASE 2: FORWARD PASS (per transformer layer × 80 layers)
═══════════════════════════════════════════════════════════════

Inside each server (Tensor Parallelism over NVLink):
┌──────────────────────────────────────────┐
│ Server 0: 8 GPUs each compute 1/8 of    │
│ the attention and MLP for their data     │
│                                          │
│ GPU 0: heads 0-7    GPU 4: heads 32-39  │
│ GPU 1: heads 8-15   GPU 5: heads 40-47  │
│ GPU 2: heads 16-23  GPU 6: heads 48-55  │
│ GPU 3: heads 24-31  GPU 7: heads 56-63  │
│                                          │
│ After attention: NCCL All-Reduce on      │
│ NVLink (900 GB/s) to combine results    │
│ Time: ~0.05 ms per layer                │
│                                          │
│ After MLP: NCCL All-Reduce on NVLink    │
│ Time: ~0.05 ms per layer                │
└──────────────────────────────────────────┘

All 8 servers do this SIMULTANEOUSLY on different data.
Hardware used: GPU Tensor Cores, NVLink
NCCL: All-Reduce within each server

═══════════════════════════════════════════════════════════════
PHASE 3: LOSS COMPUTATION
═══════════════════════════════════════════════════════════════

Each GPU computes loss on its portion of data.
Compare model's predictions vs actual next tokens.

Hardware used: GPU
NCCL: not involved (local computation)

═══════════════════════════════════════════════════════════════
PHASE 4: BACKWARD PASS + GRADIENT SYNC
═══════════════════════════════════════════════════════════════

This is the MOST communication-heavy phase.

Step A: Each GPU computes gradients locally (backward through layers)
Step B: Gradients must be AVERAGED across all 64 GPUs

The gradient sync uses NCCL All-Reduce ACROSS servers:

Server 0                    Server 1              Server 7
GPU 0-7                    GPU 8-15              GPU 56-63
  │                          │                      │
  │  gradients (140 GB)      │  gradients           │  gradients
  │                          │                      │
  └──────────────────────────┼──────────────────────┘
                             │
                    NCCL All-Reduce
                    over InfiniBand
                    (400 GB/s per node)
                             │
                    Algorithm: Ring
                    (bandwidth-optimal)
                             │
                    Time: ~1-2 seconds
                    for 70B model gradients
                             │
  ┌──────────────────────────┼──────────────────────┐
  │                          │                      │
  │  averaged gradients      │  averaged gradients  │  averaged gradients
  │  (identical on all GPUs) │                      │
Server 0                    Server 1              Server 7

Hardware used: InfiniBand (inter-node), NVLink (intra-node)
NCCL: All-Reduce across entire cluster

═══════════════════════════════════════════════════════════════
PHASE 5: OPTIMIZER UPDATE
═══════════════════════════════════════════════════════════════

Each GPU uses the averaged gradients to update its portion of
the model weights. All GPUs update independently (no communication).

Hardware used: GPU
NCCL: not involved

═══════════════════════════════════════════════════════════════
PHASE 6: REPEAT
═══════════════════════════════════════════════════════════════

Go back to Phase 1 with the next batch of training data.
Repeat millions of times until the model is trained.
```

---

## Part 7: Complete Hardware Bill of Materials

Here is everything you need to buy and build this cluster:

```
┌────────────────────────────────────────────────────────────────┐
│           HARDWARE BILL OF MATERIALS (8-Node Cluster)          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  COMPUTE (8 servers):                                          │
│  ├── 8× GPU Server (DGX H100 or equivalent)                   │
│  │   ├── 2× AMD EPYC 9654 CPUs (96 cores each)               │
│  │   ├── 2 TB DDR5 RAM                                        │
│  │   ├── 8× NVIDIA H100 SXM 80GB GPUs                        │
│  │   ├── 4× NVSwitch chips (connect all 8 GPUs)              │
│  │   ├── 8× ConnectX-7 InfiniBand NICs (400 Gb/s each)       │
│  │   ├── 8× 3.84 TB NVMe SSDs                                │
│  │   └── 2× 3000W power supplies                              │
│  │                                                             │
│  NETWORKING:                                                   │
│  ├── 2× NVIDIA QM9700 Leaf Switches (64-port 400 Gb/s)       │
│  ├── 2× NVIDIA QM9700 Core Switches (full bisection BW)      │
│  ├── ~80× InfiniBand cables (400 Gb/s, various lengths)       │
│  ├── 2× Ethernet management switches (for Kubernetes control) │
│  │                                                             │
│  STORAGE:                                                      │
│  ├── Lustre Parallel File System                               │
│  │   ├── 2× Metadata Servers (MDS)                            │
│  │   ├── 4× Object Storage Servers (OSS)                      │
│  │   ├── 48× 16 TB NVMe SSDs (768 TB usable)                 │
│  │   └── InfiniBand-connected to all compute nodes            │
│  │                                                             │
│  INFRASTRUCTURE:                                               │
│  ├── 2× 42U server racks                                      │
│  ├── Cooling: liquid cooling for GPU servers (~80 kW total)    │
│  ├── Power: 100 kW power delivery (with redundancy)           │
│  ├── UPS: Uninterruptible power supply                         │
│  └── 1× Management/login server (for Kubernetes control plane)│
│                                                                │
│  SOFTWARE (free / open-source):                                │
│  ├── Ubuntu 22.04 Server (operating system)                    │
│  ├── Kubernetes v1.28+ (cluster orchestration)                 │
│  ├── NVIDIA GPU Operator (GPU drivers + management)           │
│  ├── Volcano (batch scheduling)                                │
│  ├── Kubeflow Training Operator (distributed training)        │
│  ├── NVIDIA NCCL (GPU communication)                          │
│  ├── PyTorch + DeepSpeed (training framework)                 │
│  ├── vLLM (inference serving)                                  │
│  ├── Prometheus + Grafana (monitoring)                         │
│  └── Multus + RDMA device plugin (InfiniBand for K8s)         │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Part 8: Summary — How Everything Connects

```
┌──────────────────────────────────────────────────────────────┐
│                    COMPLETE ARCHITECTURE                      │
│                                                              │
│  Layer 5: APPLICATION                                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Training: PyTorch + DeepSpeed + Megatron-LM         │    │
│  │ Inference: vLLM + Triton Inference Server            │    │
│  │ "These are the programs that actually train/run LLMs"│    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         │ uses                               │
│  Layer 4: COMMUNICATION │                                    │
│  ┌──────────────────────▼──────────────────────────────┐    │
│  │ NCCL (GPU-to-GPU data movement)                     │    │
│  │ "Automatically finds the fastest path between GPUs   │    │
│  │  and moves data using optimal algorithms"            │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         │ runs on                            │
│  Layer 3: SCHEDULING    │                                    │
│  ┌──────────────────────▼──────────────────────────────┐    │
│  │ Kubernetes + Volcano + Kubeflow Training Operator    │    │
│  │ "Decides which jobs run on which GPUs, manages       │    │
│  │  failures, autoscales inference"                     │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         │ manages                            │
│  Layer 2: NETWORKING    │                                    │
│  ┌──────────────────────▼──────────────────────────────┐    │
│  │ NVLink (900 GB/s inside servers)                    │    │
│  │ InfiniBand (400 Gb/s between servers)               │    │
│  │ "The physical roads that data travels on"            │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         │ connects                           │
│  Layer 1: HARDWARE      │                                    │
│  ┌──────────────────────▼──────────────────────────────┐    │
│  │ 64× NVIDIA H100 GPUs (8 servers × 8 GPUs)          │    │
│  │ 2 TB RAM per server, NVMe SSDs, Lustre storage      │    │
│  │ "The physical machines that do the actual compute"   │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

1. **GPUs are the workers, NVLink is the fast hallway inside a building, InfiniBand is the highway between buildings.** You need both for LLM training because one server doesn't have enough GPUs.

2. **NCCL is not an accelerator — it's the communication library.** It sits between your training code and the hardware, automatically choosing the fastest way to move data between GPUs. Without NCCL, your GPUs would spend most of their time waiting for data instead of computing.

3. **NVLink handles intra-node communication (tensor parallelism)** — splitting individual layers across GPUs within one server. It's 14× faster than PCIe, which matters because this happens every layer, every step.

4. **InfiniBand handles inter-node communication (data parallelism)** — synchronizing gradients across servers. GPUDirect RDMA is critical — it lets GPUs send data to the network without involving the CPU.

5. **Kubernetes manages the cluster** — assigning GPUs to jobs, restarting failed containers, autoscaling inference replicas. Volcano adds gang scheduling (start all pods together or none). Kubeflow adds distributed training support.

6. **Always test NCCL performance with nccl-tests before training.** Many "slow training" problems are actually network misconfigurations. If nccl-tests shows low bandwidth, fix the network first.

7. **Training and inference have very different requirements.** Training needs all GPUs working together for days. Inference needs a few GPUs per model copy, with autoscaling for variable traffic. Kubernetes handles both on the same cluster.

8. **The software is all free and open-source.** The expensive part is the hardware — 8 DGX H100 servers cost roughly $2-3 million. The software stack (Kubernetes, NCCL, PyTorch, vLLM) costs nothing.
