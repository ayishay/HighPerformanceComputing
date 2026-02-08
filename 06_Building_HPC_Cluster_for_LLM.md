# Building a High Performance Computing Cluster for Training and Serving LLMs

## What Are We Building?

We are building a **GPU cluster** — a group of powerful servers connected together — that can:
1. **Train** large language models (like LLaMA 70B, GPT-4 scale)
2. **Serve** (inference) those models to users via an API

Our technology choices:
- **GPUs with NVLink** — for fast GPU-to-GPU communication inside each server
- **InfiniBand** — for fast server-to-server communication across the cluster
- **NCCL** — the communication library that moves data between GPUs efficiently
- **Kubernetes** — the scheduler that manages which jobs run on which GPUs

> **Important clarification:** NCCL is a **communication library**, not an accelerator. The GPUs are the accelerators (they do the actual math). NCCL is the software that helps GPUs talk to each other. Think of GPUs as workers and NCCL as the phone system they use to coordinate.

---

## The Big Picture: What Does the Cluster Look Like?

Before diving into details, here's the full picture of what we're building:

```
                         ┌──────────────────────────────┐
                         │      Users / Applications     │
                         │   (send training jobs or       │
                         │    inference requests)         │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │    Kubernetes Control Plane    │
                         │  (the "brain" that decides     │
                         │   what runs where)             │
                         │                                │
                         │  API Server ← you talk to this │
                         │  Scheduler  ← places jobs      │
                         │  etcd       ← remembers state  │
                         └──────────────┬───────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
                    ▼                   ▼                   ▼
        ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
        │   GPU Server 0    │ │   GPU Server 1    │ │   GPU Server N    │
        │                   │ │                   │ │                   │
        │  ┌─────────────┐  │ │  ┌─────────────┐  │ │  ┌─────────────┐  │
        │  │ 8× GPUs     │  │ │  │ 8× GPUs     │  │ │  │ 8× GPUs     │  │
        │  │ connected by │  │ │  │ connected by │  │ │  │ connected by │  │
        │  │ NVLink      │  │ │  │ NVLink      │  │ │  │ NVLink      │  │
        │  └─────────────┘  │ │  └─────────────┘  │ │  └─────────────┘  │
        │  CPU + RAM + SSD  │ │  CPU + RAM + SSD  │ │  CPU + RAM + SSD  │
        │  InfiniBand NIC   │ │  InfiniBand NIC   │ │  InfiniBand NIC   │
        └────────┬──────────┘ └────────┬──────────┘ └────────┬──────────┘
                 │                     │                     │
                 └─────────────────────┼─────────────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │   InfiniBand Network    │
                          │  (high-speed switches    │
                          │   connecting all servers) │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │   Shared Storage         │
                          │  (Lustre / NFS / S3)     │
                          │  Training data, models,  │
                          │  checkpoints             │
                          └─────────────────────────┘
```

**In plain English:** We have multiple powerful servers, each packed with 8 GPUs. Inside each server, GPUs talk to each other through NVLink (super fast, like a private highway). Between servers, they talk through InfiniBand (very fast, like an express highway — much faster than regular Ethernet). Kubernetes sits on top and decides which server runs which job. NCCL is the software running on each GPU that figures out the fastest way to send data to other GPUs.

---

## Step 1: Choose and Set Up the Hardware

### 1.1 GPU Servers (Compute Nodes)

Each server in our cluster is a powerful machine packed with GPUs. For LLM work, we want servers with **8 GPUs per server** because that's the standard configuration that maximizes NVLink connectivity.

**Recommended server options:**

| Server | GPUs | GPU Memory | NVLink | InfiniBand | Use Case |
|--------|------|-----------|--------|------------|----------|
| NVIDIA DGX H100 | 8× H100 SXM | 80 GB HBM3 each | NVSwitch (900 GB/s all-to-all) | 8× ConnectX-7 (400 Gb/s each) | Frontier training |
| NVIDIA DGX A100 | 8× A100 SXM | 80 GB HBM2e each | NVSwitch (600 GB/s all-to-all) | 8× ConnectX-6 (200 Gb/s each) | Production training |
| Custom build | 8× H100 PCIe | 80 GB HBM3 each | NVLink Bridge (pairs only) | 1-2× ConnectX-7 | Budget option |

**What's inside each server:**

```
┌─────────────────────────────────────────────────────────────┐
│                    One GPU Server (Node)                      │
│                                                              │
│  CPUs: 2× AMD EPYC 9654 (96 cores each = 192 cores total)  │
│  ──────────────────────────────────────────────────────────  │
│  Why? CPUs handle:                                           │
│    - Data loading and preprocessing (tokenization)           │
│    - Orchestrating GPU kernels                               │
│    - Running Kubernetes kubelet agent                        │
│    - Network stack management                                │
│                                                              │
│  RAM: 2 TB DDR5                                              │
│  ──────────────────────────────────────────────────────────  │
│  Why? LLM training needs lots of CPU RAM for:               │
│    - Data loading buffers (hold next batches ready)          │
│    - CPU offloading (ZeRO-Offload puts optimizer states here)│
│    - Operating system and Kubernetes overhead                │
│                                                              │
│  Storage: 8× 3.84 TB NVMe SSDs (30 TB total)               │
│  ──────────────────────────────────────────────────────────  │
│  Why? Fast local storage for:                                │
│    - Caching training data (faster than reading from network)│
│    - Writing checkpoints quickly                             │
│    - Container images (Kubernetes pulls images here)         │
│                                                              │
│  GPUs: 8× NVIDIA H100 SXM (80 GB HBM3 each)               │
│  ──────────────────────────────────────────────────────────  │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                       │
│  │GPU 0 │ │GPU 1 │ │GPU 2 │ │GPU 3 │                       │
│  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘                       │
│     │        │        │        │                             │
│  ═══╪════════╪════════╪════════╪═══  ← NVSwitch Fabric      │
│     │        │        │        │       (900 GB/s any-to-any) │
│  ┌──┴───┐ ┌──┴───┐ ┌──┴───┐ ┌──┴───┐                       │
│  │GPU 4 │ │GPU 5 │ │GPU 6 │ │GPU 7 │                       │
│  └──────┘ └──────┘ └──────┘ └──────┘                       │
│                                                              │
│  Network: 8× NVIDIA ConnectX-7 (InfiniBand 400 Gb/s each)  │
│  ──────────────────────────────────────────────────────────  │
│  Why 8 NICs? Each GPU gets its own dedicated network card!  │
│  GPU 0 → NIC 0, GPU 1 → NIC 1, etc.                        │
│  This means GPU 0 on Server A can send data directly to     │
│  GPU 0 on Server B without going through the CPU.           │
│  This is called "GPUDirect RDMA."                           │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Understanding NVLink: Why It Matters

**The problem NVLink solves:**

GPUs inside the same server need to exchange data constantly during LLM training. Without NVLink, they'd communicate through PCIe (the standard connection):

```
Without NVLink (PCIe only):
  GPU 0 ──PCIe──► CPU ──PCIe──► GPU 1
  Speed: ~64 GB/s (PCIe 5.0 x16)
  Problem: CPU is a bottleneck. Data has to go through CPU memory.

With NVLink:
  GPU 0 ══NVLink══► GPU 1
  Speed: 900 GB/s (H100 NVSwitch)
  Benefit: 14× faster! Data goes directly GPU-to-GPU.
```

**NVLink vs NVSwitch — what's the difference?**

```
NVLink (point-to-point):
  Each GPU has a few direct NVLink connections to specific other GPUs.
  Like having dedicated phone lines between specific people.

  GPU 0 ════ GPU 1
  GPU 0 ════ GPU 2
  GPU 0 ╳╳╳╳ GPU 7  (no direct link — must go through intermediary)

NVSwitch (any-to-any):
  A special chip that connects ALL GPUs to ALL other GPUs simultaneously.
  Like a phone switchboard where anyone can call anyone at full speed.

  GPU 0 ═╗
  GPU 1 ═╬═══ NVSwitch ═══ All GPUs talk at 900 GB/s simultaneously
  GPU 2 ═╣
  ...    ║
  GPU 7 ═╝

  DGX H100 uses NVSwitch → every GPU pair gets full bandwidth.
  This is critical for Tensor Parallelism where every GPU in a layer
  must combine results after every operation.
```

**Why 900 GB/s matters for LLMs:**

During tensor parallelism, after every transformer layer, all 8 GPUs must combine their partial results (all-reduce). With a 70B model:

```
Data to synchronize per layer: ~64 MB (hidden states)
Number of layers: 80
Synchronizations per step: 80 × 2 = 160 (forward + backward)

Total data moved per step: 160 × 64 MB = 10.24 GB

At 900 GB/s (NVSwitch): 10.24 GB / 900 GB/s = 11.4 ms  ✓ Fast!
At 64 GB/s (PCIe only): 10.24 GB / 64 GB/s  = 160 ms   ✗ 14× slower!

Those extra 149 ms PER STEP add up to hours/days over a full training run.
```

### 1.3 InfiniBand Network: Connecting Servers Together

**The problem InfiniBand solves:**

NVLink only works inside one server. But a 70B-parameter model needs more than 8 GPUs (one server). We need servers to talk to each other. Regular Ethernet is too slow:

```
Regular Ethernet (25-100 Gb/s):
  Server 0 ──Ethernet──► Server 1
  Speed: 3-12.5 GB/s
  Latency: 10-50 microseconds
  Problem: Way too slow for training. GPUs wait for data.

InfiniBand NDR (400 Gb/s per port):
  Server 0 ══InfiniBand══► Server 1
  Speed: 50 GB/s per port (×8 ports = 400 GB/s per server)
  Latency: 1-2 microseconds
  Benefit: 30-100× faster than Ethernet! Near-zero latency.
```

**InfiniBand network topology:**

The InfiniBand switches connect all servers together. The topology (how switches are wired) matters a lot:

```
Fat-Tree Topology (most common for AI clusters):

            ┌──────────┐ ┌──────────┐
            │Core SW 0 │ │Core SW 1 │    ← Core layer (top)
            └────┬┬────┘ └────┬┬────┘      Full bandwidth between
                 ││           ││            any two servers
            ┌────┘└────┐ ┌───┘└─────┐
            ▼          ▼ ▼          ▼
       ┌─────────┐ ┌─────────┐ ┌─────────┐
       │Leaf SW 0│ │Leaf SW 1│ │Leaf SW 2│  ← Leaf layer (bottom)
       └─┬─┬─┬─┬┘ └─┬─┬─┬─┬┘ └─┬─┬─┬─┬┘    Connected to servers
         │ │ │ │     │ │ │ │     │ │ │ │
         ▼ ▼ ▼ ▼     ▼ ▼ ▼ ▼     ▼ ▼ ▼ ▼
        Servers      Servers      Servers

Why fat-tree?
  - "Non-blocking" = every server can talk to every other server
    at full speed simultaneously
  - No bottlenecks even when all servers communicate at once
  - Critical for all-reduce operations during training
```

**Hardware you need:**
- **InfiniBand Host Channel Adapters (HCAs)**: NVIDIA ConnectX-7 cards in each server
- **InfiniBand Switches**: NVIDIA Quantum-2 NDR switches (400 Gb/s per port)
- **InfiniBand Cables**: Fiber optic cables connecting servers to switches

**GPUDirect RDMA: The Secret Sauce**

The best part about combining InfiniBand with NVIDIA GPUs is **GPUDirect RDMA**. This lets GPU memory on Server A send data directly to GPU memory on Server B, completely bypassing the CPU:

```
Without GPUDirect RDMA (old way):
  GPU 0 (Server A)
    │ copy to CPU RAM ← slow, wastes CPU time
    ▼
  CPU RAM (Server A)
    │ send over InfiniBand
    ▼
  CPU RAM (Server B)
    │ copy to GPU RAM ← slow again
    ▼
  GPU 0 (Server B)

  Total: 4 data copies, high latency, CPU busy doing nothing useful

With GPUDirect RDMA (our setup):
  GPU 0 (Server A)
    │ direct send over InfiniBand ← GPU talks directly to NIC!
    ▼
  GPU 0 (Server B)

  Total: 1 data transfer, minimal latency, CPU is free to do other work
  This is possible because each GPU has its own dedicated InfiniBand NIC
```

### 1.4 Shared Storage

All servers need access to the same data (training datasets, model checkpoints). We need a **shared file system** that every server can read/write.

```
Options:

1. Lustre (Parallel File System)
   ┌────────┐ ┌────────┐ ┌────────┐
   │Storage │ │Storage │ │Storage │   ← Multiple storage servers
   │Server 0│ │Server 1│ │Server 2│      with many hard drives
   └───┬────┘ └───┬────┘ └───┬────┘
       └──────────┼──────────┘
                  │
       InfiniBand / Ethernet
                  │
       ┌──────────┼──────────┐
       ▼          ▼          ▼
   Server 0   Server 1   Server N     ← All compute servers see
   (sees /data) (sees /data) (sees /data)  the same /data directory

   Speed: 100+ GB/s aggregate (data spread across many servers)
   Best for: Large training datasets (terabytes)
   Used by: Most HPC clusters, national labs

2. NFS (Network File System)
   Simpler but slower. One server shares a directory.
   Speed: 1-10 GB/s (bottlenecked by single server)
   Best for: Small clusters, config files, code

3. S3-compatible object storage (MinIO, Ceph)
   Cloud-style storage. Good for Kubernetes-native workflows.
   Speed: Variable, depends on setup
   Best for: Cloud-hybrid setups, model artifact storage
```

**Recommended approach:**
- **Lustre** for training data and checkpoints (fast, parallel)
- **NFS** for code, configs, and small files (simple, easy to set up)
- **Local NVMe** on each server for caching (fastest, but not shared)

### 1.5 Cluster Size Planning

How many servers do you need? It depends on the model size:

```
Example: Training LLaMA 70B

Model memory requirements (per GPU, with different parallelism):
  Parameters (BF16):           140 GB
  Optimizer states (Adam):     560 GB (4× parameters)
  Gradients (BF16):            140 GB
  Activations:                 ~50-100 GB (depends on batch size)
  ─────────────────────────────────────
  Total:                       ~890 GB

  Single H100 GPU memory:     80 GB
  Minimum GPUs needed:        890 / 80 ≈ 12 GPUs (with ZeRO-3)

  But for good performance (not just fitting in memory):
  Recommended:                64 GPUs (8 servers × 8 GPUs)
  Reason: More GPUs = larger batch size = better training stability
          + enough room for activations and overhead

Cluster sizing examples:

┌──────────────┬─────────┬──────────┬──────────────────────┐
│ Model Size   │ Min GPUs│ Servers  │ Notes                │
├──────────────┼─────────┼──────────┼──────────────────────┤
│ 7B (LLaMA 3) │ 1-8     │ 1        │ Fits on 1 server     │
│ 13B          │ 2-8     │ 1        │ Fits on 1 server     │
│ 70B          │ 16-64   │ 2-8      │ Needs multi-node     │
│ 175B (GPT-3) │ 64-256  │ 8-32     │ Needs large cluster  │
│ 400B+        │ 256+    │ 32+      │ Massive cluster      │
└──────────────┴─────────┴──────────┴──────────────────────┘
```

---

## Step 2: Install and Configure the Software Stack

### 2.1 The Full Software Stack (Bottom to Top)

Think of the software as layers, like a building. Each layer depends on the one below it:

```
┌─────────────────────────────────────────────────┐
│  Layer 7: User's Training Code                   │  ← Your train.py
│  (PyTorch, DeepSpeed, Megatron-LM)              │
├─────────────────────────────────────────────────┤
│  Layer 6: ML Frameworks & Libraries              │  ← PyTorch, vLLM, etc.
│  (torch.distributed, Flash Attention)            │
├─────────────────────────────────────────────────┤
│  Layer 5: Communication Library                  │  ← NCCL (GPU-to-GPU comms)
│  (NCCL for GPUs, Gloo for CPUs)                 │
├─────────────────────────────────────────────────┤
│  Layer 4: Kubernetes + GPU Operator              │  ← Job scheduling
│  (scheduler, device plugin, monitoring)          │
├─────────────────────────────────────────────────┤
│  Layer 3: Container Runtime                      │  ← Docker / containerd
│  (NVIDIA Container Toolkit)                     │
├─────────────────────────────────────────────────┤
│  Layer 2: GPU Drivers + Libraries                │  ← CUDA, cuDNN, cuBLAS
│  (NVIDIA Driver 535+, CUDA 12.x)               │
├─────────────────────────────────────────────────┤
│  Layer 1: Operating System                       │  ← Ubuntu 22.04 LTS
│  (Linux kernel with GPU and IB support)          │
├─────────────────────────────────────────────────┤
│  Layer 0: Hardware                               │  ← GPUs, NVLink, InfiniBand
│  (H100 GPUs, NVSwitch, ConnectX-7, IB Switches) │
└─────────────────────────────────────────────────┘
```

### 2.2 Layer 1: Operating System Setup

Every server needs the same base operating system. Ubuntu 22.04 LTS is the standard choice.

```bash
# On EVERY server in the cluster:

# 1. Install Ubuntu 22.04 LTS Server (minimal install)
#    Use a provisioning tool like PXE boot, MAAS, or Foreman
#    to install the same image on all servers automatically.

# 2. Configure InfiniBand drivers (MLNX_OFED)
#    This is the driver package for ConnectX InfiniBand cards.
#    Download from NVIDIA (formerly Mellanox) website.
wget https://content.mellanox.com/ofed/MLNX_OFED-23.10/MLNX_OFED_LINUX-23.10-x.x.x-ubuntu22.04-x86_64.tgz
tar xzf MLNX_OFED_LINUX-*.tgz
cd MLNX_OFED_LINUX-*
./mlnxofedinstall --add-kernel-support

# 3. Verify InfiniBand is working
ibstat          # Shows status of each IB port (should say "Active")
ibping          # Ping another server over InfiniBand
ib_write_bw     # Benchmark InfiniBand bandwidth between two servers

# Expected output of ibstat:
#   Port 1:
#     State: Active       ← Good! Port is connected
#     Physical state: LinkUp
#     Rate: 400 Gb/s      ← NDR InfiniBand speed
```

### 2.3 Layer 2: GPU Drivers and CUDA

```bash
# On EVERY server:

# 1. Install NVIDIA GPU driver
#    The driver lets the operating system talk to the GPUs.
sudo apt-get install nvidia-driver-535

# 2. Verify GPUs are detected
nvidia-smi
# Should show all 8 GPUs with their temperatures, memory, etc.
# Example output:
# +--------+------+------+------+
# | GPU  0 | H100 | 80GB | 32°C |
# | GPU  1 | H100 | 80GB | 31°C |
# | ...    | ...  | ...  | ...  |
# | GPU  7 | H100 | 80GB | 33°C |
# +--------+------+------+------+

# 3. Install CUDA Toolkit
#    CUDA is the programming platform for NVIDIA GPUs.
#    It includes the compiler, libraries, and tools.
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sudo sh cuda_12.2.0_535.54.03_linux.run

# 4. Install cuDNN (deep learning library) and NCCL
sudo apt-get install libcudnn8 libnccl2 libnccl-dev

# 5. Verify NVLink is working
nvidia-smi nvlink -s
# Shows NVLink status for each GPU pair
# All links should show "Active"

# 6. Verify GPU-to-GPU bandwidth
# Run NVIDIA's bandwidth test
/usr/local/cuda/samples/1_Utilities/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest
# Should show ~800-900 GB/s between GPU pairs (NVSwitch)
```

### 2.4 Layer 3: Container Runtime

Kubernetes runs everything in containers. We need Docker/containerd plus NVIDIA's container toolkit (so containers can access GPUs).

```bash
# On EVERY server:

# 1. Install containerd (Kubernetes-preferred container runtime)
sudo apt-get install containerd

# 2. Install NVIDIA Container Toolkit
#    This is the bridge that lets containers see and use GPUs.
#    Without it, a container is "blind" to GPUs.
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install nvidia-container-toolkit

# 3. Configure containerd to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=containerd
sudo systemctl restart containerd

# 4. Test: Run a container that uses GPUs
sudo ctr run --rm --gpus 0 nvcr.io/nvidia/cuda:12.2.0-base-ubuntu22.04 test nvidia-smi
# Should show GPU info inside the container — proof that containers can access GPUs!
```

### 2.5 Layer 4: Kubernetes Installation

Now we install Kubernetes across all servers. One server becomes the "control plane" (the brain), and all others become "worker nodes" (the muscle).

```
Cluster layout:

  Server 0: Kubernetes Control Plane + Worker (or dedicated control plane)
  Server 1-N: Worker nodes (run GPU jobs)

  In small clusters (< 16 nodes): Control plane shares a server with workers
  In large clusters (16+ nodes): Dedicate 3 servers to control plane (for redundancy)
```

```bash
# ─────────────────────────────────────────────
# On ALL servers: Install Kubernetes components
# ─────────────────────────────────────────────

# 1. Install kubeadm, kubelet, kubectl
sudo apt-get install -y kubelet kubeadm kubectl
sudo apt-mark hold kubelet kubeadm kubectl  # Prevent auto-updates

# ─────────────────────────────────────────────
# On the CONTROL PLANE server (Server 0):
# ─────────────────────────────────────────────

# 2. Initialize the Kubernetes cluster
#    This creates the control plane (API server, scheduler, etcd)
sudo kubeadm init \
    --pod-network-cidr=10.244.0.0/16 \
    --control-plane-endpoint=server-0.example.com

# 3. Set up kubectl access
mkdir -p $HOME/.kube
sudo cp /etc/kubernetes/admin.conf $HOME/.kube/config

# 4. Install a network plugin (pods need networking)
#    Flannel is simple and works well
kubectl apply -f https://raw.githubusercontent.com/flannel-io/flannel/master/Documentation/kube-flannel.yml

# 5. Get the join command (you'll need this for worker nodes)
kubeadm token create --print-join-command
# Output: kubeadm join server-0:6443 --token abc123 --discovery-token-ca-cert-hash sha256:xyz789

# ─────────────────────────────────────────────
# On EACH WORKER server (Server 1, 2, 3, ...):
# ─────────────────────────────────────────────

# 6. Join the cluster (paste the command from step 5)
sudo kubeadm join server-0:6443 --token abc123 --discovery-token-ca-cert-hash sha256:xyz789

# ─────────────────────────────────────────────
# Back on CONTROL PLANE: Verify all nodes joined
# ─────────────────────────────────────────────

kubectl get nodes
# Output:
# NAME       STATUS   ROLES           AGE   VERSION
# server-0   Ready    control-plane   10m   v1.28.0
# server-1   Ready    <none>          5m    v1.28.0
# server-2   Ready    <none>          5m    v1.28.0
# server-3   Ready    <none>          5m    v1.28.0
# ...
```

### 2.6 Layer 4 (continued): NVIDIA GPU Operator

The GPU Operator automatically sets up everything GPUs need in Kubernetes. Instead of manually installing drivers and plugins on each node, the operator does it all:

```bash
# On the control plane server:

# 1. Install Helm (Kubernetes package manager)
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# 2. Add NVIDIA Helm repository
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

# 3. Install the GPU Operator
#    This single command installs:
#    - GPU device plugin (tells K8s about GPUs)
#    - DCGM exporter (GPU monitoring → Prometheus)
#    - GPU Feature Discovery (labels nodes with GPU type)
#    - MIG Manager (if using Multi-Instance GPU)
helm install gpu-operator nvidia/gpu-operator \
    --namespace gpu-operator \
    --create-namespace \
    --set driver.enabled=true \
    --set toolkit.enabled=true \
    --set dcgmExporter.enabled=true \
    --set migManager.enabled=false

# 4. Wait for all GPU Operator pods to be ready
kubectl -n gpu-operator get pods -w
# Wait until all pods show "Running" or "Completed"

# 5. Verify GPUs are visible to Kubernetes
kubectl describe node server-1 | grep nvidia.com/gpu
# Output:
#   nvidia.com/gpu:  8        ← Kubernetes sees 8 GPUs on this node!
#   nvidia.com/gpu:  8        ← 8 are allocatable
```

**What just happened?** The GPU Operator automatically:
1. Detected all NVIDIA GPUs on every node
2. Installed device plugins so Kubernetes knows "this node has 8 GPUs"
3. Started monitoring agents that track GPU temperature, utilization, memory
4. Labeled nodes with GPU type (e.g., `nvidia.com/gpu.product=NVIDIA-H100-SXM`)
5. Now when a pod requests `nvidia.com/gpu: 4`, Kubernetes knows which nodes have free GPUs

### 2.7 Configure InfiniBand for Kubernetes

By default, Kubernetes pods use a virtual overlay network (slow). For LLM training, we need pods to use the **real InfiniBand network** directly. There are two approaches:

**Approach A: Host Networking (Simpler)**

Pods share the host's network stack directly. No overlay network overhead.

```yaml
# In your training pod/job spec, add:
spec:
  hostNetwork: true      # Pod uses the host's network directly
  dnsPolicy: ClusterFirstWithHostNet
```

Pros: Simple, full InfiniBand performance, zero overhead
Cons: Pods share the host network (port conflicts possible), less isolation

**Approach B: RDMA Device Plugin (More Kubernetes-native)**

Use the `k8s-rdma-shared-dev-plugin` to expose InfiniBand devices to pods as resources:

```bash
# Install RDMA shared device plugin
kubectl apply -f https://raw.githubusercontent.com/Mellanox/k8s-rdma-shared-dev-plugin/master/deployment/rdma-shared-dev-plugin-ds.yaml
```

```yaml
# Then in your pod spec, request RDMA devices:
resources:
  limits:
    nvidia.com/gpu: 8
    rdma/rdma_shared_device_a: 1    # Request InfiniBand access
```

**Recommended: Use host networking** for training jobs (maximum performance). Use overlay networking for inference/serving (better isolation).

### 2.8 Layer 5: NCCL Configuration

NCCL doesn't need a separate installation — it comes with PyTorch and CUDA. But it needs to be **configured correctly** to use our InfiniBand network and NVLink topology.

**How NCCL decides how to send data:**

```
When GPU 0 on Server A needs to send data to GPU 3 on Server B,
NCCL automatically decides the best path:

Step 1: NCCL detects the hardware topology
  "GPU 0 is connected to NIC 0 via PCIe"
  "NIC 0 is an InfiniBand ConnectX-7"
  "GPU 3 on Server B is connected to NIC 3"

Step 2: NCCL picks the fastest transport
  Same GPU?      → Memory copy (instant)
  Same server?   → NVLink (900 GB/s)
  Different server? → InfiniBand via GPUDirect RDMA (50 GB/s per port)

Step 3: NCCL picks the best algorithm
  Small message (<256 KB)?  → Tree algorithm (low latency)
  Large message (>256 KB)?  → Ring algorithm (high bandwidth)
  InfiniBand with SHARP?    → CollNet (in-switch reduction)
```

**Critical NCCL environment variables for our cluster:**

```bash
# These go in your training container's environment or job spec

# ── Network Configuration ──
NCCL_IB_DISABLE=0                  # Enable InfiniBand (DO NOT disable it!)
NCCL_SOCKET_IFNAME=ib0             # Use InfiniBand interface, not Ethernet
NCCL_IB_HCA=mlx5                   # Use Mellanox ConnectX cards
NCCL_IB_GID_INDEX=3                # GID index for RoCE (if using RoCE instead of IB)
NCCL_NET_GDR_LEVEL=5               # Enable GPUDirect RDMA (GPU talks directly to NIC)

# ── Performance Tuning ──
NCCL_BUFFSIZE=16777216             # 16 MB communication buffer (larger = better for big messages)
NCCL_NTHREADS=512                  # Threads per NCCL kernel
NCCL_CROSS_NIC=1                   # Allow using multiple NICs per GPU (if topology allows)

# ── Debugging (turn on during setup, off during production) ──
NCCL_DEBUG=INFO                    # Print what NCCL is doing (topology, algorithm choices)
NCCL_DEBUG_SUBSYS=ALL              # Full detail

# Example NCCL_DEBUG output when starting training:
#   NCCL INFO Using network IB                    ← Good! Using InfiniBand
#   NCCL INFO Ring 00: 0[0]->1[1] via P2P/NVLink  ← Good! Using NVLink within server
#   NCCL INFO Ring 00: 7[7]->8[0] via NET/IB      ← Good! Using InfiniBand between servers
#   NCCL INFO Algorithm Ring protocol Simple       ← Ring algorithm for large data
```

**Verify NCCL performance with nccl-tests:**

```bash
# Build nccl-tests inside a container or on bare metal
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=1 CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr/local/nccl

# Test 1: Intra-node all-reduce (8 GPUs on 1 server)
# This tests NVLink performance
mpirun -np 8 ./build/all_reduce_perf -b 1M -e 4G -f 2 -g 1

# Expected results (H100 with NVSwitch):
#   Message Size    Bus Bandwidth
#   1 MB            ~200 GB/s
#   128 MB          ~400 GB/s
#   4 GB            ~450 GB/s     ← Should be close to NVLink theoretical max

# Test 2: Inter-node all-reduce (16 GPUs across 2 servers)
# This tests InfiniBand performance
mpirun -np 16 --hostfile hosts.txt \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_SOCKET_IFNAME=ib0 \
    ./build/all_reduce_perf -b 1M -e 4G -f 2 -g 1

# Expected results (8× NDR 400Gb InfiniBand per server):
#   Message Size    Bus Bandwidth
#   1 MB            ~50 GB/s
#   128 MB          ~150 GB/s
#   4 GB            ~200 GB/s     ← Should approach aggregate IB bandwidth

# If numbers are much lower than expected:
#   - Check NCCL_DEBUG output for "Using network TCP" (bad! Should be IB)
#   - Check GPUDirect RDMA is enabled (NCCL_NET_GDR_LEVEL=5)
#   - Check InfiniBand link status with ibstat
#   - Check for PCIe topology issues with nvidia-smi topo -m
```

### 2.9 Layer 6-7: ML Frameworks

```bash
# Inside your training container (Dockerfile):

# Start from NVIDIA's PyTorch container (has everything pre-installed)
FROM nvcr.io/nvidia/pytorch:24.01-py3

# This container already includes:
# - PyTorch with CUDA support
# - NCCL (optimized for this CUDA version)
# - cuDNN, cuBLAS
# - NVIDIA Apex (mixed precision)
# - Flash Attention

# Install additional LLM training libraries
pip install deepspeed                # Microsoft's distributed training library
pip install transformers datasets    # Hugging Face model/data libraries
pip install flash-attn               # Efficient attention implementation
pip install wandb                    # Experiment tracking (optional)

# For inference serving:
pip install vllm                     # High-throughput LLM serving engine
```

---

## Step 3: Set Up Kubernetes for LLM Workloads

### 3.1 Label Nodes by GPU Type

Kubernetes needs to know what kind of GPUs each node has, so it can schedule jobs to the right place:

```bash
# The GPU Operator auto-labels nodes, but you can add custom labels too:
kubectl label nodes server-1 gpu-type=h100
kubectl label nodes server-1 infiniband=true
kubectl label nodes server-1 nvlink=nvswitch

# Verify labels
kubectl get nodes --show-labels | grep gpu-type
```

### 3.2 Reserve GPU Nodes for GPU Workloads

Don't let random non-GPU jobs waste your expensive GPU servers:

```bash
# Taint all GPU nodes: "Only GPU workloads allowed here"
kubectl taint nodes server-1 nvidia.com/gpu=present:NoSchedule
kubectl taint nodes server-2 nvidia.com/gpu=present:NoSchedule
# ... repeat for all GPU nodes

# Now only pods with this toleration can run on GPU nodes:
# tolerations:
# - key: "nvidia.com/gpu"
#   operator: "Equal"
#   value: "present"
#   effect: "NoSchedule"
```

### 3.3 Install Kubeflow Training Operator

This teaches Kubernetes how to run distributed training jobs:

```bash
# Install Kubeflow Training Operator
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone"

# Verify it's running
kubectl -n kubeflow get pods
# Should show training-operator-xxx pod in Running state

# Now Kubernetes understands:
# - PyTorchJob   (distributed PyTorch training)
# - MPIJob       (MPI-based distributed training)
# - TFJob        (TensorFlow training)
```

### 3.4 Install Volcano Scheduler (for batch job queuing)

Standard Kubernetes scheduler doesn't understand HPC-style batch queuing. Volcano adds:
- Gang scheduling (all pods start together or none start)
- Fair-share queues (multiple teams share resources fairly)
- Priority preemption

```bash
# Install Volcano
kubectl apply -f https://raw.githubusercontent.com/volcano-sh/volcano/master/installer/volcano-development.yaml

# Create a GPU queue for training jobs
kubectl apply -f - <<EOF
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: gpu-training
spec:
  weight: 1
  reclaimable: true
  capability:
    nvidia.com/gpu: 64     # This queue can use up to 64 GPUs
EOF
```

### 3.5 Install Monitoring Stack

You need to see what's happening on your cluster — GPU utilization, temperatures, job status:

```bash
# Install Prometheus + Grafana using Helm
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install kube-prometheus prometheus-community/kube-prometheus-stack \
    --namespace monitoring \
    --create-namespace

# The GPU Operator's DCGM exporter automatically exposes GPU metrics
# Prometheus will scrape them and Grafana will display dashboards showing:
# - Per-GPU utilization (SM activity, Tensor Core activity)
# - GPU memory usage
# - GPU temperature and power draw
# - NVLink bandwidth
# - InfiniBand throughput
# - Training throughput (tokens/s — from your training code)
```

---

## Step 4: Run LLM Training on the Cluster

Now everything is set up. Let's train an LLM!

### 4.1 Distributed Training Job (PyTorchJob)

This example trains LLaMA 70B across 4 servers (32 GPUs):

```yaml
# file: llama-70b-training.yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: llama-70b-pretrain
  namespace: training
spec:
  # Elastic training: can scale between 4 and 8 nodes
  elasticPolicy:
    rdzvBackend: c10d
    minReplicas: 4
    maxReplicas: 8
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        metadata:
          labels:
            app: llm-training
        spec:
          hostNetwork: true                    # Use InfiniBand directly
          dnsPolicy: ClusterFirstWithHostNet
          tolerations:
          - key: "nvidia.com/gpu"
            operator: "Equal"
            value: "present"
            effect: "NoSchedule"
          affinity:
            nodeAffinity:
              requiredDuringSchedulingIgnoredDuringExecution:
                nodeSelectorTerms:
                - matchExpressions:
                  - key: nvidia.com/gpu.product
                    operator: In
                    values:
                    - NVIDIA-H100-SXM           # Only H100 nodes
          containers:
          - name: trainer
            image: my-registry/llm-trainer:v1
            command:
            - torchrun
            - --nproc_per_node=8                # 8 GPUs per server
            - --nnodes=4                        # 4 servers total
            - train.py
            - --model=meta-llama/Llama-3-70B
            - --tensor-parallel-size=8          # Split layers across 8 GPUs (intra-node)
            - --pipeline-parallel-size=2        # Split model into 2 pipeline stages
            - --data-parallel-size=2            # 2 data-parallel replicas
            - --micro-batch-size=2
            - --global-batch-size=256
            - --bf16                            # Use BF16 mixed precision
            - --flash-attention                 # Use Flash Attention
            - --activation-checkpointing        # Save memory by recomputing activations
            env:
            # NCCL configuration for InfiniBand
            - name: NCCL_IB_DISABLE
              value: "0"
            - name: NCCL_SOCKET_IFNAME
              value: "ib0"
            - name: NCCL_NET_GDR_LEVEL
              value: "5"
            - name: NCCL_IB_HCA
              value: "mlx5"
            - name: NCCL_DEBUG
              value: "WARN"                    # INFO for debugging, WARN for production
            resources:
              limits:
                nvidia.com/gpu: 8              # All 8 GPUs on this node
                memory: "1500Gi"               # Most of the node's RAM
                cpu: "180"                     # Most of the node's CPUs
              requests:
                nvidia.com/gpu: 8
                memory: "1500Gi"
                cpu: "180"
            volumeMounts:
            - name: training-data
              mountPath: /data
            - name: checkpoints
              mountPath: /checkpoints
            - name: shared-memory               # CRITICAL for PyTorch DataLoader
              mountPath: /dev/shm
          volumes:
          - name: training-data
            persistentVolumeClaim:
              claimName: training-data-pvc       # Lustre-backed PVC
          - name: checkpoints
            persistentVolumeClaim:
              claimName: checkpoints-pvc
          - name: shared-memory
            emptyDir:
              medium: Memory
              sizeLimit: "256Gi"                # Large shared memory for data loading
    Worker:
      replicas: 3                               # 3 workers + 1 master = 4 nodes = 32 GPUs
      template:
        # ... same spec as Master
```

**Understanding the parallelism configuration:**

```
Our 32 GPUs are organized as:

  4 servers × 8 GPUs = 32 GPUs total

  Tensor Parallel (TP) = 8:
    All 8 GPUs within a server work on the SAME layer together.
    Each GPU holds 1/8 of each layer's weight matrices.
    They communicate via NVLink (900 GB/s) — super fast!

  Pipeline Parallel (PP) = 2:
    The 80 layers of LLaMA 70B are split into 2 stages:
      Stage 0 (Layers 0-39):  Runs on Server 0 and Server 2
      Stage 1 (Layers 40-79): Runs on Server 1 and Server 3
    Activations flow between stages via InfiniBand.

  Data Parallel (DP) = 2:
    We have 2 complete copies of the pipeline:
      Copy A: Server 0 (stage 0) + Server 1 (stage 1)
      Copy B: Server 2 (stage 0) + Server 3 (stage 1)
    Each copy processes different training data.
    Gradients are synchronized via all-reduce over InfiniBand.

  Verification: TP × PP × DP = 8 × 2 × 2 = 32 GPUs ✓

  Visual layout:

  ┌─ Data Parallel Group A ─────────────────────┐
  │                                              │
  │  Server 0 (Pipeline Stage 0)                 │
  │  [GPU0│GPU1│GPU2│GPU3│GPU4│GPU5│GPU6│GPU7]   │
  │   ↑────── Tensor Parallel (NVLink) ──────↑   │
  │              │ activations (InfiniBand)       │
  │              ▼                               │
  │  Server 1 (Pipeline Stage 1)                 │
  │  [GPU0│GPU1│GPU2│GPU3│GPU4│GPU5│GPU6│GPU7]   │
  │                                              │
  └──────────────────┬───────────────────────────┘
                     │ gradient sync (InfiniBand all-reduce)
  ┌──────────────────┴───────────────────────────┐
  │                                              │
  │  Server 2 (Pipeline Stage 0)                 │
  │  [GPU0│GPU1│GPU2│GPU3│GPU4│GPU5│GPU6│GPU7]   │
  │              │ activations (InfiniBand)       │
  │              ▼                               │
  │  Server 3 (Pipeline Stage 1)                 │
  │  [GPU0│GPU1│GPU2│GPU3│GPU4│GPU5│GPU6│GPU7]   │
  │                                              │
  └─ Data Parallel Group B ─────────────────────┘
```

**Launch the training:**

```bash
# Submit the training job
kubectl apply -f llama-70b-training.yaml

# Watch the pods come up
kubectl -n training get pods -w
# NAME                          READY   STATUS    RESTARTS   AGE
# llama-70b-pretrain-master-0   1/1     Running   0          30s
# llama-70b-pretrain-worker-0   1/1     Running   0          28s
# llama-70b-pretrain-worker-1   1/1     Running   0          28s
# llama-70b-pretrain-worker-2   1/1     Running   0          27s

# Check training logs
kubectl -n training logs llama-70b-pretrain-master-0 -f
# Output:
# Step 1: loss=11.234, tokens/s=50000, MFU=42%
# Step 2: loss=10.891, tokens/s=52000, MFU=44%
# Step 3: loss=10.567, tokens/s=53000, MFU=45%
# ... (loss goes down, throughput goes up as training warms up)

# Check GPU utilization on a specific node
kubectl exec -n training llama-70b-pretrain-master-0 -- nvidia-smi
```

---

## Step 5: Run LLM Inference Serving

After training completes, deploy the model for users to query:

### 5.1 vLLM Inference Deployment

```yaml
# file: llm-serving.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-70b-serving
  namespace: inference
spec:
  replicas: 2                  # 2 replicas for high availability
  selector:
    matchLabels:
      app: llama-serving
  template:
    metadata:
      labels:
        app: llama-serving
    spec:
      tolerations:
      - key: "nvidia.com/gpu"
        operator: "Equal"
        value: "present"
        effect: "NoSchedule"
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
        - --model=/models/llama-70b          # Path to trained model
        - --tensor-parallel-size=8           # Use all 8 GPUs on the node
        - --max-model-len=4096               # Maximum sequence length
        - --gpu-memory-utilization=0.90      # Use 90% of GPU memory
        - --enable-chunked-prefill           # Better scheduling of requests
        ports:
        - containerPort: 8000
          name: http
        resources:
          limits:
            nvidia.com/gpu: 8
            memory: "512Gi"
          requests:
            nvidia.com/gpu: 8
            memory: "512Gi"
        volumeMounts:
        - name: model-storage
          mountPath: /models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-storage-pvc       # PVC pointing to trained model files
---
# Expose the service
apiVersion: v1
kind: Service
metadata:
  name: llama-serving-svc
  namespace: inference
spec:
  selector:
    app: llama-serving
  ports:
  - port: 8000
    targetPort: 8000
    name: http
  type: LoadBalancer                         # External access
---
# Auto-scale based on GPU utilization
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llama-hpa
  namespace: inference
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llama-70b-serving
  minReplicas: 2                             # Always have at least 2 running
  maxReplicas: 6                             # Scale up to 6 during peak traffic
  metrics:
  - type: Pods
    pods:
      metric:
        name: DCGM_FI_DEV_GPU_UTIL          # Scale based on GPU utilization
      target:
        type: AverageValue
        averageValue: "80"                   # Add more replicas when GPUs > 80% busy
```

```bash
# Deploy
kubectl apply -f llm-serving.yaml

# Test the API
curl http://<LOAD_BALANCER_IP>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-70b",
    "messages": [{"role": "user", "content": "What is machine learning?"}],
    "max_tokens": 200
  }'
```

**How autoscaling works:**

```
Low traffic (2 AM):
  2 replicas running (minimum)
  GPU utilization: 20%
  HPA: "GPUs are underutilized, no need to scale"

Peak traffic (2 PM):
  2 replicas running
  GPU utilization: 90% (above 80% threshold!)
  HPA: "GPUs are overloaded, scaling up!"
  → Creates replica 3, then 4
  GPU utilization drops to 55%
  HPA: "That's better, holding at 4 replicas"

Traffic drops (8 PM):
  4 replicas running
  GPU utilization: 30%
  HPA: "GPUs are mostly idle, scaling down"
  → Removes replicas back to 2
```

---

## Step 6: Monitor and Maintain the Cluster

### 6.1 What to Monitor

```
┌─────────────────────────────────────────────────────────────┐
│                    Cluster Health Dashboard                   │
│                                                              │
│  ── GPU Health ──────────────────────────────────────────    │
│  GPU Utilization (per node):    [████████░░] 82%             │
│  Tensor Core Activity:          [█████████░] 91%             │
│  GPU Memory Used:               72/80 GB per GPU             │
│  GPU Temperature:               71°C (OK, throttle at 83°C) │
│  Power Draw:                    650W / 700W TDP              │
│                                                              │
│  ── Network Health ─────────────────────────────────────     │
│  NVLink Bandwidth (intra-node): 850 / 900 GB/s (94%)        │
│  InfiniBand Bandwidth:          180 / 200 GB/s (90%)        │
│  IB Port Errors:                0 (good!)                    │
│  NCCL All-Reduce Time:          12 ms per step               │
│                                                              │
│  ── Training Progress ──────────────────────────────────     │
│  Tokens per second:             420,000                      │
│  MFU:                           52%                          │
│  Current Loss:                  2.34 (decreasing)            │
│  Step:                          45,231 / 500,000             │
│  ETA:                           ██████████████░░░░ 65%       │
│                                                              │
│  ── Cluster Resources ──────────────────────────────────     │
│  Total GPUs: 64    Used: 48    Free: 16                      │
│  Active Jobs: 3    Queued: 7                                 │
│  Storage Used: 45 TB / 100 TB (45%)                          │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Common Issues and How to Fix Them

| Problem | Symptom | How to Diagnose | Fix |
|---------|---------|-----------------|-----|
| GPU not detected | `nvidia.com/gpu: 0` in node description | `kubectl describe node`, check GPU Operator pods | Restart GPU Operator, check driver |
| InfiniBand down | Training hangs, NCCL timeout errors | `ibstat` on the node, check NCCL_DEBUG logs | Check cable, restart IB driver |
| NCCL using TCP instead of IB | Very slow training, NCCL logs say "Using network Socket" | Set `NCCL_DEBUG=INFO`, look for transport | Set `NCCL_IB_DISABLE=0`, `NCCL_SOCKET_IFNAME=ib0` |
| GPU thermal throttling | Performance drops after 30+ minutes | `nvidia-smi -q -d TEMPERATURE` | Check cooling, reduce power limit |
| OOM (Out of Memory) | Pod crashes with CUDA OOM | Check `torch.cuda.max_memory_allocated()` | Reduce batch size, enable activation checkpointing |
| Pod stuck in Pending | Job won't start | `kubectl describe pod <name>` | Check if GPUs are available, check taints/tolerations |
| Slow data loading | GPU utilization drops to 0% periodically | Profile with Nsight Systems, check CPU usage | More DataLoader workers, faster storage, prefetch |
| Checkpoint write is slow | Training pauses during checkpointing | Measure checkpoint time | Use async checkpointing, faster storage |

### 6.3 Alerting Rules

Set up Prometheus alerts for critical issues:

```yaml
# Prometheus alerting rules
groups:
- name: gpu-alerts
  rules:
  # Alert if GPU temperature is dangerously high
  - alert: GPUTemperatureCritical
    expr: DCGM_FI_DEV_GPU_TEMP > 83
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "GPU {{ $labels.gpu }} on {{ $labels.node }} is overheating ({{ $value }}°C)"

  # Alert if a GPU has errors
  - alert: GPUXidErrors
    expr: DCGM_FI_DEV_XID_ERRORS > 0
    for: 1m
    labels:
      severity: warning
    annotations:
      summary: "GPU {{ $labels.gpu }} reporting XID errors — may need replacement"

  # Alert if InfiniBand has errors
  - alert: InfiniBandErrors
    expr: node_infiniband_port_data_received_bytes_total == 0
    for: 10m
    labels:
      severity: critical
    annotations:
      summary: "InfiniBand port on {{ $labels.node }} has no traffic — link may be down"

  # Alert if training throughput drops significantly
  - alert: TrainingThroughputDrop
    expr: training_tokens_per_second < 300000
    for: 15m
    labels:
      severity: warning
    annotations:
      summary: "Training throughput dropped to {{ $value }} tokens/s (expected >400K)"
```

---

## Complete Architecture Summary

```
┌──────────────────────────────────────────────────────────────────┐
│                    COMPLETE HPC CLUSTER FOR LLMs                  │
│                                                                  │
│  ┌──── Kubernetes Control Plane ──────────────────────────────┐  │
│  │  API Server + Scheduler + etcd + Controller Manager        │  │
│  │  + GPU Operator + Kubeflow Training Operator               │  │
│  │  + Volcano Scheduler + Prometheus/Grafana                  │  │
│  └────────────────────────┬───────────────────────────────────┘  │
│                           │                                      │
│  ┌────────────────────────▼───────────────────────────────────┐  │
│  │              InfiniBand NDR Fat-Tree Network                │  │
│  │         (Quantum-2 switches, 400 Gb/s per port)            │  │
│  │                                                            │  │
│  │    ┌──────────────┐     ┌──────────────┐                   │  │
│  │    │ Core Switch  │     │ Core Switch  │                   │  │
│  │    └──┬────┬──┬───┘     └──┬────┬──┬───┘                   │  │
│  │       │    │  │            │    │  │                        │  │
│  │    ┌──▼──┐│┌─▼───┐     ┌──▼──┐│┌─▼───┐                    │  │
│  │    │Leaf ││ │Leaf │     │Leaf ││ │Leaf │                    │  │
│  │    │SW 0 ││ │SW 1 │     │SW 2 ││ │SW 3 │                    │  │
│  │    └┬┬┬┬─┘│ └┬┬┬┬─┘     └┬┬┬┬─┘│ └┬┬┬┬─┘                    │  │
│  └─────┼┼┼┼──┼──┼┼┼┼──────┼┼┼┼──┼──┼┼┼┼──────────────────────┘  │
│        ││││  │  ││││      ││││  │  ││││                          │
│  ┌─────▼▼▼▼──┘  ▼▼▼▼──┐  ┌▼▼▼▼──┘  ▼▼▼▼──┐                    │
│  │  GPU Server 0-3     │  │  GPU Server 4-7 │                    │
│  │                     │  │                 │    ×N servers       │
│  │  Per server:        │  │  Per server:    │                    │
│  │  • 8× H100 SXM GPU │  │  • 8× H100 GPU │                    │
│  │  • NVSwitch 900GB/s │  │  • NVSwitch     │                    │
│  │  • 8× ConnectX-7 IB │  │  • 8× CX-7 IB  │                    │
│  │  • 2× EPYC CPU     │  │  • 2× EPYC CPU  │                    │
│  │  • 2 TB RAM         │  │  • 2 TB RAM     │                    │
│  │  • 30 TB NVMe SSD  │  │  • 30 TB NVMe   │                    │
│  └─────────────────────┘  └─────────────────┘                    │
│        │                        │                                │
│  ┌─────▼────────────────────────▼──────────────────────────────┐ │
│  │              Shared Storage (Lustre / GPFS)                  │ │
│  │         Multiple storage servers with HDDs/SSDs              │ │
│  │         Training data, model checkpoints, logs               │ │
│  │         Capacity: 100+ TB, Throughput: 100+ GB/s            │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌── Software Stack (on every GPU server) ─────────────────────┐ │
│  │  Ubuntu 22.04 → NVIDIA Driver → CUDA 12 → cuDNN → NCCL    │ │
│  │  → containerd → NVIDIA Container Toolkit → kubelet          │ │
│  │  → PyTorch → DeepSpeed/Megatron-LM → Your training code    │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

1. **GPUs are the workers, NVLink is the fast internal highway, InfiniBand is the fast external highway, NCCL is the traffic controller, and Kubernetes is the job assignment office.** Each piece has a specific role, and they all work together.

2. **NVLink (intra-node) is ~4-18× faster than InfiniBand (inter-node).** This is why tensor parallelism (heavy communication) goes within a server, and data/pipeline parallelism (lighter communication) goes across servers.

3. **NCCL automatically figures out the fastest path.** You configure the environment variables to tell NCCL what hardware exists, and it handles the rest — choosing NVLink vs InfiniBand, Ring vs Tree algorithm, etc.

4. **Kubernetes needs extra components for GPU workloads:** GPU Operator (hardware), Kubeflow Training Operator (distributed training), Volcano (batch scheduling), and host networking (InfiniBand access).

5. **Always verify each layer works before building the next:** First confirm GPUs work (`nvidia-smi`), then NVLink (`p2pBandwidthLatencyTest`), then InfiniBand (`ibstat`, `ib_write_bw`), then NCCL (`nccl-tests`), then Kubernetes (`kubectl get nodes`), then run a small training test before the real job.

6. **For production, use both SLURM and Kubernetes:** Many organizations run SLURM for large-scale training (maximum performance on bare metal) and Kubernetes for inference serving (autoscaling, API management). The guide above uses Kubernetes for both, which works well for medium-scale clusters but sacrifices some training performance due to container/network overhead.

7. **Monitor everything.** GPU temperature, utilization, InfiniBand errors, training loss, tokens/s — set up dashboards and alerts so you catch problems before they waste days of training.
