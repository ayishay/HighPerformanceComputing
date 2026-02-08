# Complete Guide: Building an HPC System for LLM Training and Inference
## Using GPU + NVLink + InfiniBand + NCCL + Kubernetes

---

## Overview

This guide walks you through building a production-grade High Performance Computing cluster specifically designed for training and serving Large Language Models. We'll integrate:

- **GPU nodes with NVLink** — Fast intra-node GPU communication
- **InfiniBand networking** — High-speed inter-node connectivity
- **NCCL** — NVIDIA's optimized GPU communication library (note: NCCL is a communication library, not an accelerator)
- **Kubernetes** — Container orchestration and resource scheduling

**Important clarification:** NCCL (NVIDIA Collective Communications Library) is not an accelerator — it's a **communication library** that enables efficient data transfer between GPUs. The actual accelerators are the GPUs themselves (e.g., NVIDIA H100, A100).

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Kubernetes Control Plane                       │
│              (API Server, Scheduler, Controllers)                 │
└───────────────────────────┬──────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼──────────┐ ┌──────▼─────────┐ ┌──────▼─────────┐
│   Worker Node 0   │ │  Worker Node 1  │ │  Worker Node N  │
│                   │ │                 │ │                 │
│  ┌─────────────┐  │ │  ┌───────────┐  │ │  ┌───────────┐  │
│  │8x H100 GPUs │  │ │  │8x H100 GPUs│  │ │  │8x H100 GPUs│  │
│  │  Connected  │  │ │  │ Connected  │  │ │  │ Connected  │  │
│  │via NVSwitch │  │ │  │via NVSwitch│  │ │  │via NVSwitch│  │
│  │             │  │ │  │            │  │ │  │            │  │
│  │ 900 GB/s    │  │ │  │ 900 GB/s   │  │ │  │ 900 GB/s   │  │
│  └─────────────┘  │ │  └───────────┘  │ │  └───────────┘  │
│                   │ │                 │ │                 │
│  ┌─────────────┐  │ │  ┌───────────┐  │ │  ┌───────────┐  │
│  │8x InfiniBand│  │ │  │8x InfiniBand│ │ │  │8x InfiniBand│ │
│  │   NICs      │  │ │  │   NICs     │  │ │  │   NICs     │  │
│  │(ConnectX-7) │  │ │  │(ConnectX-7)│  │ │  │(ConnectX-7)│  │
│  └──────┬──────┘  │ │  └─────┬─────┘  │ │  └─────┬─────┘  │
└─────────┼─────────┘ └────────┼────────┘ └────────┼────────┘
          │                    │                   │
          └────────────────────┼───────────────────┘
                               │
                ┌──────────────▼────────────────┐
                │   InfiniBand Switch Fabric     │
                │     (400 Gb/s NDR per port)    │
                │   Non-blocking, full bisection │
                └────────────────────────────────┘
                               │
                ┌──────────────▼────────────────┐
                │   Shared Storage System        │
                │   (Lustre / Ceph / NFS)        │
                │   100+ TB for datasets/models  │
                └────────────────────────────────┘
```

---

## Part 1: Hardware Components

### 1.1 GPU Compute Nodes

**Recommended configuration per node:**

| Component | Specification | Purpose |
|-----------|--------------|---------|
| GPUs | 8x NVIDIA H100 SXM (80GB each) | Primary compute |
| NVLink/NVSwitch | 900 GB/s all-to-all bandwidth | Intra-node GPU communication |
| CPUs | 2x Intel Xeon Platinum or AMD EPYC (64+ cores) | Host processing, data prep |
| RAM | 2TB DDR5 | CPU memory for data loading |
| Local storage | 4x 7.68TB NVMe SSD in RAID | Fast scratch space |
| Network | 8x NVIDIA ConnectX-7 (400Gb InfiniBand) | Inter-node communication |
| Form factor | NVIDIA DGX H100 or equivalent | Integrated solution |

**Why these choices:**
- **8 GPUs per node** — Optimal for tensor parallelism (split layers across all 8)
- **NVSwitch** — Full non-blocking GPU-to-GPU connectivity within node
- **8 InfiniBand NICs** — One per GPU for GPUDirect RDMA
- **Large RAM** — Prevents CPU bottleneck during data preprocessing

### 1.2 InfiniBand Network Fabric

**Network topology:** Fat-tree or Rail-optimized

```
                    ┌─────────────────┐
                    │  Core Switches  │
                    │   (Tier 3)      │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
        ┌─────▼────┐   ┌─────▼────┐   ┌────▼─────┐
        │  Leaf    │   │  Leaf    │   │  Leaf    │
        │ Switch 0 │   │ Switch 1 │   │ Switch N │
        │ (Tier 2) │   │ (Tier 2) │   │ (Tier 2) │
        └────┬─────┘   └────┬─────┘   └────┬─────┘
             │              │              │
    ┌────────┼────┐  ┌──────┼──────┐  ┌────┼─────┐
  Node0   Node1  Node2  Node3    NodeN-1  NodeN
  8xGPU   8xGPU  8xGPU  8xGPU    8xGPU    8xGPU
```

**Network specifications:**
- **Speed:** 400 Gb/s NDR InfiniBand (latest generation)
- **Latency:** Sub-microsecond (< 1 μs node-to-node)
- **Topology:** 2:1 or 1:1 oversubscription (1:1 for training)
- **Protocol:** InfiniBand with GPUDirect RDMA support
- **Switch ASICs:** NVIDIA Quantum-2 or Mellanox switches

**Cable types:**
- **Intra-rack:** Copper DAC (Direct Attach Copper) for short distances
- **Inter-rack:** Optical fiber (AOC/transceivers) for longer runs

### 1.3 Storage System

**Parallel file system for datasets:**
- **Lustre** (most common in HPC)
- **GPFS/Spectrum Scale** (IBM solution)
- **BeeGFS** (open source alternative)
- **Ceph with CephFS** (cloud-native option)

**Storage requirements:**
- **Capacity:** 100TB - 10PB depending on scale
- **Bandwidth:** 100+ GB/s aggregate read throughput
- **IOPS:** 1M+ for metadata operations
- **Access pattern:** Shared across all nodes, POSIX-compliant

---

## Part 2: Software Stack

### 2.1 Base Operating System

**Recommended:** Ubuntu 22.04 LTS or Rocky Linux 9 (RHEL-compatible)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y build-essential linux-headers-$(uname -r) \
    git wget curl vim htop nvtop iftop \
    net-tools rdma-core libibverbs-dev ibverbs-utils \
    perftest infiniband-diags
```

### 2.2 NVIDIA GPU Drivers and CUDA

**Install NVIDIA drivers:**

```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sudo tee /etc/apt/sources.list.d/libnvidia-container.list

# Install driver (535+ recommended for H100)
sudo apt update
sudo apt install -y nvidia-driver-535 nvidia-utils-535

# Reboot
sudo reboot

# Verify
nvidia-smi
```

**Install CUDA Toolkit:**

```bash
# Download CUDA 12.2+ (required for H100)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-2

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
```

### 2.3 NCCL Installation

NCCL is the critical communication library for multi-GPU operations.

```bash
# Method 1: Install via apt (easiest)
sudo apt install -y libnccl2 libnccl-dev

# Method 2: Build from source (for latest version)
git clone https://github.com/NVIDIA/nccl.git
cd nccl
make -j src.build CUDA_HOME=/usr/local/cuda
sudo make install

# Verify installation
ls /usr/local/cuda/lib64/libnccl*
# Should show: libnccl.so, libnccl_static.a

# Test NCCL
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=1 CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr/local/cuda
```

**NCCL environment variables for InfiniBand:**

```bash
# Create NCCL config file
cat > ~/nccl_env.sh <<'EOF'
# Enable InfiniBand
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0

# InfiniBand HCA selection
export NCCL_IB_HCA=mlx5

# Enable GPUDirect RDMA
export NCCL_NET_GDR_LEVEL=5
export NCCL_NET_GDR_READ=1

# Use all InfiniBand ports
export NCCL_IB_GID_INDEX=3

# Algorithm selection (Ring for large messages)
export NCCL_ALGO=Ring

# Protocol (Simple for bulk transfer)
export NCCL_PROTO=Simple

# Performance tuning
export NCCL_BUFFSIZE=8388608
export NCCL_CROSS_NIC=1
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=4

# Debugging (set to WARN or INFO in production)
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

EOF

# Source it in training scripts
source ~/nccl_env.sh
```

### 2.4 InfiniBand Drivers (MLNX_OFED)

```bash
# Download Mellanox OFED (required for GPUDirect RDMA)
wget https://content.mellanox.com/ofed/MLNX_OFED-23.10-1.1.9.0/MLNX_OFED_LINUX-23.10-1.1.9.0-ubuntu22.04-x86_64.tgz

tar -xzf MLNX_OFED_LINUX-23.10-1.1.9.0-ubuntu22.04-x86_64.tgz
cd MLNX_OFED_LINUX-23.10-1.1.9.0-ubuntu22.04-x86_64

# Install with GPU support
sudo ./mlnxofedinstall --with-nfsrdma --with-nvmf --add-kernel-support --force

# Start drivers
sudo /etc/init.d/openibd restart

# Verify InfiniBand
ibstat
ibstatus

# Should show link UP, rate 400 Gb/s
```

**Verify GPUDirect RDMA:**

```bash
# Check if NVIDIA Peer Memory module is loaded
lsmod | grep nv_peer_mem

# If not loaded:
sudo modprobe nv_peer_mem

# Test IB bandwidth
ib_write_bw -a -d mlx5_0 -x 0  # On receiver node
ib_write_bw -a -d mlx5_0 -x 0 <receiver_ip>  # On sender node

# Should show close to 50 GB/s (400 Gb/s)
```

---

## Part 3: Kubernetes Cluster Setup

### 3.1 Install Container Runtime (containerd)

```bash
# Install containerd
sudo apt install -y containerd

# Configure containerd for NVIDIA runtime
sudo mkdir -p /etc/containerd
containerd config default | sudo tee /etc/containerd/config.toml

# Edit config to enable systemd cgroup
sudo sed -i 's/SystemdCgroup = false/SystemdCgroup = true/' /etc/containerd/config.toml

# Restart
sudo systemctl restart containerd
sudo systemctl enable containerd
```

**Install NVIDIA Container Toolkit:**

```bash
# Add NVIDIA container toolkit repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt update
sudo apt install -y nvidia-container-toolkit

# Configure containerd to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=containerd

# Restart containerd
sudo systemctl restart containerd
```

### 3.2 Install Kubernetes

**On all nodes (control plane + workers):**

```bash
# Disable swap (required by Kubernetes)
sudo swapoff -a
sudo sed -i '/ swap / s/^/#/' /etc/fstab

# Load kernel modules
cat <<EOF | sudo tee /etc/modules-load.d/k8s.conf
overlay
br_netfilter
EOF

sudo modprobe overlay
sudo modprobe br_netfilter

# Configure sysctl
cat <<EOF | sudo tee /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
EOF

sudo sysctl --system

# Install kubeadm, kubelet, kubectl
sudo apt update
sudo apt install -y apt-transport-https ca-certificates curl

curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.28/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg

echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.28/deb/ /' | sudo tee /etc/apt/sources.list.d/kubernetes.list

sudo apt update
sudo apt install -y kubelet kubeadm kubectl
sudo apt-mark hold kubelet kubeadm kubectl
```

**Initialize control plane (master node):**

```bash
# Initialize cluster with pod network CIDR
sudo kubeadm init --pod-network-cidr=10.244.0.0/16 \
    --service-cidr=10.96.0.0/12 \
    --kubernetes-version=v1.28.0

# Save the join command shown in output

# Configure kubectl for regular user
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

# Verify
kubectl get nodes
```

**Install CNI Plugin (Calico for InfiniBand support):**

```bash
# Install Calico (good for InfiniBand environments)
kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.26.1/manifests/tigera-operator.yaml
kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.26.1/manifests/custom-resources.yaml

# Wait for pods to be ready
kubectl get pods -n calico-system -w

# Alternative: Use host networking for training pods (bypasses CNI overhead)
```

**Join worker nodes:**

```bash
# On each worker node, run the join command from kubeadm init:
sudo kubeadm join <control-plane-ip>:6443 --token <token> \
    --discovery-token-ca-cert-hash sha256:<hash>

# Verify from control plane
kubectl get nodes
# All nodes should show Ready
```

### 3.3 Install NVIDIA GPU Operator

The GPU Operator automates GPU software stack deployment.

```bash
# Add NVIDIA Helm repository
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

# Install GPU Operator
helm install --wait --generate-name \
    -n gpu-operator --create-namespace \
    nvidia/gpu-operator \
    --set driver.enabled=false \
    --set toolkit.enabled=true \
    --set devicePlugin.enabled=true \
    --set dcgmExporter.enabled=true \
    --set gfd.enabled=true

# Note: driver.enabled=false because we installed drivers manually
# If you want operator to manage drivers, set driver.enabled=true

# Verify GPU Operator pods
kubectl get pods -n gpu-operator

# Check if GPUs are detected
kubectl get nodes -o=custom-columns=NAME:.metadata.name,GPUs:.status.capacity.'nvidia\.com/gpu'
```

**Test GPU access in a pod:**

```bash
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
spec:
  restartPolicy: OnFailure
  containers:
  - name: cuda-container
    image: nvcr.io/nvidia/cuda:12.2.0-base-ubuntu22.04
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
EOF

# Check logs
kubectl logs gpu-test
# Should show GPU info from nvidia-smi
```

### 3.4 Enable InfiniBand in Kubernetes

**Install RDMA Device Plugin:**

```bash
# Clone RDMA device plugin
git clone https://github.com/Mellanox/k8s-rdma-shared-dev-plugin.git
cd k8s-rdma-shared-dev-plugin

# Deploy
kubectl create -f deployments/rdma-shared-dev-plugin.yaml

# Verify
kubectl get pods -n kube-system | grep rdma

# Check RDMA resources
kubectl get nodes -o json | jq '.items[].status.allocatable'
# Should show: "rdma/rdma_shared_device_a": "1000"
```

**Configure pods to use InfiniBand:**

```yaml
resources:
  limits:
    nvidia.com/gpu: 8
    rdma/rdma_shared_device_a: 1  # Request InfiniBand access
  requests:
    nvidia.com/gpu: 8
    rdma/rdma_shared_device_a: 1
```

### 3.5 Install Kubeflow Training Operator

This enables distributed training jobs (PyTorchJob, MPIJob).

```bash
# Install Training Operator
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.7.0"

# Verify
kubectl get pods -n kubeflow
kubectl get crd | grep kubeflow
# Should show: pytorchjobs.kubeflow.org, mpijobs.kubeflow.org
```

---

## Part 4: Deploy LLM Training Workload

### 4.1 Build Training Container

**Dockerfile:**

```dockerfile
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Install additional dependencies
RUN pip install --no-cache-dir \
    transformers \
    datasets \
    accelerate \
    deepspeed \
    wandb \
    sentencepiece \
    tokenizers

# Install flash-attention for efficiency
RUN pip install flash-attn --no-build-isolation

# Copy training script
COPY train.py /workspace/train.py
COPY dataset_config.yaml /workspace/dataset_config.yaml

WORKDIR /workspace

# NCCL environment variables
ENV NCCL_IB_DISABLE=0
ENV NCCL_SOCKET_IFNAME=ib0
ENV NCCL_IB_HCA=mlx5
ENV NCCL_NET_GDR_LEVEL=5
ENV NCCL_DEBUG=INFO

CMD ["python", "train.py"]
```

**Build and push:**

```bash
docker build -t myregistry.io/llm-trainer:v1 .
docker push myregistry.io/llm-trainer:v1
```

### 4.2 Create Persistent Volume for Data

**Storage class for shared filesystem:**

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: training-data-pv
spec:
  capacity:
    storage: 10Ti
  accessModes:
    - ReadOnlyMany
  nfs:
    server: nfs-server.example.com
    path: /mnt/datasets/pile
  mountOptions:
    - vers=4.1
    - rsize=1048576
    - wsize=1048576
    - hard
    - timeo=600
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: training-data-pvc
  namespace: training
spec:
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 10Ti
  volumeName: training-data-pv
```

### 4.3 Submit Distributed Training Job

**PyTorchJob for multi-node LLM training:**

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: llama-70b-training
  namespace: training
spec:
  elasticPolicy:
    rdzvBackend: c10d
    minReplicas: 8
    maxReplicas: 8
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        metadata:
          annotations:
            # Use host networking for maximum InfiniBand performance
            k8s.v1.cni.cncf.io/networks: ib-network
        spec:
          hostNetwork: true  # Critical for InfiniBand performance
          dnsPolicy: ClusterFirstWithHostNet
          containers:
          - name: pytorch
            image: myregistry.io/llm-trainer:v1
            command:
              - torchrun
              - --nproc_per_node=8
              - --nnodes=8
              - --rdzv_backend=c10d
              - --rdzv_endpoint=$(MASTER_ADDR):29500
              - train.py
              - --model=meta-llama/Llama-3-70b
              - --dataset=/data/pile
              - --batch-size=4
              - --gradient-accumulation=8
              - --bf16
              - --tensor-parallel-size=8
              - --pipeline-parallel-size=2
              - --data-parallel-size=4
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
              value: "INFO"
            - name: MASTER_ADDR
              value: "llama-70b-training-master-0"
            - name: MASTER_PORT
              value: "29500"
            resources:
              limits:
                nvidia.com/gpu: 8
                rdma/rdma_shared_device_a: 1
                memory: "1024Gi"
                cpu: "128"
              requests:
                nvidia.com/gpu: 8
                rdma/rdma_shared_device_a: 1
                memory: "1024Gi"
                cpu: "128"
            volumeMounts:
            - name: training-data
              mountPath: /data
              readOnly: true
            - name: output
              mountPath: /workspace/output
            - name: shared-memory
              mountPath: /dev/shm
          volumes:
          - name: training-data
            persistentVolumeClaim:
              claimName: training-data-pvc
          - name: output
            nfs:
              server: nfs-server.example.com
              path: /mnt/outputs/llama-70b
          - name: shared-memory
            emptyDir:
              medium: Memory
              sizeLimit: "256Gi"
          nodeSelector:
            node.kubernetes.io/instance-type: dgx-h100
            nvidia.com/gpu.product: NVIDIA-H100-SXM
          tolerations:
          - key: "nvidia.com/gpu"
            operator: "Exists"
            effect: "NoSchedule"
    Worker:
      replicas: 7  # 7 workers + 1 master = 8 nodes total
      template:
        metadata:
          annotations:
            k8s.v1.cni.cncf.io/networks: ib-network
        spec:
          hostNetwork: true
          dnsPolicy: ClusterFirstWithHostNet
          containers:
          - name: pytorch
            image: myregistry.io/llm-trainer:v1
            # Same config as master
            command:
              - torchrun
              - --nproc_per_node=8
              - --nnodes=8
              - --rdzv_backend=c10d
              - --rdzv_endpoint=$(MASTER_ADDR):29500
              - train.py
              - --model=meta-llama/Llama-3-70b
              - --dataset=/data/pile
              - --batch-size=4
              - --gradient-accumulation=8
              - --bf16
              - --tensor-parallel-size=8
              - --pipeline-parallel-size=2
              - --data-parallel-size=4
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
              value: "INFO"
            - name: MASTER_ADDR
              value: "llama-70b-training-master-0"
            - name: MASTER_PORT
              value: "29500"
            resources:
              limits:
                nvidia.com/gpu: 8
                rdma/rdma_shared_device_a: 1
                memory: "1024Gi"
                cpu: "128"
              requests:
                nvidia.com/gpu: 8
                rdma/rdma_shared_device_a: 1
                memory: "1024Gi"
                cpu: "128"
            volumeMounts:
            - name: training-data
              mountPath: /data
              readOnly: true
            - name: output
              mountPath: /workspace/output
            - name: shared-memory
              mountPath: /dev/shm
          volumes:
          - name: training-data
            persistentVolumeClaim:
              claimName: training-data-pvc
          - name: output
            nfs:
              server: nfs-server.example.com
              path: /mnt/outputs/llama-70b
          - name: shared-memory
            emptyDir:
              medium: Memory
              sizeLimit: "256Gi"
          nodeSelector:
            node.kubernetes.io/instance-type: dgx-h100
            nvidia.com/gpu.product: NVIDIA-H100-SXM
          tolerations:
          - key: "nvidia.com/gpu"
            operator: "Exists"
            effect: "NoSchedule"
```

**Submit job:**

```bash
kubectl create namespace training
kubectl apply -f llama-training-job.yaml

# Monitor job
kubectl get pytorchjobs -n training
kubectl get pods -n training -w

# Check logs from master
kubectl logs -n training llama-70b-training-master-0 -f

# Check GPU utilization
kubectl exec -n training llama-70b-training-master-0 -- nvidia-smi
```

---

## Part 5: Deploy LLM Inference Service

### 5.1 Inference Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-inference
  namespace: inference
spec:
  replicas: 3  # 3 replicas for high availability
  selector:
    matchLabels:
      app: llama-inference
  template:
    metadata:
      labels:
        app: llama-inference
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
        - --model=/models/llama-3-70b
        - --tensor-parallel-size=8
        - --pipeline-parallel-size=1
        - --max-model-len=4096
        - --gpu-memory-utilization=0.95
        - --dtype=bfloat16
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: NCCL_IB_DISABLE
          value: "0"
        - name: NCCL_SOCKET_IFNAME
          value: "ib0"
        resources:
          limits:
            nvidia.com/gpu: 8
            rdma/rdma_shared_device_a: 1
          requests:
            nvidia.com/gpu: 8
            rdma/rdma_shared_device_a: 1
        volumeMounts:
        - name: model-storage
          mountPath: /models
          readOnly: true
      volumes:
      - name: model-storage
        nfs:
          server: nfs-server.example.com
          path: /mnt/models/llama-3-70b
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-H100-SXM
---
apiVersion: v1
kind: Service
metadata:
  name: llama-inference-svc
  namespace: inference
spec:
  type: LoadBalancer
  selector:
    app: llama-inference
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llama-inference-hpa
  namespace: inference
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llama-inference
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: vllm_queue_depth
      target:
        type: AverageValue
        averageValue: "10"
  - type: Resource
    resource:
      name: nvidia.com/gpu
      target:
        type: Utilization
        averageUtilization: 80
```

---

## Part 6: Monitoring and Observability

### 6.1 Deploy Prometheus + Grafana

```bash
# Add Prometheus Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus + Grafana
helm install prometheus prometheus-community/kube-prometheus-stack \
    -n monitoring --create-namespace \
    --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false

# Access Grafana
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80

# Default credentials: admin / prom-operator
```

### 6.2 NVIDIA DCGM Exporter

Already installed by GPU Operator. Metrics available at:

```
dcgm_gpu_utilization
dcgm_gpu_memory_used
dcgm_gpu_temp
dcgm_pcie_tx_throughput
dcgm_pcie_rx_throughput
dcgm_nvlink_bandwidth_total
```

**Import Grafana dashboard:**
- Dashboard ID: 12239 (NVIDIA DCGM Exporter)

### 6.3 NCCL Bandwidth Testing

```bash
# Create test pod
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: nccl-test
spec:
  hostNetwork: true
  containers:
  - name: nccl
    image: nvcr.io/nvidia/pytorch:23.10-py3
    command: ["/bin/bash", "-c", "sleep infinity"]
    resources:
      limits:
        nvidia.com/gpu: 8
        rdma/rdma_shared_device_a: 1
    env:
    - name: NCCL_IB_DISABLE
      value: "0"
    - name: NCCL_DEBUG
      value: "INFO"
EOF

# Exec into pod
kubectl exec -it nccl-test -- bash

# Run NCCL all-reduce test
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=1 CUDA_HOME=/usr/local/cuda
./build/all_reduce_perf -b 1G -e 8G -f 2 -g 8

# Expected bandwidth:
# Intra-node (NVLink): ~800-900 GB/s (busbw)
# Inter-node (InfiniBand): ~40-50 GB/s per GPU pair
```

---

## Part 7: Best Practices and Optimizations

### 7.1 Node Labeling for GPU Types

```bash
# Label nodes with GPU types
kubectl label nodes dgx-h100-01 nvidia.com/gpu.product=NVIDIA-H100-SXM
kubectl label nodes dgx-h100-01 node.kubernetes.io/instance-type=dgx-h100

# Label nodes with InfiniBand topology
kubectl label nodes dgx-h100-01 ib-rail=rail-0
kubectl label nodes dgx-h100-02 ib-rail=rail-0
kubectl label nodes dgx-h100-03 ib-rail=rail-1
kubectl label nodes dgx-h100-04 ib-rail=rail-1

# Use in pod affinity to keep jobs on same rail
```

### 7.2 Topology-Aware Scheduling

```yaml
# Ensure pods land on nodes with low-latency connections
affinity:
  podAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
    - labelSelector:
        matchExpressions:
        - key: job-name
          operator: In
          values:
          - llama-70b-training
      topologyKey: ib-rail
```

### 7.3 Shared Memory Configuration

LLM training needs large /dev/shm for NCCL:

```yaml
volumes:
- name: shared-memory
  emptyDir:
    medium: Memory
    sizeLimit: "256Gi"  # At least 128Gi for large models
```

### 7.4 NCCL Tuning per Cluster

Benchmark and tune NCCL variables:

```bash
# For InfiniBand clusters
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3  # Use all HCAs
export NCCL_IB_GID_INDEX=3
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_NET_GDR_LEVEL=5

# For large-scale training (64+ GPUs)
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_BUFFSIZE=8388608

# If using SHARP (switch-assisted all-reduce)
export NCCL_COLLNET_ENABLE=1
export NCCL_SHARP_DISABLE=0
```

### 7.5 Resource Quotas per Namespace

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: training
spec:
  hard:
    requests.nvidia.com/gpu: "64"  # Max 64 GPUs in this namespace
    limits.nvidia.com/gpu: "64"
    pods: "20"
```

---

## Part 8: Troubleshooting

### 8.1 NCCL Communication Failures

**Symptom:** Training hangs or crashes with NCCL errors

**Debug:**

```bash
# Check NCCL debug output
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,COLL

# Common issues:
# 1. InfiniBand not detected
ibstat  # Should show link ACTIVE

# 2. GPUDirect RDMA not working
lsmod | grep nv_peer_mem  # Should be loaded

# 3. Wrong network interface
export NCCL_SOCKET_IFNAME=ib0  # Set correct IB interface

# 4. Firewall blocking
sudo ufw allow from <node-subnet>  # Allow inter-node traffic
```

### 8.2 GPU Not Detected in Kubernetes

```bash
# Check GPU Operator logs
kubectl logs -n gpu-operator -l app=nvidia-device-plugin-daemonset

# Verify nvidia-smi works on host
nvidia-smi

# Check containerd runtime
sudo nvidia-ctk runtime configure --runtime=containerd
sudo systemctl restart containerd
```

### 8.3 InfiniBand Performance Issues

```bash
# Test IB bandwidth between nodes
# On receiver:
ib_write_bw -d mlx5_0 -a

# On sender:
ib_write_bw -d mlx5_0 -a <receiver_ip>

# Should show 45-50 GB/s for 400Gb InfiniBand

# Check for packet loss
ibdiagnet

# Check for link errors
ibdiagnet -r

# Monitor IB counters
ibqueryerrors
```

---

## Part 9: Cost and Scaling Considerations

### 9.1 Hardware Costs (Approximate)

| Component | Unit Cost | Quantity | Total |
|-----------|-----------|----------|-------|
| DGX H100 node (8x H100) | $300,000 | 8 nodes | $2,400,000 |
| InfiniBand switch (400Gb, 40-port) | $150,000 | 2 | $300,000 |
| Storage system (Lustre, 1PB) | $500,000 | 1 | $500,000 |
| Networking cables/optics | $50,000 | - | $50,000 |
| **Total** | | | **~$3,250,000** |

**Alternative: Cloud-based:**
- AWS p5.48xlarge (8x H100): ~$98/hour
- 8 nodes for 1 week training: $1.3M
- Break-even: ~25 weeks of continuous use

### 9.2 Scaling Beyond 8 Nodes

For 64+ node clusters:

1. **Multi-rail InfiniBand** — Multiple IB fabrics for bandwidth
2. **2-tier leaf-spine** — Core + leaf switches
3. **SHARP support** — In-network reduction (NVIDIA Quantum switches)
4. **Multi-job scheduling** — Use Volcano or Kueue for queue management
5. **Failure handling** — Elastic training with automatic node replacement

---

## Part 10: Summary and Quick Start

### Quick deployment checklist:

1. ✅ **Hardware setup**
   - Install GPUs with NVLink
   - Configure InfiniBand networking
   - Set up shared storage

2. ✅ **Software installation**
   - Ubuntu 22.04 + NVIDIA drivers + CUDA
   - MLNX_OFED for InfiniBand
   - NCCL + NCCL tests
   - Container runtime + NVIDIA Container Toolkit

3. ✅ **Kubernetes deployment**
   - Initialize cluster with kubeadm
   - Install CNI (Calico)
   - Deploy GPU Operator
   - Deploy RDMA device plugin
   - Install Training Operator

4. ✅ **Validate setup**
   - Run nvidia-smi on all nodes
   - Test InfiniBand with ib_write_bw
   - Run NCCL all-reduce benchmark
   - Deploy test GPU pod

5. ✅ **Deploy training job**
   - Build training container
   - Create PV/PVC for data
   - Submit PyTorchJob
   - Monitor with Prometheus + Grafana

6. ✅ **Deploy inference service**
   - Create inference deployment
   - Expose via Service/Ingress
   - Configure autoscaling

### Key configuration files:

All configurations are in this guide. Store them in a Git repo:

```
my-llm-cluster/
├── k8s/
│   ├── gpu-operator/
│   ├── training-operator/
│   ├── jobs/
│   │   └── llama-training-job.yaml
│   └── services/
│       └── llama-inference.yaml
├── docker/
│   └── Dockerfile.trainer
├── scripts/
│   ├── install-drivers.sh
│   ├── install-k8s.sh
│   └── nccl_env.sh
└── monitoring/
    └── grafana-dashboards/
```

---

## Conclusion

You now have a complete blueprint for building an HPC system optimized for LLM training and inference using:
- **GPUs with NVLink** for fast intra-node communication (900 GB/s)
- **InfiniBand networking** for high-speed inter-node connectivity (400 Gb/s)
- **NCCL** for efficient GPU-to-GPU data transfer
- **Kubernetes** for orchestration, scheduling, and production deployment

This architecture supports:
- ✅ Training models up to 100B+ parameters
- ✅ Multi-node distributed training (3D parallelism)
- ✅ Auto-scaling inference serving
- ✅ High availability and fault tolerance
- ✅ Efficient resource utilization

**Next steps:** Start with a 2-4 node pilot cluster, validate performance with NCCL benchmarks, then scale to production size.
