# Resource Schedulers: Kubernetes and SLURM for HPC in AI/LLM

## What Is a Resource Scheduler?

A resource scheduler is the **brain of an HPC cluster**. Think of it like an **air traffic controller for computing resources**. Just like an airport needs to decide which planes land on which runways and when to avoid collisions, a scheduler decides:

- **Who** gets to use which resources (which user/group owns each GPU)
- **When** their job runs (immediately, or wait in a queue?)
- **Where** (which nodes/GPUs) the job is placed (which server should run this job)
- **How much** resource (CPU, GPU, memory, time) each job receives (don't let one user hog everything)

**Why do we need this?** Without a scheduler, imagine 100 researchers all trying to use the same 8 GPUs:
- Users would fight over resources ("I need the GPU NOW!")
- Multiple jobs would try to use the same GPU and break
- Expensive hardware would sit idle while waiting for decisions
- Some users would get lucky and run immediately, others would never get a chance

A good scheduler solves these problems by maximizing **utilization** (keep all GPUs busy so nothing is wasted) while ensuring **fairness** (every user gets their turn and a fair share of resources).

---

## The Two Dominant Schedulers

| Aspect | SLURM | Kubernetes |
|--------|-------|------------|
| Origin | HPC / supercomputing (2002) | Cloud-native / containerized apps (Google, 2014) |
| Primary design goal | Batch job scheduling on bare metal | Container orchestration for microservices |
| Job model | Submit script → wait in queue → run | Deploy pods/containers → auto-scheduled |
| GPU support | Native (GRES) | Via device plugins + operators |
| Typical users | Research labs, national labs, universities | Cloud teams, MLOps, production inference |
| Networking | InfiniBand, bare metal | Overlay networks (Flannel, Calico) |
| State model | Jobs have start/end | Pods can run indefinitely |

---

## SLURM (Simple Linux Utility for Resource Management)

### Overview

SLURM is the **industry standard scheduler for HPC clusters**. It manages most of the world's top supercomputers, including those used to train frontier LLMs.

**Used by:** NVIDIA (Selene/Eos), Meta (RSC), most university HPC centers, national labs (ORNL, LLNL, NERSC)

### Architecture

```
┌──────────────────────────────────────────────────┐
│                   SLURM Cluster                   │
│                                                   │
│  ┌─────────────┐          ┌─────────────────┐    │
│  │  slurmctld   │◄────────│   SlurmDBD      │    │
│  │ (Controller) │         │ (Accounting DB)  │    │
│  │              │         │ MySQL/MariaDB    │    │
│  └──────┬───────┘         └─────────────────┘    │
│         │                                         │
│         │  Manages + Monitors                     │
│         ▼                                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │  slurmd   │ │  slurmd   │ │  slurmd   │ ...    │
│  │ (Node 0)  │ │ (Node 1)  │ │ (Node 2)  │        │
│  │ 8x GPU    │ │ 8x GPU    │ │ 8x GPU    │        │
│  └──────────┘ └──────────┘ └──────────┘          │
│         ▲                                         │
│         │                                         │
│  ┌──────────────────┐                             │
│  │  Login/Head Node  │  ← Users submit jobs here  │
│  │  (sbatch, srun)   │                             │
│  └──────────────────┘                             │
└──────────────────────────────────────────────────┘
```

**Key daemons (background services):**
- **slurmctld** ("SLURM controller") — The main boss. Lives on the head/login node. Its job is to listen when users submit jobs, decide where each job should run, and monitor all the compute nodes. Think of it as the air traffic control tower.
- **slurmd** ("SLURM daemon") — Runs on every compute node (the actual servers with GPUs). When the controller says "run this job on Node 0," the slurmd on Node 0 launches it, watches it, and reports back the status. Think of these as workers on the ground at different airports.
- **slurmdbd** ("SLURM database daemon") — Keeps a record (database) of everything: which job ran when, how long it took, who ran it, how many resources it used. Used for billing, fairness tracking, and auditing.

### Core Concepts

**Partitions (Queues)**
Partitions are like **separate lines at a store**. Each partition groups servers (nodes) that have similar capabilities or are meant for similar purposes. When you submit a job, you specify which partition it goes to, and the scheduler will only use nodes from that partition.

Example: A cluster might have:
- A "fast GPU" partition with the newest H100 GPUs for important research
- A "standard GPU" partition with older A100 GPUs for general use
- A "CPU-only" partition for non-GPU work

Each partition can have different rules:
- How long your job can run
- How many simultaneous jobs you can have
- Which users have access
- What priority it gets

```
Partition: gpu-h100  ("the express lane" — fastest, limited)
  Nodes: node[001-032] (32 servers)
  GPUs per node: 8x H100 GPUs (newest, fastest)
  Max time: 72 hours (short jobs only)
  Max jobs per user: 4 (limit so others get a turn)
  Priority: high (these jobs run before others)

Partition: gpu-a100  ("the regular lane" — standard)
  Nodes: node[033-064] (32 servers)
  GPUs per node: 8x A100 GPUs (slightly older)
  Max time: 168 hours (up to a week)
  Max jobs per user: 8 (more jobs allowed)
  Priority: normal (wait their turn)

Partition: cpu-only  ("the slow lane" — no GPUs)
  Nodes: node[065-128] (64 servers with just CPUs)
  Max time: 24 hours
  Priority: low (last to run)
```

**GRES (Generic Resources)**
GRES is SLURM's way of keeping track of special hardware like **GPUs**. Without GRES, SLURM would only know about CPUs and memory, not GPUs. GRES lets SLURM track:
- What type of GPU (H100, A100, RTX 4090, etc.)
- How many GPUs each node has
- Which GPUs are free vs. busy

```bash
# Configuration on each node (tells SLURM what hardware exists)
GresTypes=gpu                         # This node has GPUs
NodeName=node001 Gres=gpu:h100:8    # Specifically: 8 H100 GPUs

# Now when a user requests GPUs, SLURM can satisfy that request
# Example: User says "I need 4 GPUs" → SLURM finds a node with at least 4 free H100s
```

**Job Types:**
- **Batch jobs** (`sbatch`) — You write a script, submit it, and walk away. The scheduler runs it when resources are free. Perfect for long training jobs. Example: "Train my LLaMA model, I'll check results tomorrow."
- **Interactive jobs** (`srun`) — You get an interactive shell on a compute node. Run commands one by one and see results immediately. Useful for debugging or testing. Example: "Give me a node with 2 GPUs so I can test my training code."
- **Job arrays** (`--array`) — Submit 100 similar jobs with one command, each with different parameters. Perfect for hyperparameter sweeps. Example: "Run my training code with learning rates 1e-5, 2e-5, 5e-5, 1e-4" (4 jobs automatically submitted).

### SLURM for LLM Training — Practical Examples

**Example 1: Single-node multi-GPU training**

This script trains a model on **one server with 8 GPUs**. Good for moderate-sized models or testing before scaling up.

```bash
#!/bin/bash
#SBATCH --job-name=llm-finetune          # Give your job a friendly name
#SBATCH --partition=gpu-h100              # Use H100 partition
#SBATCH --nodes=1                         # Use 1 server only
#SBATCH --ntasks-per-node=1               # 1 job per node
#SBATCH --gpus-per-node=8                 # Reserve all 8 GPUs on that node
#SBATCH --cpus-per-task=64                # Also give me 64 CPU cores (for data loading)
#SBATCH --mem=512G                        # Give me 512 GB RAM
#SBATCH --time=24:00:00                   # My job will take up to 24 hours
#SBATCH --output=logs/%j.out              # Save output log (helpful for debugging)
#SBATCH --error=logs/%j.err               # Save error log

module load cuda/12.2                     # Set up GPU software
module load nccl/2.18                     # Set up communication library

# Launch training across all 8 GPUs
torchrun --nproc_per_node=8 \
    train.py \
    --model llama-7b \
    --data /shared/datasets/pile \
    --batch-size 32 \
    --bf16
```

**Example 2: Multi-node distributed LLM training (64 GPUs)**

This uses **8 servers × 8 GPUs = 64 GPUs total**. The model is automatically split across all 64 GPUs. Perfect for large models like LLaMA-70B.

```bash
#!/bin/bash
#SBATCH --job-name=llm-pretrain-70b
#SBATCH --partition=gpu-h100
#SBATCH --nodes=8                         # Use 8 different servers
#SBATCH --ntasks-per-node=8               # Run 8 copies of my code (one on each GPU)
#SBATCH --gpus-per-node=8                 # 8 GPUs per server = 64 GPUs total
#SBATCH --cpus-per-task=8                 # Each copy gets 8 CPU cores for preprocessing data
#SBATCH --mem=0                           # Use ALL available memory on each server
#SBATCH --time=72:00:00                   # 72-hour time limit for big training jobs
#SBATCH --exclusive                       # "Get me exclusive access" = no one else uses these servers
#SBATCH --output=logs/%j.out

# Get master node address for distributed training
# (computers need to talk to each other, so one becomes the "master")
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500

# Launch with srun — SLURM handles spawning across all 8 nodes automatically
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
    --model llama-70b \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --data-parallel-size 4
```

**Example 3: Hyperparameter sweep with job arrays**
```bash
#!/bin/bash
#SBATCH --job-name=hp-sweep
#SBATCH --partition=gpu-a100
#SBATCH --array=0-11
#SBATCH --gpus-per-node=4
#SBATCH --time=8:00:00

LEARNING_RATES=(1e-5 2e-5 5e-5 1e-4)
BATCH_SIZES=(16 32 64)

LR_IDX=$((SLURM_ARRAY_TASK_ID / 3))
BS_IDX=$((SLURM_ARRAY_TASK_ID % 3))

python train.py \
    --lr ${LEARNING_RATES[$LR_IDX]} \
    --batch-size ${BATCH_SIZES[$BS_IDX]} \
    --output results/run_${SLURM_ARRAY_TASK_ID}
```

### SLURM Scheduling Policies

When jobs are waiting in the queue, SLURM calculates a **priority score** for each job. The job with the highest score runs next. The score includes:

```
Job Priority = (Age Weight × Age Factor)
             + (Fairshare Weight × Fairshare Factor)
             + (Job Size Weight × Job Size Factor)
             + (Partition Weight × Partition Factor)
             + (QOS Weight × QOS Factor)
```

**In plain English:** It's like SLURM asks:
- "Has this job been waiting a long time?" (Age — old jobs should go first)
- "Has this user/group used a lot of resources already?" (Fairshare — don't let power users hog everything)
- "Is this job big?" (Job Size — maybe encourage big jobs to finish faster)
- "Which partition is this in?" (Partition — high-priority jobs go first)
- "Does this user have special access?" (QOS — researchers might get VIP treatment)

| Factor | Description |
|--------|-------------|
| **Age** | How long the job has been waiting in the queue |
| **Fairshare** | Whether the user/group has used more or less than their fair share |
| **Job Size** | Larger jobs can get priority (encourages full node usage) |
| **Partition** | Different partitions have different base priorities |
| **QOS** | Quality of Service tiers (e.g., premium, standard, low) |

**Backfill Scheduling:**
SLURM is smart about **wasting no computing power**. If a huge job (using 8 entire nodes) is waiting for all 8 nodes to become free, SLURM doesn't let those nodes sit idle. Instead, it runs smaller jobs in the gaps!

**Example:** Imagine you have 10 compute nodes:

```
Timeline without backfill (BAD — wasting power):
──────────────────────────────────────────►
Nodes 1-5: [Other jobs ending]   IDLE WAITING   [  Large Job X starts (needs 8 nodes) ]
Nodes 6-10:[Other jobs]          IDLE WAITING   [continues...]

Timeline WITH backfill (GOOD — uses everything):
──────────────────────────────────────────►
Nodes 1-5: [Other jobs ending]  [Small Job A]   [  Large Job X starts ]
Nodes 6-7: [Other jobs]         [Small Job B]   [Large Job X continues]
Nodes 8-10:[ending]             [Small Job C]   [Large Job X continues]

→ Small jobs fit in the gaps and finish BEFORE Large Job needs those nodes!
→ No wasted computing power!
```

**Preemption:**
Sometimes a high-priority job arrives and there are no free nodes. SLURM can **pause or cancel lower-priority jobs** to free up resources for the important one. Think of it like a VIP cutting in line. This keeps the system responsive to critical work, though it can be frustrating if your job gets preempted.

(Some clusters turn this off because researchers don't like their jobs being interrupted!)

### Key SLURM Commands

| Command | Purpose |
|---------|---------|
| `sbatch script.sh` | Submit a batch job |
| `srun --gpus=4 python train.py` | Run an interactive job |
| `squeue -u $USER` | View your jobs in the queue |
| `scancel <job_id>` | Cancel a job |
| `sinfo` | View node/partition status |
| `sacct -j <job_id>` | View job accounting (runtime, memory used) |
| `scontrol show job <job_id>` | Detailed job information |
| `scontrol show node <node>` | Detailed node information |

---

## Kubernetes (K8s)

### Overview

Kubernetes is the **standard orchestration platform for containerized workloads**. Originally designed for web services, it has been extended for AI/ML workloads through operators and custom resource definitions.

**Used by:** Cloud AI platforms (GKE, EKS, AKS), production inference serving, MLOps pipelines

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Kubernetes Cluster                    │
│                                                      │
│  ┌────────────────── Control Plane ───────────────┐  │
│  │  ┌──────────┐ ┌──────────┐ ┌───────────────┐  │  │
│  │  │API Server│ │Scheduler │ │Controller Mgr │  │  │
│  │  └──────────┘ └──────────┘ └───────────────┘  │  │
│  │  ┌──────────┐                                  │  │
│  │  │  etcd     │  (cluster state store)           │  │
│  │  └──────────┘                                  │  │
│  └────────────────────────────────────────────────┘  │
│                         │                            │
│         ┌───────────────┼───────────────┐            │
│         ▼               ▼               ▼            │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐     │
│  │  Worker     │  │  Worker     │  │  Worker     │    │
│  │  Node 0     │  │  Node 1     │  │  Node 2     │    │
│  │ ┌────────┐  │  │ ┌────────┐  │  │ ┌────────┐  │    │
│  │ │kubelet │  │  │ │kubelet │  │  │ │kubelet │  │    │
│  │ ├────────┤  │  │ ├────────┤  │  │ ├────────┤  │    │
│  │ │Pod: LLM│  │  │ │Pod: LLM│  │  │ │Pod: LLM│  │    │
│  │ │ train  │  │  │ │ train  │  │  │ │ train  │  │    │
│  │ │8x GPU  │  │  │ │8x GPU  │  │  │ │8x GPU  │  │    │
│  │ └────────┘  │  │ └────────┘  │  │ └────────┘  │    │
│  └────────────┘  └────────────┘  └────────────┘     │
└─────────────────────────────────────────────────────┘
```

**Key components:**
- **API Server** — Entry point for all cluster operations. You talk to the API Server (using `kubectl` commands) to tell Kubernetes what you want. Like a receptionist who takes your order.
- **Scheduler** — Decides which node a pod (container) should run on based on resource requests, affinity, taints/tolerations. Like an assignment officer who matches jobs to workers.
- **Controller Manager** — Runs controllers that maintain desired state. If you say "I want 5 copies of my training job," a controller automatically creates them and restarts any that fail. Like a manager who keeps workers productive.
- **etcd** — A database that stores all cluster state (which pods are running, which nodes exist, etc.). If the cluster crashes, it can recover from etcd.
- **kubelet** — Agent on each node that manages pods and reports node status back. Like a supervisor reporting daily to the manager.

### Core Concepts for AI Workloads

**Pods**
The smallest deployable unit in Kubernetes. A pod wraps one or more containers (usually just one). Think of it as a lightweight "box" that contains your code and all its dependencies.

**In SLURM terms:** A pod is like a batch job, except it can run indefinitely and be managed/restarted automatically.

```yaml
# A basic GPU pod that runs an LLM inference server
apiVersion: v1
kind: Pod
metadata:
  name: llm-inference
spec:
  containers:
  - name: model-server
    image: vllm/vllm-openai:latest
    resources:
      limits:
        nvidia.com/gpu: 4       # Request 4 GPUs
        memory: "256Gi"
      requests:
        nvidia.com/gpu: 4
        memory: "256Gi"
```

**GPU Device Plugin**
Kubernetes doesn't natively understand GPUs. Think of it like buying a printer — your computer doesn't automatically know how to use it. NVIDIA's **device plugin** is the "driver" that teaches Kubernetes about GPUs.

What it does:
- Discovers all GPUs on each node
- Reports them to Kubernetes ("Node 1 has 8 GPUs free")
- Allocates specific GPUs to containers
- Prevents multiple containers from using the same GPU

```
┌───────────────────────────────────┐
│          Worker Node              │
│  ┌─────────────────────────────┐  │
│  │  NVIDIA Device Plugin (DS)  │  │
│  │  - Discovers GPUs on node   │  │
│  │  - Reports to kubelet       │  │
│  │  - Allocates GPUs to pods   │  │
│  └──────────────┬──────────────┘  │
│                 ▼                  │
│  kubelet knows: nvidia.com/gpu: 8 │
│                 (8 GPUs available) │
│  ┌──────┐ ┌──────┐ ┌──────┐      │
│  │Pod A │ │Pod B │ │(free)│      │
│  │2 GPUs│ │4 GPUs│ │2 GPUs│      │
│  └──────┘ └──────┘ └──────┘      │
└───────────────────────────────────┘
```

**GPU Operator (NVIDIA)**
Automates the full GPU software stack on Kubernetes (everything your GPU needs to run). Without it, cluster admins have to manually install drivers, libraries, and monitoring tools on each node. The operator makes it automatic:
- GPU driver installation
- Container toolkit (lets containers access GPUs)
- Device plugin (the "driver" we just talked about)
- DCGM monitoring (watches GPU health)
- MIG (Multi-Instance GPU) manager (splits one GPU into multiple smaller ones)
- GDRCopy for GPUDirect RDMA (GPU-to-GPU communication without CPU)

### Kubernetes for LLM Training

**Kubeflow Training Operator** — This is a Kubernetes add-on that understands **distributed training jobs**. Without it, Kubernetes doesn't know what a "distributed training job" is. The operator makes it easy to launch multi-node training with a simple configuration file.

Key benefit: Instead of manually creating multiple Pods and coordinating them, you describe what you want ("8 nodes, 8 GPUs each") and the operator handles all the details.

```yaml
# PyTorchJob for distributed LLM training
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: llama-70b-training
spec:
  elasticPolicy:
    rdzvBackend: c10d
    minReplicas: 4
    maxReplicas: 8
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        spec:
          containers:
          - name: trainer
            image: my-registry/llm-trainer:v1
            command:
            - torchrun
            - --nproc_per_node=8
            - train.py
            - --model=llama-70b
            resources:
              limits:
                nvidia.com/gpu: 8
                rdma/rdma_shared_device_a: 1   # InfiniBand
              requests:
                nvidia.com/gpu: 8
                memory: "512Gi"
                cpu: "64"
          volumes:
          - name: training-data
            persistentVolumeClaim:
              claimName: pile-dataset-pvc
          - name: shared-memory
            emptyDir:
              medium: Memory
              sizeLimit: "128Gi"
    Worker:
      replicas: 7    # 7 workers + 1 master = 8 nodes
      template:
        spec:
          containers:
          - name: trainer
            image: my-registry/llm-trainer:v1
            # ... same as master
```

### Kubernetes for LLM Inference Serving

Serving (inference) is very different from training:
- **Training:** "Please use 8 GPUs for the next 72 hours to train a model"
- **Serving:** "I have a trained model. Users will ask it questions constantly. Automatically scale up when many users come, scale down when it's quiet."

Kubernetes is **great** at serving because it can automatically add/remove copies of your model based on demand:

```yaml
# Serving an LLM with vLLM on Kubernetes
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-serving
spec:
  replicas: 3           # 3 replicas for high availability
  selector:
    matchLabels:
      app: llm-serving
  template:
    metadata:
      labels:
        app: llm-serving
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
        - --model=meta-llama/Llama-3-8B
        - --tensor-parallel-size=4
        - --max-model-len=4096
        resources:
          limits:
            nvidia.com/gpu: 4
---
apiVersion: v1
kind: Service
metadata:
  name: llm-service
spec:
  selector:
    app: llm-serving
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
---
# Autoscale based on GPU utilization or request queue depth
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-serving
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: gpu_utilization
      target:
        type: AverageValue
        averageValue: "80"
```

### Kubernetes Scheduling for GPUs

**Node Affinity** — Tell Kubernetes "I need a specific type of GPU." Without this, your training job might land on an old GPU node even though fast H100s are available. With this, you say "Only schedule me on H100 nodes":

```yaml
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: nvidia.com/gpu.product
          operator: In
          values:
          - NVIDIA-H100-SXM  # Only use H100 GPUs, nothing else
```

**Topology-Aware Scheduling** — If your job uses GPUs that need to communicate super-fast with each other (like for distributed training), make sure they land on the same node so they use fast NVLink instead of slow network cables. This example keeps 2 replicas of your job together:

```yaml
# Ensure pods land on nodes where GPUs share NVLink
topologySpreadConstraints:
- maxSkew: 1
  topologyKey: kubernetes.io/hostname
  whenUnsatisfiable: DoNotSchedule
```

**Taints and Tolerations** — Kubernetes way of saying "this node is for special workloads only." A taint is a "do not schedule here" mark. A toleration is a "I'm special, I can run here" badge.

Example: You have expensive H100 GPUs that cost $5000 each. You don't want people wasting them on small CPU tasks. You taint the H100 nodes with "gpu=true:NoSchedule" (meaning: no pods allowed by default). Then only pods with a special toleration badge can run there:

```bash
# Taint GPU nodes (tells Kubernetes: "No one can use these nodes by default")
kubectl taint nodes gpu-node-01 gpu=true:NoSchedule

# Only pods with matching toleration can be scheduled there
```

```yaml
tolerations:
- key: "gpu"
  operator: "Equal"
  value: "true"
  effect: "NoSchedule"  # This pod is OK with the "gpu=true" taint
```

---

## Kubernetes vs SLURM: Detailed Comparison for AI/LLM

### Scheduling Model

```
SLURM: Queue-based batch scheduling
─────────────────────────────────────
User submits → Job waits in queue → Resources free → Job runs → Job ends

Timeline:
  Queue:    [Job A waiting] [Job B waiting] [Job C waiting]
  Running:  [   Job X   ] [   Job Y   ]
  Done:     [Job W complete]

Kubernetes: Declarative desired-state scheduling
──────────────────────────────────────────────
User declares desired state → Scheduler continuously reconciles

  Desired: 3 replicas of LLM server with 4 GPUs each
  Current: 2 running, 1 pending (waiting for GPU node)
  Action:  Scheduler watches for free GPUs, schedules 3rd replica
```

### Feature Comparison

| Feature | SLURM | Kubernetes |
|---------|-------|------------|
| **Job queuing** | Native, sophisticated (fairshare, backfill, preemption) | Basic (requires Volcano, Kueue, or Yunikorn) |
| **Multi-node jobs** | Native (`srun` spans nodes seamlessly) | Requires operators (Kubeflow, MPI Operator) |
| **GPU scheduling** | Native GRES, topology-aware | Device plugin + GPU Operator |
| **InfiniBand / RDMA** | Native, zero overhead | Requires host networking or RDMA device plugins |
| **Containers** | Supported via Singularity/Enroot | Native (Docker/containerd) |
| **Auto-scaling** | Not built-in (fixed cluster) | Native HPA/VPA, cluster autoscaler |
| **Self-healing** | Job restart via `--requeue` | Automatic pod restart, rescheduling |
| **Multi-tenancy** | Accounts, partitions, QOS | Namespaces, ResourceQuotas, LimitRanges |
| **Monitoring** | `sacct`, Prometheus exporters | Prometheus + Grafana ecosystem |
| **Network performance** | Excellent (bare metal) | Overhead from overlay networks |
| **Ease of use for researchers** | Familiar (bash scripts) | Steeper learning curve (YAML, kubectl) |
| **CI/CD integration** | Limited | Native (ArgoCD, Flux, Tekton) |
| **Ecosystem** | MPI, OpenMP, NCCL | Helm charts, operators, service mesh |

### When to Use Which

**Choose SLURM when:**
- **Training frontier LLMs** (100B+ parameters) — You need every millisecond of GPU speed. SLURM gives you bare-metal performance with zero overhead.
- **Maximum GPU-to-GPU network performance is critical** — Your GPUs need to talk super fast. InfiniBand + bare metal = lowest latency.
- **Bare-metal InfiniBand clusters** — You have expensive InfiniBand hardware. SLURM uses it efficiently; Kubernetes adds overhead with overlay networks.
- **Research/academic environment with batch workloads** — Researchers just want to submit a script and walk away. SLURM is perfect for that.
- **Multi-node training spanning tens to hundreds of GPUs** — SLURM was built for this. The ecosystem (MPI, NCCL) is mature and optimized.
- **Users are researchers comfortable with bash/CLI** — Your users write bash scripts, not YAML. Don't make them learn Kubernetes.

**Choose Kubernetes when:**
- **Serving/inference at scale** — You need to handle variable traffic. Kubernetes autoscaling is great for this.
- **Mixed workloads** (training + inference + data pipelines) — One platform handles everything from model serving to ETL.
- **Cloud-native infrastructure** (GKE, EKS, AKS) — You're already using Kubernetes in the cloud. Use it for AI too.
- **Need CI/CD pipelines** — GitHub Actions → automatic model test → automatic deployment. Kubernetes integrates perfectly with this.
- **Multi-tenant production environment** — Cloud provider managing many customers' models. Kubernetes isolation is solid.
- **Team prefers infrastructure-as-code** — "Everything is a YAML file" appeals to DevOps/SRE teams.

**Use both** (common in real production):
```
┌─────────────────────────────────────────────┐
│              Organization                    │
│                                              │
│  ┌──────────────────┐  ┌─────────────────┐  │
│  │  SLURM Cluster   │  │  K8s Cluster    │  │
│  │                  │  │                 │  │
│  │  LLM Training    │──│  LLM Serving    │  │
│  │  Research jobs   │  │  API endpoints  │  │
│  │  Batch inference │  │  MLOps pipeline │  │
│  │  Bare metal +    │  │  Autoscaling    │  │
│  │  InfiniBand      │  │  Cloud-native   │  │
│  └──────────────────┘  └─────────────────┘  │
│           │                     │            │
│           └──── Shared Storage ─┘            │
│              (Lustre / S3 / NFS)             │
└─────────────────────────────────────────────┘
```

---

## Other Schedulers Worth Knowing

### Volcano (Kubernetes-native batch scheduler)

Volcano adds **HPC-style scheduling** to Kubernetes. It's for people who like Kubernetes but want SLURM-like features. Key features:
- **Gang scheduling** — \"Start all 8 pods together or none at all.\" Prevents weird partial-job states where 3 out of 8 training nodes are running (training would be deadlocked).
- **Fair-share queuing** — Multiple teams can queue jobs, and they get equal time slices even if one team submits lots of jobs.
- **Task dependencies** — \"Task B only starts after Task A finishes\" — useful for multi-stage training.
- **Priority-based preemption** — High-priority jobs can interrupt lower-priority ones (SLURM feature that Kubernetes normally doesn't have).

```yaml
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: llm-training
spec:
  minAvailable: 8       # Gang scheduling: need all 8 pods or none
  schedulerName: volcano
  queue: gpu-queue
  tasks:
  - replicas: 8        # Want 8 copies, all or nothing
    name: trainer
    template:
      spec:
        containers:
        - name: trainer
          image: llm-trainer:v1
          resources:
            limits:
              nvidia.com/gpu: 8
```

### Kueue (Kubernetes-native job queuing)

Google's solution for job queuing in Kubernetes. Think of it as "SLURM fairshare for Kubernetes." Features:
- **ClusterQueues with resource quotas** — "Team A gets 1000 GPU-hours per month, Team B gets 500." Prevents one team from hoarding resources.
- **Fair sharing between teams** — If both teams want resources and they're equal priority, they take turns.
- **Priority-based preemption** — High-priority jobs can interrupt lower-priority ones (useful for urgent research deadlines).
- **Integrates with standard Kubernetes Job, PyTorchJob, etc.** — Unlike Volcano which requires special "volcanojobs," Kueue works with plain Kubernetes.

### PBS Pro / Torque

These are **older HPC schedulers** from the 1990s-2000s. If you see them, you're probably at a university or national lab that hasn't updated their infrastructure recently.
- Similar to SLURM but less actively developed (PBS Pro has some users, Torque is mostly dead)
- Missing modern features that SLURM has (gang scheduling, better GPU support)
- Being replaced by SLURM in most new deployments
- **Bottom line:** Only use if you inherited a PBS cluster; don't pick it for new work.

### Ray (Application-level scheduler)

Ray is **not a cluster scheduler** (like SLURM or Kubernetes). It's a framework that manages resources **within a single application**, not across a whole cluster.

Think of it this way: SLURM decides "give this GPU to Job A," Ray decides "within Job A, give this GPU task to Worker 1."

Common uses in LLM world:
- **vLLM inference serving** — Manages multiple inference workers to spread requests across GPUs
- **Ray Tune** — Hyperparameter tuning (trains 100 models in parallel with different parameters)
- **Reinforcement Learning** — Parallel simulation workers + training workers all in one Ray application

**When to use Ray:** When you have a single long-running application (not batch jobs) that needs to manage its own work internally.

---

## Key Takeaways

1. **SLURM is the default choice for HPC-scale LLM training** — Battle-tested by top labs. Bare-metal performance. Excellent for distributed training across many GPUs. If you're training a 70B-parameter model, SLURM is your scheduler.

2. **Kubernetes dominates for LLM serving/inference and MLOps** — Autoscaling (automatically add copies when traffic spikes), self-healing (automatically restart dead containers), cloud-native ecosystem. Perfect for production API serving.

3. **The gap is closing** — Tools like Volcano and Kueue bring HPC-style scheduling to Kubernetes. Tools like Enroot/Pyxis bring containers to SLURM. They're becoming more similar over time.

4. **GPU scheduling requires extra infrastructure** on both platforms:
   - SLURM: Configure GRES (just a text file)
   - Kubernetes: Install device plugin + GPU Operator

5. **Network topology matters**:
   - SLURM: Excellent on bare-metal InfiniBand (zero overhead)
   - Kubernetes: Adds overhead with overlay networks (okay for inference, not ideal for training)

6. **Many organizations use both** — SLURM for training (batch jobs, maximum performance), Kubernetes for serving (autoscaling, always-on), with shared storage (Lustre, S3, NFS) connecting them. This is the real-world setup at Meta, Google, and other big AI labs.
