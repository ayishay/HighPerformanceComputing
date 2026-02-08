# Performance Analysis of Accelerator-Based Systems for AI/LLM

## Why Performance Analysis Matters

Training a frontier LLM can cost **$10-100+ million** in compute. A 10% efficiency improvement on a 10,000-GPU cluster saves millions of dollars and weeks of wall-clock time. Performance analysis answers critical questions:

- Are my GPUs actually doing useful work, or sitting idle?
- Is the bottleneck compute, memory, communication, or I/O?
- How close am I to the hardware's theoretical peak?
- Where should I invest optimization effort for the biggest return?

---

## The Roofline Model: First Principles of Performance

The Roofline Model is the **foundational framework** for understanding accelerator performance. It tells you whether your workload is **compute-bound** or **memory-bound**.

### Core Concept

Every operation has two requirements:
1. **Compute**: How many floating-point operations (FLOPs) it needs
2. **Memory**: How many bytes it must read/write

The ratio is called **Arithmetic Intensity (AI)**:

```
Arithmetic Intensity = FLOPs / Bytes Accessed (FLOP/Byte)
```

The accelerator has two ceilings:
1. **Peak Compute** (FLOPS): Maximum operations per second
2. **Peak Memory Bandwidth** (Bytes/s): Maximum data transfer rate

The **ridge point** is where the two ceilings meet:

```
Ridge Point = Peak Compute / Peak Memory Bandwidth (FLOP/Byte)
```

### Roofline Diagram

```
Performance
(TFLOPS)     ▲
             │          ┌─────────────────────── Peak Compute (989 TFLOPS FP16)
             │         ╱│
             │        ╱ │   Compute-Bound Region
             │       ╱  │   (AI > ridge point)
             │      ╱   │
             │     ╱    │
             │    ╱  Ridge Point
             │   ╱   (295 FLOP/Byte for H100)
             │  ╱
             │ ╱  Memory-Bound Region
             │╱   (AI < ridge point)
             └──────────────────────────────────► Arithmetic Intensity
                                                  (FLOP/Byte)

H100 SXM Example:
  Peak FP16 Compute:     989 TFLOPS
  Peak Memory Bandwidth: 3.35 TB/s
  Ridge Point:           989 / 3.35 = 295 FLOP/Byte
```

### Applying Roofline to LLM Operations

| Operation | Arithmetic Intensity | Bound | Why |
|-----------|---------------------|-------|-----|
| Matrix Multiply (large batch) | High (~200-1000 FLOP/B) | Compute | Large matrices → lots of reuse |
| Matrix Multiply (small batch / inference) | Low (~1-10 FLOP/B) | Memory | Small batch → each weight loaded, used once |
| Attention (without FlashAttention) | Low (~5-10 FLOP/B) | Memory | Must read/write O(n²) attention matrix |
| Attention (FlashAttention) | Medium (~50-100 FLOP/B) | Compute | Fused kernel, no O(n²) materialization |
| LayerNorm / RMSNorm | Very low (~1-2 FLOP/B) | Memory | Element-wise ops, minimal compute |
| Softmax | Very low (~2-5 FLOP/B) | Memory | Read all, compute exp, normalize |
| Embedding lookup | ~0 FLOP/B | Memory | Pure data movement |
| GELU / SiLU activation | Very low (~1 FLOP/B) | Memory | Element-wise |

**Key insight for LLMs:**
- **Training** (large batch sizes) → Most time spent in large matrix multiplies → **Compute-bound**
- **Inference** (batch size 1 or small) → Each weight loaded from HBM, used for few operations → **Memory-bound**
- This is why inference optimization focuses on **quantization** (reduce bytes) and **batching** (increase arithmetic intensity)

---

## Key Performance Metrics

### 1. Throughput Metrics

**Tokens per second (tokens/s)**
The most practical metric — how many tokens the system processes per unit time.

```
Training throughput:
  tokens/s = (batch_size × sequence_length × num_GPUs) / step_time

Example:
  batch_size = 4 per GPU
  sequence_length = 4096
  num_GPUs = 64
  step_time = 2.5 seconds

  tokens/s = (4 × 4096 × 64) / 2.5 = 419,430 tokens/s
```

**Samples per second**
```
samples/s = (global_batch_size) / step_time
```

### 2. Hardware Utilization Metrics

**Model FLOPs Utilization (MFU)**

The gold standard metric for LLM training efficiency. MFU measures what fraction of the hardware's **peak theoretical FLOPS** is actually used for model computation.

```
MFU = (Model FLOPs per step) / (Peak FLOPS × Step Time)

Model FLOPs for a transformer forward + backward pass (approximate):
  FLOPs ≈ 6 × N × B × S

  where:
    N = number of parameters
    B = batch size (per GPU)
    S = sequence length
    6 = 2 (forward multiply-add) + 4 (backward: 2× forward for gradients)

Example: LLaMA 70B training
  N = 70 × 10⁹
  B × S = 4 × 4096 = 16384 tokens per GPU
  FLOPs per step per GPU = 6 × 70e9 × 16384 = 6.88 × 10¹⁵ FLOPs

  H100 Peak FP16: 989 TFLOPS = 989 × 10¹² FLOPS
  Step time: 2.5 seconds

  MFU = 6.88e15 / (989e12 × 2.5) = 6.88e15 / 2.47e15 = 2.78

  Wait — that's > 1.0? That means our batch size or step time estimate
  is off. Let's recalculate with realistic numbers:

  Step time = 8.5 seconds (more realistic for 70B)
  MFU = 6.88e15 / (989e12 × 8.5) = 6.88e15 / 8.41e15 = 0.82 = 82%

Good MFU targets:
  > 50%: Acceptable
  > 60%: Good
  > 70%: Very good
  > 80%: Excellent (near state-of-the-art)

Real-world reported MFU:
  PaLM (Google, 6144 TPUv4):         ~46-58%
  LLaMA 2 (Meta, 2048 A100):         ~43%
  Megatron-LM (NVIDIA, 3072 H100):   ~55-60%
```

**Hardware FLOPs Utilization (HFU)**

Like MFU but includes **all** FLOPs (including communication, recomputation from activation checkpointing). HFU ≥ MFU always.

```
HFU = (All FLOPs including recomputation) / (Peak FLOPS × Step Time)
```

**GPU Utilization (SM Activity)**
```
GPU Utilization = Time SMs are executing kernels / Total time

Measured by: nvidia-smi, DCGM, Nsight Systems

Pitfall: 100% "GPU utilization" in nvidia-smi does NOT mean the GPU
         is working at peak efficiency. It only means at least one SM
         had at least one warp scheduled. The GPU could be at 100%
         utilization but only 10% of its peak FLOPS.
```

**Tensor Core Utilization**
```
TC Utilization = Time Tensor Cores are active / Total time

This is the metric that actually correlates with useful LLM throughput.
Measured by: Nsight Compute, DCGM
```

### 3. Memory Metrics

```
Peak Memory Usage:
  Measured by: torch.cuda.max_memory_allocated()
  Target: Use as much GPU memory as possible without OOM

Memory Bandwidth Utilization:
  Actual bandwidth / Peak bandwidth
  Measured by: Nsight Compute memory throughput metrics
  H100: 3.35 TB/s peak → achieving > 80% = 2.68 TB/s is good

Memory Efficiency:
  Useful data / Total memory allocated
  Fragmentation, padding, and alignment waste memory
```

### 4. Communication Metrics

```
Communication Overhead:
  = Time spent in NCCL collectives / Total step time
  Target: < 20-30% of total step time

All-Reduce Bandwidth:
  = Data volume / All-reduce time
  Compare against: nccl-tests baseline for your hardware

Communication-Computation Overlap:
  = Time communication is hidden behind computation / Total communication time
  Target: > 80% overlap (most communication hidden)
```

### 5. I/O Metrics

```
Data Loading Throughput:
  = Batch data size / Data loading time
  Target: Data loading should not be the bottleneck

  If data loading time > compute time → I/O bound
  Solutions: More DataLoader workers, NVMe, prefetching, pre-tokenization

Checkpoint I/O:
  = Checkpoint size / Checkpoint write time
  70B model checkpoint: ~140 GB
  Target: < 60 seconds on parallel file system
```

---

## Performance Analysis Tools

### Level 1: Quick Monitoring (Real-time)

**nvidia-smi**
```bash
# Continuous monitoring (updates every 1 second)
nvidia-smi dmon -s pucvmet -d 1

# Key columns:
#  pwr:  Power consumption (watts) — should be near TDP (700W for H100)
#  temp: Temperature — throttling starts at ~83°C
#  sm:   SM utilization % — should be high during compute
#  mem:  Memory utilization % — memory controller busy
#  fb:   Framebuffer (VRAM) used — how much GPU memory consumed
#  rxpci/txpci: PCIe bandwidth

# One-shot snapshot
nvidia-smi --query-gpu=timestamp,name,pci.bus_id,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv -l 1
```

**DCGM (Data Center GPU Manager)**

Enterprise-grade GPU monitoring with Prometheus integration:

```bash
# Start DCGM daemon
nv-hostengine

# Enable GPU metrics collection
dcgmi profile --enable

# Watch key metrics
dcgmi dmon -e 1001,1002,1003,1004,1005,1009,1010,1011,1012

# Metric IDs:
#  1001: SM Activity
#  1002: SM Occupancy
#  1003: Tensor Core Activity  ← most important for LLMs
#  1004: DRAM Active
#  1005: FP64 Active
#  1009: FP16 Active
#  1010: FP32 Active
#  1011: Memory Bandwidth Utilization
#  1012: NVLink Bandwidth
```

**Prometheus + Grafana Dashboard**

```yaml
# DCGM exporter for Kubernetes
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: dcgm-exporter
spec:
  selector:
    matchLabels:
      app: dcgm-exporter
  template:
    spec:
      containers:
      - name: dcgm-exporter
        image: nvcr.io/nvidia/k8s/dcgm-exporter:3.3.0
        ports:
        - containerPort: 9400    # Prometheus scrape endpoint
        volumeMounts:
        - name: device-plugin
          mountPath: /var/lib/kubelet/device-plugins
```

Key Grafana dashboard panels:
```
┌─────────────────────────────────────────────────────────┐
│                    GPU Cluster Dashboard                  │
│                                                          │
│  ┌─────────────────┐  ┌─────────────────┐               │
│  │ SM Utilization   │  │ Tensor Core %    │              │
│  │ ████████░░ 82%  │  │ ██████████ 95%  │              │
│  └─────────────────┘  └─────────────────┘               │
│  ┌─────────────────┐  ┌─────────────────┐               │
│  │ Memory Used      │  │ HBM Bandwidth   │              │
│  │ 72/80 GB (90%)  │  │ 2.8/3.35 TB/s   │              │
│  └─────────────────┘  └─────────────────┘               │
│  ┌─────────────────┐  ┌─────────────────┐               │
│  │ Power Draw       │  │ GPU Temperature  │              │
│  │ 650/700W (93%)  │  │ 72°C            │              │
│  └─────────────────┘  └─────────────────┘               │
│  ┌─────────────────────────────────────────┐            │
│  │ Training Throughput (tokens/s)           │            │
│  │ ▁▂▃▅▆█████████████████████████████████  │            │
│  │               400K tokens/s             │            │
│  └─────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────┘
```

---

### Level 2: Profiling (Detailed Analysis)

**NVIDIA Nsight Systems — System-Level Profiler**

Nsight Systems captures a **timeline** of all GPU activity, CPU activity, and communication. It's the first tool to use when investigating performance issues.

```bash
# Profile a training run (capture 5 training steps)
nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    --gpu-metrics-device=all \
    --output=llm_training_profile \
    --duration=60 \
    python train.py --model llama-7b --steps 5

# For multi-GPU profiling
nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas,nccl \
    --output=profile_rank_%q{RANK} \
    torchrun --nproc_per_node=8 train.py
```

**What Nsight Systems shows:**

```
Timeline View (simplified):

Time →  0ms        50ms       100ms      150ms      200ms
        │          │          │          │          │
CPU:    │▓▓ data   │          │▓▓ data   │          │
        │  load    │          │  load    │          │
        │          │          │          │          │
GPU 0:  │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │ ░░░░░░░  │▓▓▓▓▓▓▓▓▓│
        │  Forward + Backward│ NCCL     │Next step │
        │  (compute kernels) │ AllReduce│          │
        │          │          │          │          │
GPU 1:  │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │ ░░░░░░░  │▓▓▓▓▓▓▓▓▓│
        │          │          │          │          │
NCCL:   │          │          │ ████████ │          │
        │          │          │ AllReduce│          │
NVLink: │          │          │ ▓▓▓▓▓▓▓▓ │          │
        │          │          │ 800 GB/s │          │

▓ = Active  ░ = Waiting  █ = Communication  (blank) = Idle

Key things to look for:
1. Gaps between kernels (GPU idle → kernel launch overhead)
2. Long NCCL operations (communication bottleneck)
3. CPU data loading overlapping with GPU compute (good)
4. GPU waiting for CPU (data loading bottleneck)
```

**NVIDIA Nsight Compute — Kernel-Level Profiler**

Nsight Compute provides **deep analysis of individual CUDA kernels**. Use it after Nsight Systems identifies a slow kernel.

```bash
# Profile specific kernels
ncu --set full \
    --kernel-name "ampere_bf16_s16816gemm" \
    --launch-count 5 \
    --output=kernel_profile \
    python train.py --steps 1

# Profile all kernels in a range
ncu --set full \
    --launch-skip 100 --launch-count 50 \
    --output=all_kernels \
    python train.py
```

**What Nsight Compute shows for a single kernel:**

```
Kernel: ampere_bf16_s16816gemm_256x128_ldg8_f2f_stages_64x1_nn
Grid: [512, 4, 1]   Block: [128, 1, 1]

┌────────────────────────────────────────────────────┐
│              Roofline Analysis                       │
│                                                      │
│  Achieved: 812 TFLOPS (82% of peak 989 TFLOPS)     │
│  Arithmetic Intensity: 342 FLOP/Byte                │
│  Status: COMPUTE BOUND ✓                            │
│                                                      │
│  ┌────────────────────────────┐                     │
│  │ Compute:  82% of peak     │ ████████░░           │
│  │ Memory:   71% of peak     │ ███████░░░           │
│  │ SM Occupancy: 75%         │ ████████░░           │
│  │ Tensor Core Usage: 94%    │ █████████░           │
│  └────────────────────────────┘                     │
│                                                      │
│  Bottleneck: Instruction scheduling (warp stalls)   │
│  Suggestion: Increase tile size or occupancy         │
└────────────────────────────────────────────────────┘
```

**Key Nsight Compute metrics:**

| Metric | What It Tells You | Good Target |
|--------|-------------------|-------------|
| SM Throughput | % of peak SM instruction throughput | > 80% |
| Tensor Core Utilization | % of time TCs are executing | > 85% for GEMM |
| DRAM Throughput | % of peak HBM bandwidth | > 75% if memory-bound |
| L2 Hit Rate | Cache effectiveness | > 50% for repeated data |
| Occupancy | Active warps / max warps per SM | > 50% |
| Warp Stall Reasons | Why warps are not executing | Minimize stalls |

---

### Level 3: Framework-Level Profiling

**PyTorch Profiler**

```python
import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

# Profile with TensorBoard output
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(
        wait=2,        # Skip first 2 steps
        warmup=2,      # Warmup for 2 steps
        active=6,      # Profile 6 steps
        repeat=1
    ),
    on_trace_ready=tensorboard_trace_handler("./profiler_logs"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True
) as prof:
    for step, batch in enumerate(dataloader):
        if step >= 10:
            break
        loss = model(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        prof.step()   # Signal profiler that a step is complete

# Print summary
print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=20
))
```

**Output example:**
```
-------------------------------------------------------
Name                    CPU Time   CUDA Time   # Calls
-------------------------------------------------------
aten::mm                  2.1ms      45.2ms      192     ← Matrix multiply
aten::bmm                 0.8ms      12.3ms       64     ← Batched matmul (attention)
aten::scaled_dot_product  1.2ms      18.7ms       32     ← Flash Attention
nccl:all_reduce           0.1ms      22.4ms        8     ← Gradient sync
aten::layer_norm          0.3ms       4.1ms       64     ← LayerNorm
aten::gelu                0.1ms       1.8ms       32     ← Activation
aten::embedding           0.1ms       0.9ms        1     ← Token embedding
-------------------------------------------------------
Total                    12.5ms     142.0ms
Self CUDA time:                     142.0ms
```

**DeepSpeed Flops Profiler**

```python
from deepspeed.profiling.flops_profiler import FlopsProfiler

profiler = FlopsProfiler(model)

for step, batch in enumerate(dataloader):
    if step == 5:
        profiler.start_profile()

    loss = model(batch)
    loss.backward()
    optimizer.step()

    if step == 5:
        profiler.stop_profile()
        flops = profiler.get_total_flops()
        macs = profiler.get_total_macs()
        params = profiler.get_total_params()
        profiler.print_model_profile(
            profile_step=step,
            output_file="deepspeed_profile.txt"
        )
        profiler.end_profile()
```

**Output example:**
```
-------------------------- DeepSpeed Flops Profiler --------------------------
Profile Summary at step 5:
Notations:
  GFLOPs: billion floating-point operations per second
  MACs:   multiply-accumulate operations

Model Parameters:        6.74 B
FLOPs per iteration:     2.63 TFLOPs
MACs per iteration:      1.32 TMACs
FLOPs per GPU:           329 GFLOPs
Duration:                0.42 s
Throughput (per GPU):    783 GFLOPs/s
MFU (per GPU):           79.2%     ← This is what we care about

Module-level breakdown:
  LlamaAttention:       42.3% of total FLOPs
  LlamaMLP:             55.1% of total FLOPs
  LlamaRMSNorm:          0.2% of total FLOPs
  Other:                 2.4% of total FLOPs
```

---

### Level 4: Communication Profiling

**NCCL Debug Logging**

```bash
# Enable NCCL debug info
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Output shows:
# - Detected topology (NVLink, PCIe, InfiniBand)
# - Algorithm chosen (Ring, Tree, CollNet)
# - Channel count and buffer sizes
# - Per-operation timing
```

**Example NCCL debug output:**
```
node001:0:0 [0] NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] -1/-1/-1->0->1
node001:0:0 [0] NCCL INFO Channel 00/08 : 0 1 2 3 4 5 6 7
node001:0:0 [0] NCCL INFO Using network InfiniBand
node001:0:0 [0] NCCL INFO Ring 00 : 0[0] -> 1[1] via P2P/NVLink
node001:0:0 [0] NCCL INFO AllReduce: opCount 42 sendbuff 0x...
                         recvbuff 0x... count 262144 datatype 7
                         op 0 root 0 comm 0x... [nranks=8]
                         stream 0x... algorithm Ring protocol LL128
                         time 0.124 ms
```

**PyTorch Communication Profiling**

```python
import torch.distributed as dist

# Built-in communication logging
with dist.distributed_c10d._profile() as prof:
    for step in range(5):
        loss = model(batch)
        loss.backward()
        optimizer.step()

# Print communication stats
print(prof.report())
```

**nccl-tests Benchmarking**

```bash
# Benchmark all-reduce across the actual training cluster
# Run this BEFORE training to establish baseline

# Intra-node (8 GPUs, NVLink)
mpirun -np 8 ./build/all_reduce_perf -b 1M -e 4G -f 2 -g 1

# Inter-node (16 GPUs, 2 nodes, InfiniBand)
mpirun -np 16 --hostfile hosts.txt \
    -x NCCL_IB_DISABLE=0 \
    ./build/all_reduce_perf -b 1M -e 4G -f 2 -g 1

# Compare results against theoretical peak:
#   NVLink H100: ~450 GB/s bus bandwidth (bidirectional ring)
#   InfiniBand NDR 400Gb: ~50 GB/s per port, ~200 GB/s with 4 ports
```

---

## Common Performance Bottlenecks and Solutions

### Bottleneck 1: Low Tensor Core Utilization

**Symptoms:**
- Low MFU (< 40%)
- Nsight Compute shows low TC utilization
- `nvidia-smi` shows high GPU utilization but low throughput

**Causes and Solutions:**

| Cause | Solution |
|-------|----------|
| Matrix dimensions not aligned to tile sizes | Pad dimensions to multiples of 8 (FP16) or 16 (INT8) |
| Using FP32 instead of mixed precision | Enable BF16/FP16 mixed precision training |
| Small batch size | Increase micro-batch size or use gradient accumulation |
| Too many small kernels | Use kernel fusion (Flash Attention, fused optimizers) |
| Non-GEMM operations dominating | Use fused kernels for LayerNorm, activation, etc. |

```python
# Enable mixed precision in PyTorch
scaler = torch.amp.GradScaler('cuda')
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Bottleneck 2: Memory Bandwidth Saturation

**Symptoms:**
- Nsight Compute shows DRAM throughput near peak
- Adding more compute (larger batch) doesn't improve throughput
- Common during inference with small batch sizes

**Causes and Solutions:**

| Cause | Solution |
|-------|----------|
| Small batch inference | Batch requests together (continuous batching in vLLM) |
| Large model, few operations per weight | Quantize model (INT8, INT4, FP8) |
| Attention reading O(n²) memory | Use Flash Attention |
| Repeated memory access | Improve kernel data locality, use shared memory |

```
Quantization impact on memory bandwidth:

FP16 inference (70B model):
  Weight data per forward pass: 140 GB
  H100 bandwidth: 3.35 TB/s
  Minimum time (bandwidth-limited): 140 / 3350 = 41.8 ms per token

INT4 inference (70B model):
  Weight data per forward pass: 35 GB
  Minimum time (bandwidth-limited): 35 / 3350 = 10.4 ms per token
  ↑ 4x speedup just from reduced data movement
```

### Bottleneck 3: Communication Overhead

**Symptoms:**
- GPUs show periodic idle gaps in Nsight Systems timeline
- Scaling efficiency drops significantly with more nodes
- NCCL operations dominate step time

**Causes and Solutions:**

| Cause | Solution |
|-------|----------|
| Large gradient all-reduce | Gradient compression, reduce precision |
| No compute-communication overlap | Enable gradient bucketing (DDP default) |
| Slow interconnect | Use InfiniBand instead of Ethernet |
| Poor topology mapping | Match parallelism to topology (TP on NVLink, DP on IB) |
| Too many GPUs in DP group | Add pipeline parallelism to reduce DP group size |

**Measuring scaling efficiency:**
```
Scaling Efficiency = (Throughput on N GPUs) / (N × Throughput on 1 GPU)

Example:
  1 GPU:    1000 tokens/s
  8 GPUs:   7200 tokens/s  → 7200 / 8000 = 90% (intra-node, NVLink)
  64 GPUs:  48000 tokens/s → 48000 / 64000 = 75% (inter-node, IB)
  512 GPUs: 320000 tokens/s → 320000 / 512000 = 62.5%

Ideal: 100% (linear scaling)
Acceptable: > 70% for multi-node training
```

### Bottleneck 4: Data Loading / I/O

**Symptoms:**
- GPU utilization periodically drops to 0%
- CPU utilization is high during GPU idle periods
- Nsight Systems shows gaps between GPU kernels corresponding to data loading

**Causes and Solutions:**

| Cause | Solution |
|-------|----------|
| Slow storage | Use NVMe SSDs, parallel file system (Lustre, GPFS) |
| Not enough DataLoader workers | Increase `num_workers` in PyTorch DataLoader |
| No prefetching | Enable `prefetch_factor` in DataLoader |
| On-the-fly tokenization | Pre-tokenize and save to memory-mapped files |
| Checkpoint writing blocks training | Async checkpointing (DeepSpeed, PyTorch DCP) |

```python
# Optimized DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=4,
    num_workers=8,           # Parallel data loading
    pin_memory=True,         # Faster CPU→GPU transfer
    prefetch_factor=4,       # Prefetch 4 batches per worker
    persistent_workers=True  # Don't restart workers each epoch
)
```

### Bottleneck 5: Kernel Launch Overhead

**Symptoms:**
- Many tiny kernels visible in Nsight Systems
- Gaps between kernels (GPU idle between launches)
- CPU thread busy dispatching kernels

**Solutions:**
```python
# CUDA Graphs: Capture and replay entire sequences of kernels
# Eliminates per-kernel launch overhead

# Capture
static_input = torch.randn(B, S, D, device='cuda')
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(static_input)

# Replay (near-zero launch overhead)
static_input.copy_(real_input)
g.replay()
real_output = static_output.clone()
```

```python
# torch.compile: Fuses kernels automatically
model = torch.compile(model, mode="max-autotune")
```

---

## Performance Analysis Workflow

A systematic approach to analyzing LLM training/inference performance:

```
Step 1: Establish Baseline
─────────────────────────
├── Run nccl-tests to verify network bandwidth
├── Run GPU micro-benchmarks (GEMM throughput)
├── Calculate theoretical peak (MFU ceiling)
└── Record: tokens/s, step time, GPU memory usage

         │
         ▼

Step 2: Quick Monitoring (nvidia-smi / DCGM)
─────────────────────────────────────────────
├── Check GPU utilization (is it consistently high?)
├── Check memory usage (is it near capacity?)
├── Check power draw (near TDP = GPU working hard)
├── Check temperature (throttling?)
└── Check NVLink/PCIe bandwidth

         │
         ▼

Step 3: System-Level Profile (Nsight Systems)
──────────────────────────────────────────────
├── Capture 5-10 training steps
├── Identify: Is the bottleneck compute, memory, communication, or I/O?
├── Look for GPU idle gaps (bubbles)
├── Measure NCCL collective time vs compute time
├── Check CPU-GPU synchronization points
└── Quantify compute-communication overlap

         │
         ▼

Step 4: Kernel-Level Profile (Nsight Compute) — if compute-bound
───────────────────────────────────────────────────────────────
├── Profile hot kernels (top 5 by time)
├── Check Roofline position (compute vs memory bound)
├── Check Tensor Core utilization
├── Check memory access patterns (coalesced? bank conflicts?)
├── Check occupancy and warp stall reasons
└── Identify optimization opportunities

         │
         ▼

Step 5: Framework-Level Analysis (PyTorch Profiler)
───────────────────────────────────────────────────
├── Break down time by operation type
├── Identify unexpected operators consuming time
├── Check for unnecessary CPU-GPU synchronization
├── Verify mixed precision is active
└── Check DataLoader performance

         │
         ▼

Step 6: Optimize and Re-measure
────────────────────────────────
├── Apply optimizations one at a time
├── Re-measure after each change
├── Verify improvement with MFU / tokens/s
└── Repeat from Step 2
```

---

## Advanced: Performance Modeling

### Predicting Training Time

```
Total Training Time = (Total Tokens / Throughput in tokens/s)

Total Tokens for LLM training:
  LLaMA 2 7B:   2 trillion tokens
  LLaMA 2 70B:  2 trillion tokens
  GPT-4:        ~13 trillion tokens (estimated)

Example:
  Model: 70B parameters
  Cluster: 2048 H100 GPUs
  MFU: 50%

  FLOPs per token: 6 × 70e9 = 420 GFLOPs
  Total FLOPs: 420e9 × 2e12 = 8.4e23 FLOPs

  Cluster peak: 2048 × 989e12 = 2.025e18 FLOPS
  Effective: 2.025e18 × 0.50 = 1.013e18 FLOPS

  Training time: 8.4e23 / 1.013e18 = 829,222 seconds
                                     = 9.6 days
```

### Predicting Communication Time

```
All-Reduce time (Ring algorithm):
  T = 2 × (N-1)/N × M / BW

  where:
    N  = number of GPUs
    M  = message size (model parameters in bytes)
    BW = bandwidth per link

Example: 70B model, 64 GPUs, InfiniBand NDR (50 GB/s)
  M = 140 GB (FP16)
  T = 2 × (63/64) × 140 / 50 = 5.5 seconds

  With 4 InfiniBand ports (200 GB/s aggregate):
  T = 2 × (63/64) × 140 / 200 = 1.38 seconds
```

---

## Key Takeaways

1. **MFU is the gold standard metric** for LLM training efficiency — it measures what fraction of theoretical peak compute is used for actual model math.

2. **The Roofline Model is your first analysis tool** — it immediately tells you if a workload is compute-bound or memory-bound, directing where to focus optimization.

3. **LLM training is compute-bound; LLM inference is memory-bound** — this fundamental difference drives different optimization strategies (mixed precision for training, quantization for inference).

4. **Use tools in order of increasing detail**: nvidia-smi → DCGM → Nsight Systems → Nsight Compute → PyTorch Profiler. Don't jump to kernel profiling before understanding the system-level picture.

5. **Common bottlenecks in order of frequency**: communication overhead > low Tensor Core utilization > data loading > memory bandwidth > kernel launch overhead.

6. **Always benchmark the interconnect first** (nccl-tests) — many "training is slow" issues are actually network problems.

7. **Optimize one thing at a time and re-measure** — performance analysis is iterative. Changes can have unexpected interactions.

8. **Track tokens/s and MFU continuously during training** — performance can degrade over time due to thermal throttling, network congestion, or data pipeline issues.
