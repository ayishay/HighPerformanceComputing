# Accelerators (GPU, FPGA, TPU, ASIC) and Mapping LLMs onto Them

## What Are Accelerators?

Accelerators are specialized hardware processors designed to perform specific types of computation much faster than general-purpose CPUs. In the context of AI/LLM workloads, accelerators excel at the core operation of deep learning: **matrix multiplication and tensor operations**.

A CPU has a few powerful cores optimized for sequential logic and branching. An accelerator trades that flexibility for **massive parallelism** — thousands of simpler cores executing the same operation on different data simultaneously.

---

## Types of Accelerators

### 1. GPU (Graphics Processing Unit)

**What it is:** Originally designed for rendering graphics, GPUs have thousands of small cores that execute the same instruction on many data points in parallel (SIMT — Single Instruction, Multiple Threads).

**Why it dominates AI:**
- Matrix multiplications map naturally onto GPU architecture
- Mature software ecosystem (CUDA, cuDNN, cuBLAS)
- High memory bandwidth via HBM (High Bandwidth Memory)

**Key GPUs for LLMs:**

| GPU | VRAM | Memory Bandwidth | FP16 TFLOPS | Interconnect |
|-----|------|-------------------|-------------|-------------|
| NVIDIA H100 SXM | 80 GB HBM3 | 3.35 TB/s | 989 | NVLink 4.0 (900 GB/s) |
| NVIDIA A100 | 80 GB HBM2e | 2.0 TB/s | 312 | NVLink 3.0 (600 GB/s) |
| NVIDIA B200 | 192 GB HBM3e | 8.0 TB/s | 2,250 | NVLink 5.0 (1,800 GB/s) |
| AMD MI300X | 192 GB HBM3 | 5.3 TB/s | 1,307 | Infinity Fabric |

**GPU Architecture Internals:**
```
GPU Chip
├── Streaming Multiprocessors (SMs) — H100 has 132 SMs
│   ├── CUDA Cores (FP32/FP64 math)
│   ├── Tensor Cores (matrix multiply-accumulate, e.g., FP16/BF16/FP8)
│   ├── Register File (fast per-thread storage)
│   └── Shared Memory / L1 Cache
├── L2 Cache (shared across all SMs)
├── HBM Memory Controllers
└── NVLink / PCIe Interface
```

**Tensor Cores** are the key innovation for AI — they perform a fused matrix multiply-accumulate (MMA) operation on small matrix tiles (e.g., 16x16) in a single clock cycle, delivering an order of magnitude more throughput than standard CUDA cores for mixed-precision math.

---

### 2. TPU (Tensor Processing Unit)

**What it is:** Google's custom ASIC (Application-Specific Integrated Circuit) designed from the ground up for neural network training and inference.

**Key features:**
- Systolic array architecture — data flows through a grid of processing elements in a wave-like pattern, maximizing data reuse
- Tightly integrated with Google's software stack (JAX, TensorFlow, XLA compiler)
- Connected via Inter-Chip Interconnect (ICI) in pod configurations (thousands of TPUs)

**TPU Versions:**

| TPU Version | HBM | Peak BF16 TFLOPS | Notes |
|-------------|-----|-------------------|-------|
| TPU v4 | 32 GB HBM2e | 275 | Used in PaLM training |
| TPU v5e | 16 GB HBM2e | 197 | Cost-optimized for inference |
| TPU v5p | 95 GB HBM2e | 459 | Training-focused |
| TPU v6 (Trillium) | 32 GB HBM | 920 | Latest generation |

**Systolic Array Illustration:**
```
Data flows right →
             ┌────┐ ┌────┐ ┌────┐ ┌────┐
Weights  →   │ PE │→│ PE │→│ PE │→│ PE │→ Output
flow down    └──┬─┘ └──┬─┘ └──┬─┘ └──┬─┘
             ┌──▼─┐ ┌──▼─┐ ┌──▼─┐ ┌──▼─┐
         →   │ PE │→│ PE │→│ PE │→│ PE │→
             └──┬─┘ └──┬─┘ └──┬─┘ └──┬─┘
             ┌──▼─┐ ┌──▼─┐ ┌──▼─┐ ┌──▼─┐
         →   │ PE │→│ PE │→│ PE │→│ PE │→
             └────┘ └────┘ └────┘ └────┘

PE = Processing Element (multiply-accumulate unit)
Each PE multiplies, adds to accumulator, and passes data to neighbors.
```

---

### 3. FPGA (Field-Programmable Gate Array)

**What it is:** A chip with a grid of configurable logic blocks and interconnects that can be reprogrammed after manufacturing to implement any digital circuit.

**Strengths:**
- Fully customizable datapath — you design the exact pipeline for your computation
- Low latency (no instruction fetch/decode overhead)
- Power efficient for specific fixed workloads
- Reconfigurable — can be reprogrammed for different model architectures

**Weaknesses for LLMs:**
- Much lower raw throughput than GPUs for matrix math
- Harder to program (requires HDL: Verilog/VHDL, or HLS: High-Level Synthesis)
- Smaller memory capacity compared to GPU HBM
- Limited software ecosystem for AI

**Where FPGAs excel in AI:**
- Low-latency inference at the edge
- Custom precision formats (e.g., 4-bit, 3-bit quantized models)
- Sparse computation (skipping zero values efficiently)
- Microsoft uses FPGAs (Project Brainwave/Catapult) in Azure for real-time inference

**FPGA Architecture:**
```
┌──────────────────────────────────────┐
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐   │
│  │ CLB │─│ CLB │─│ CLB │─│ CLB │   │  CLB = Configurable Logic Block
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘   │
│  ┌──┴──┐ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐   │
│  │ CLB │─│ DSP │─│ CLB │─│BRAM │   │  DSP = Digital Signal Processor
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘   │  BRAM = Block RAM
│  ┌──┴──┐ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐   │
│  │BRAM │─│ CLB │─│ DSP │─│ CLB │   │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘   │
│     │  Programmable Interconnect │   │
│  ┌──┴──────────────────────┴──┐     │
│  │     I/O Blocks (PCIe, DDR) │     │
│  └────────────────────────────┘     │
└──────────────────────────────────────┘
```

---

### 4. Custom AI ASICs

Purpose-built chips for AI with no reconfigurability:

| Chip | Maker | Notes |
|------|-------|-------|
| TPU | Google | Systolic arrays, tightly coupled with JAX/TF |
| Trainium/Inferentia | AWS | NeuronCore architecture for AWS workloads |
| Gaudi 3 | Intel/Habana | Targets training, integrated with PyTorch |
| Wafer-Scale Engine | Cerebras | Entire wafer as one chip — 850,000 cores, 44 GB on-chip SRAM |
| Dojo | Tesla | Custom chip for autonomous driving training |

---

## Accelerator Comparison Summary

| Feature | GPU | TPU | FPGA | Custom ASIC |
|---------|-----|-----|------|-------------|
| Peak throughput | Very high | Very high | Moderate | Very high |
| Programmability | High (CUDA) | Medium (JAX/XLA) | Low (HDL/HLS) | None (fixed) |
| Memory capacity | High (up to 192 GB) | Medium | Low-Medium | Varies |
| Memory bandwidth | Very high | High | Moderate | Very high |
| Power efficiency | Moderate | High | High | Highest |
| Software ecosystem | Best (PyTorch, CUDA) | Good (JAX) | Limited | Vendor-specific |
| Flexibility | General purpose | AI-focused | Reprogrammable | Fixed function |
| Best use case | Training + Inference | Training at scale | Edge inference | Dedicated workloads |

---

## How to Map LLMs onto Accelerators

"Mapping" an LLM means deciding **how to distribute the model's computation and parameters across one or more accelerators** so that training or inference runs efficiently.

### The Core Challenge

A large language model like LLaMA 3 70B has:
- **70 billion parameters** → ~140 GB in FP16 (2 bytes per parameter)
- During training, optimizer states (Adam) add ~4x → **~560 GB total memory**
- A single H100 has only 80 GB of memory

You **cannot fit the model on one GPU**. You must distribute it.

---

### Mapping Strategy 1: Data Parallelism (DP)

**How it works:**
- Each GPU holds a **complete copy** of the model
- The training batch is split across GPUs — each processes different data
- After the forward and backward pass, gradients are **synchronized** (all-reduce)

```
          Training Batch
    ┌────────┬────────┬────────┐
    │Batch 0 │Batch 1 │Batch 2 │
    └───┬────┘───┬────┘───┬────┘
        ▼        ▼        ▼
    ┌───────┐┌───────┐┌───────┐
    │ GPU 0 ││ GPU 1 ││ GPU 2 │  ← Each holds full model copy
    │Model  ││Model  ││Model  │
    │Copy   ││Copy   ││Copy   │
    └───┬───┘└───┬───┘└───┬───┘
        │        │        │
        └────────┼────────┘
                 ▼
          All-Reduce Gradients
          (synchronize updates)
```

**When to use:** Model fits in a single GPU's memory.

**Limitation:** Each GPU must hold the entire model, so this doesn't work for very large LLMs.

---

### Mapping Strategy 2: Tensor Parallelism (TP)

**How it works:**
- Individual **layers are split** across GPUs within the same node
- For a linear layer `Y = XW`, the weight matrix `W` is **column-split** or **row-split** across GPUs
- Requires very fast interconnect (NVLink) because GPUs must communicate during every layer

```
              Single Transformer Layer
    ┌─────────────────────────────────────┐
    │         Attention: W_Q, W_K, W_V    │
    │                                     │
    │  GPU 0 gets    GPU 1 gets           │
    │  W_Q[:, 0:d/2] W_Q[:, d/2:d]       │  ← Column split
    │  W_K[:, 0:d/2] W_K[:, d/2:d]       │
    │  W_V[:, 0:d/2] W_V[:, d/2:d]       │
    │                                     │
    │  Each GPU computes partial output   │
    │  then all-reduce to combine         │
    └─────────────────────────────────────┘
```

**When to use:** Model is too large for one GPU; GPUs are connected via NVLink within a node.

**Typical TP degree:** 2, 4, or 8 (matching GPUs per node).

---

### Mapping Strategy 3: Pipeline Parallelism (PP)

**How it works:**
- The model is split **by layers** — different groups of layers go to different GPUs
- Data flows through GPUs sequentially like an assembly line
- **Micro-batching** keeps all GPUs busy (GPipe, PipeDream)

```
    Input → [GPU 0: Layers 0-19] → [GPU 1: Layers 20-39] → [GPU 2: Layers 40-59] → Output
            ─────────────────────────────────────────────────────────────
    Time →  |  micro-batch 1  |                |                |
            |  micro-batch 2  |  micro-batch 1 |                |
            |  micro-batch 3  |  micro-batch 2 |  micro-batch 1 |
            |                 |  micro-batch 3  |  micro-batch 2 |
            |                 |                 |  micro-batch 3  |
```

**When to use:** Model is too large for one node; inter-node bandwidth is limited (InfiniBand).

**Trade-off:** Pipeline bubbles (idle time) reduce efficiency. More micro-batches reduce bubble overhead.

---

### Mapping Strategy 4: Expert Parallelism (EP)

**How it works:**
- For Mixture-of-Experts (MoE) models (e.g., Mixtral, GPT-4 reportedly)
- Each GPU holds a subset of the experts
- A routing function sends each token to the appropriate expert GPU

```
    Input Token
        │
        ▼
    ┌──────────┐
    │  Router   │ ← Decides which experts to activate
    └────┬─────┘
    ┌────┼──────────────────┐
    ▼    ▼                  ▼
┌──────┐┌──────┐      ┌──────┐
│GPU 0 ││GPU 1 │ ...  │GPU 7 │
│Exp 0 ││Exp 1 │      │Exp 7 │
│Exp 1 ││Exp 2 │      │Exp 15│
└──────┘└──────┘      └──────┘
```

---

### Mapping Strategy 5: ZeRO (Zero Redundancy Optimizer)

**How it works (DeepSpeed ZeRO):**
- Partitions optimizer states, gradients, and/or parameters across GPUs
- Each GPU holds only a **shard**, not the full copy
- Communicates on-demand during forward/backward pass

**ZeRO Stages:**

| Stage | What's Partitioned | Memory Savings |
|-------|--------------------|----------------|
| ZeRO-1 | Optimizer states | ~4x |
| ZeRO-2 | + Gradients | ~8x |
| ZeRO-3 | + Parameters | ~Nx (N = number of GPUs) |

---

### Combining Strategies: 3D Parallelism

Real-world LLM training uses **all strategies together**:

```
┌─────────────────────────────────────────────────────┐
│                    3D Parallelism                    │
│                                                     │
│  Data Parallel Group (across nodes)                 │
│  ┌───────────────────┐  ┌───────────────────┐       │
│  │  Pipeline Stage 0 │  │  Pipeline Stage 0 │  DP   │
│  │  ┌─────┬─────┐    │  │  ┌─────┬─────┐    │       │
│  │  │GPU 0│GPU 1│ TP │  │  │GPU 4│GPU 5│ TP │       │
│  │  └─────┴─────┘    │  │  └─────┴─────┘    │       │
│  │  Pipeline Stage 1 │  │  Pipeline Stage 1 │       │
│  │  ┌─────┬─────┐    │  │  ┌─────┬─────┐    │       │
│  │  │GPU 2│GPU 3│ TP │  │  │GPU 6│GPU 7│ TP │       │
│  │  └─────┴─────┘    │  │  └─────┴─────┘    │       │
│  └───────────────────┘  └───────────────────┘       │
│       Node 0                  Node 1                │
└─────────────────────────────────────────────────────┘

TP = Tensor Parallelism (within node, over NVLink)
PP = Pipeline Parallelism (across pipeline stages)
DP = Data Parallelism (across node groups)
```

---

### Mapping onto Specific Accelerators

#### Mapping onto GPUs (Most Common)

**Frameworks:** PyTorch FSDP, DeepSpeed, Megatron-LM

```python
# Example: DeepSpeed ZeRO-3 config
{
    "zero_optimization": {
        "stage": 3,
        "offload_param": {"device": "cpu"},
        "offload_optimizer": {"device": "cpu"}
    },
    "bf16": {"enabled": true},
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 8
}
```

**Key mapping decisions:**
- Use BF16/FP16 mixed precision to halve memory and double throughput on Tensor Cores
- Flash Attention to reduce memory from O(n²) to O(n) for attention computation
- Activation checkpointing to trade compute for memory

#### Mapping onto TPUs

**Frameworks:** JAX + XLA, T5X, MaxText

- XLA compiler automatically partitions tensors across TPU cores
- Use `pjit` (partitioned JIT) with mesh sharding annotations
- TPU pods have fixed topologies (2D/3D torus), and the mapping must respect this

#### Mapping onto FPGAs

- Typically for **inference only** (training is impractical)
- Quantize model to INT8/INT4 to fit in limited on-chip memory
- Design custom pipeline: token embedding → attention → FFN → output
- Each layer is a hardware pipeline stage
- Batch size is usually 1 (latency-optimized)

---

## Practical Example: Mapping LLaMA 3 70B for Training

**Model specs:** 70B parameters, 80 transformer layers, hidden dim 8192, 64 attention heads

**Hardware:** 64x NVIDIA H100 GPUs (8 nodes × 8 GPUs/node)

**Mapping decision:**

| Dimension | Strategy | Degree | Scope |
|-----------|----------|--------|-------|
| Tensor Parallel | Split attention heads + FFN | TP=8 | Within each node (NVLink) |
| Pipeline Parallel | Split 80 layers into 2 stages | PP=2 | Across 2 nodes (InfiniBand) |
| Data Parallel | Replicate across remaining nodes | DP=4 | Across node groups |

**Verification:** TP × PP × DP = 8 × 2 × 4 = 64 GPUs ✓

**Memory per GPU:**
- Parameters: 140 GB / 8 (TP) / 2 (PP) = ~8.75 GB
- Optimizer states (Adam): ~35 GB / 8 / 2 = ~2.2 GB (with ZeRO-1 across DP)
- Activations: managed via activation checkpointing
- **Total: ~20-30 GB per GPU** — fits in 80 GB H100 ✓

---

## Key Takeaways

1. **GPUs are the dominant accelerator** for LLMs due to CUDA ecosystem maturity, high memory bandwidth, and Tensor Core throughput
2. **TPUs are competitive** for large-scale training in Google's ecosystem (JAX/XLA)
3. **FPGAs are niche** — best for low-latency edge inference with quantized models
4. **Mapping LLMs requires combining multiple parallelism strategies** (3D parallelism + ZeRO) because no single GPU has enough memory
5. **The interconnect determines the mapping** — fast NVLink within a node favors tensor parallelism; slower InfiniBand between nodes favors pipeline/data parallelism
6. **Memory optimization techniques** (mixed precision, Flash Attention, activation checkpointing, ZeRO offloading) are essential for fitting large models
