# LLM and HPC Study Guide

A comprehensive guide covering transformer model internals, GPU hardware, and high-performance computing for AI.

---

## Table of Contents

1. [Logits in Transformer Models](#1-logits-in-transformer-models)
2. [Cross-Entropy Loss](#2-cross-entropy-loss)
3. [LLM Inference](#3-llm-inference)
4. [Throughput vs Latency, Kernels, and FlashAttention](#4-throughput-vs-latency-kernels-and-flashattention)
5. [VRAM](#5-vram)
6. [NVLink and InfiniBand](#6-nvlink-and-infiniband)
7. [PCIe](#7-pcie)
8. [Transfer Learning vs Fine-Tuning](#8-transfer-learning-vs-fine-tuning)
9. [Gradients in LLMs](#9-gradients-in-llms)
10. [High Performance Computing (HPC)](#10-high-performance-computing-hpc)
11. [HPC for AI/LLMs Deep Dive — MPI, NCCL, SLURM, Parallel File Systems](#11-hpc-for-aillms-deep-dive)

---

## 1. Logits in Transformer Models

**Logits** are the raw, unnormalized output scores produced by the final linear layer before any probability conversion (like softmax).

### How they fit in the architecture

1. The transformer processes input tokens through attention layers and feed-forward layers, producing hidden state vectors.
2. The final hidden states are projected through a linear layer (often called the "language model head") that maps them to a vector of size `vocab_size`.
3. **These output values are the logits** — one score per token in the vocabulary.
4. Softmax is then applied to convert logits into a probability distribution over the vocabulary.

```
Hidden states  ->  Linear layer  ->  Logits (raw scores)  ->  Softmax  ->  Probabilities
[batch, dim]      [dim, vocab]     [batch, vocab]           [batch, vocab]
```

### Key points

- **Not probabilities**: Logits can be any real number (negative, zero, positive). They only become probabilities after softmax.
- **Higher logit = more likely token**: The token with the highest logit value is the model's top prediction.
- **Used in loss computation**: Cross-entropy loss is typically computed directly from logits (not probabilities) for numerical stability. PyTorch's `CrossEntropyLoss` expects logits, not softmax outputs.
- **Used in sampling**: During text generation, logits are manipulated (via temperature scaling, top-k, top-p filtering) before converting to probabilities for sampling the next token.

### Example (PyTorch)

```python
# Final layer in a transformer LM
logits = self.lm_head(hidden_states)  # shape: [batch_size, seq_len, vocab_size]

# For training: loss computed directly from logits
loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))

# For inference: convert to probabilities
probs = F.softmax(logits / temperature, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

The term "logit" originates from statistics — it's the inverse of the logistic (sigmoid) function, representing log-odds. In deep learning, it has been generalized to mean any pre-activation, unnormalized score.

---

## 2. Cross-Entropy Loss

**Cross-Entropy Loss** is the standard loss function used to train classification models, including transformer language models. It measures how far the model's predicted probability distribution is from the true distribution.

### The math

For a single example with `C` classes:

```
CrossEntropyLoss = -log(P(correct class))
```

More formally:

```
L = -sum( y_i * log(p_i) )    for i = 1..C
```

Where:
- `y_i` = 1 if `i` is the correct class, 0 otherwise (one-hot label)
- `p_i` = predicted probability for class `i` (after softmax)

Since `y` is one-hot, this simplifies to just `-log(p_correct)`.

### Intuition

| Model's confidence in correct answer | Loss |
|---------------------------------------|------|
| 0.95 (very confident, correct) | -log(0.95) = **0.05** (low) |
| 0.5 (uncertain) | -log(0.5) = **0.69** (medium) |
| 0.01 (very wrong) | -log(0.01) = **4.6** (high) |

- **Correct prediction -> low loss**
- **Wrong prediction -> high loss** (approaches infinity as confidence in correct class approaches 0)

### In transformers (language modeling)

At each position, the model predicts the next token from a vocabulary of ~50,000+ tokens. Cross-entropy loss is computed at every position:

```python
# logits: [batch, seq_len, vocab_size]
# labels: [batch, seq_len]  (the actual next token IDs)

loss = F.cross_entropy(
    logits.view(-1, vocab_size),   # flatten to [batch*seq_len, vocab_size]
    labels.view(-1)                # flatten to [batch*seq_len]
)
```

### Why logits, not probabilities?

PyTorch's `CrossEntropyLoss` combines `log_softmax + nll_loss` internally:

```
CrossEntropyLoss(logits, target) = NLLLoss(LogSoftmax(logits), target)
```

This is more **numerically stable** than computing softmax first, then taking the log. Computing `log(softmax(x))` separately can cause underflow/overflow issues.

### Relationship to perplexity

Perplexity, a common LLM evaluation metric, is just the exponentiation of cross-entropy loss:

```
Perplexity = e^(cross_entropy_loss)
```

A perplexity of 10 means the model is, on average, as uncertain as choosing uniformly among 10 options.

### Summary

- Cross-entropy loss penalizes the model for assigning low probability to the correct answer.
- It's differentiable, making it suitable for gradient-based optimization.
- Lower loss = better predictions. The model learns by minimizing this loss via backpropagation.

---

## 3. LLM Inference

**LLM Inference** is the process of using a trained language model to generate output (predictions) from input, as opposed to training where the model learns its parameters.

### Training vs Inference

| | Training | Inference |
|---|----------|-----------|
| **Goal** | Learn weights | Generate output |
| **Gradients** | Computed (backprop) | Not needed |
| **Data** | Training dataset | User prompt |
| **Cost** | Very high (days/weeks on GPUs) | Lower per request, but adds up at scale |
| **Mode** | `model.train()` | `model.eval()` |

### How LLM inference works (autoregressive generation)

LLMs generate text **one token at a time**, feeding each output back as input:

```
Prompt: "The capital of France is"

Step 1: Model input  -> "The capital of France is"
        Model output -> "Paris"    (highest probability token)

Step 2: Model input  -> "The capital of France is Paris"
        Model output -> ","

Step 3: Model input  -> "The capital of France is Paris,"
        Model output -> " known"

... and so on until a stop token or max length.
```

### The two phases

#### 1. Prefill (prompt processing)
- The entire input prompt is processed **in parallel** through the transformer.
- All prompt tokens are fed at once, producing KV (key-value) cache entries.
- This is compute-bound (lots of matrix multiplications).

#### 2. Decode (token generation)
- Tokens are generated **one at a time**, sequentially.
- Each step processes only the new token, reusing the KV cache from previous steps.
- This is memory-bandwidth-bound (loading large model weights for a single token).

```
[Prefill: process full prompt]  ->  [Decode: generate token 1]  ->  [token 2]  ->  [token 3]  -> ...
       parallel, fast                  sequential, slower per token
```

### KV Cache

The **key-value cache** stores intermediate attention results so they don't need to be recomputed at each generation step:

```
Without cache: Each new token recomputes attention over ALL previous tokens -> O(n^2)
With cache:    Reuse stored K,V vectors, only compute for the new token    -> O(n) per step
```

The KV cache is a major memory consumer — for large models, it can use tens of GBs.

### Sampling strategies

After the model produces logits, the next token is selected using a strategy:

- **Greedy**: Pick the highest-probability token. Deterministic but repetitive.
- **Temperature**: Scale logits by `T` before softmax. Higher T = more random.
- **Top-k**: Only consider the top `k` highest-probability tokens.
- **Top-p (nucleus)**: Only consider tokens whose cumulative probability reaches `p`.

```python
logits = model(input_ids)[:, -1, :]       # logits for last position
logits = logits / temperature              # scale
top_k_logits = top_k_filter(logits, k=50)  # keep top 50
probs = softmax(top_k_logits)
next_token = sample(probs)                 # randomly pick from distribution
```

### Key optimization techniques

| Technique | What it does |
|-----------|-------------|
| **KV Cache** | Avoids recomputing attention for past tokens |
| **Quantization** (INT8, INT4, GPTQ, AWQ) | Reduces model size and memory bandwidth |
| **Batching** (continuous/dynamic) | Processes multiple requests together for throughput |
| **Speculative decoding** | Small model drafts tokens, large model verifies in parallel |
| **Flash Attention** | Memory-efficient attention kernel |
| **Tensor parallelism** | Splits model across multiple GPUs |
| **PagedAttention** (vLLM) | Manages KV cache memory like virtual memory pages |

### Why inference is challenging at scale

- **Latency**: Users expect fast responses. The sequential decode phase is the bottleneck.
- **Memory**: A 70B parameter model in FP16 needs ~140 GB just for weights, plus KV cache.
- **Cost**: Serving millions of requests requires significant GPU infrastructure.
- **Throughput vs latency tradeoff**: Larger batches improve throughput but increase per-request latency.

---

## 4. Throughput vs Latency, Kernels, and FlashAttention

### Throughput vs Latency

#### Definitions

**Latency** — How long a single request takes from start to finish (measured in milliseconds/seconds).

**Throughput** — How many requests (or tokens) the system processes per unit of time (measured in requests/sec or tokens/sec).

#### The tradeoff

Imagine a restaurant:
- **Low latency**: A chef cooks one order at a time. Your food comes fast, but the restaurant serves few customers/hour.
- **High throughput**: The chef batches orders — cooks 20 steaks at once. Each individual customer waits longer, but the restaurant serves far more customers/hour.

LLM serving works the same way:

```
Batch size = 1:   Latency = 50ms/token,  Throughput = 20 tokens/sec
Batch size = 8:   Latency = 80ms/token,  Throughput = 100 tokens/sec
Batch size = 32:  Latency = 200ms/token, Throughput = 160 tokens/sec
```

As you increase batch size:
- **Throughput goes up** — GPU does more useful work per cycle (better utilization)
- **Latency goes up** — each individual request shares GPU resources and waits longer

#### Why batching helps GPU utilization

The decode phase is **memory-bandwidth-bound**: the GPU loads huge weight matrices but only computes for one token. Most of the GPU's compute cores sit idle.

```
Batch=1:  Load 140GB of weights -> multiply with 1 token vector   -> most compute wasted
Batch=32: Load 140GB of weights -> multiply with 32 token vectors -> much better utilization
```

You load the weights once but do 32x the useful work. This is why throughput improves.

#### How to balance the tradeoff

| Strategy | How it works |
|----------|-------------|
| **Continuous batching** | Don't wait for all requests to finish. As one request completes, slot in a new one immediately. Reduces wasted slots. |
| **Priority queues** | Give interactive users low-latency small batches; batch offline/bulk jobs together for throughput. |
| **Speculative decoding** | Small model generates draft tokens quickly, large model verifies multiple tokens in one forward pass. Improves latency without hurting throughput. |
| **SLA-based scheduling** | Set a max latency target (e.g., 100ms/token). Increase batch size only up to the point where the SLA is still met. |

### What is a kernel?

In the context of GPU computing, a **kernel** is a function that runs on the GPU.

```
CPU code (Python/C++)          GPU kernel
-----------------------        -----------
for i in range(1000):    ->    Launch 1000 GPU threads,
    a[i] = b[i] + c[i]        each computes one a[i] = b[i] + c[i]
                               in parallel
```

Key points:
- A kernel is launched from CPU code but executes on the GPU across thousands of threads simultaneously.
- Frameworks like PyTorch call pre-written CUDA kernels under the hood when you do tensor operations.
- You can write custom kernels (in CUDA, Triton, etc.) for operations that need special optimization.

```python
# This single Python line launches a GPU kernel internally
result = torch.matmul(A, B)
# PyTorch dispatches to a CUDA kernel like cublasSgemm
```

#### Standard vs custom kernels

```
Standard attention (PyTorch):
  Q @ K^T  ->  kernel 1 (matmul)
  / sqrt(d) -> kernel 2 (scale)
  softmax   -> kernel 3 (softmax)
  @ V       -> kernel 4 (matmul)

  Each kernel reads/writes to GPU global memory (HBM) between steps.
  4 round trips to slow memory.

Fused custom kernel (FlashAttention):
  All 4 operations in ONE kernel.
  Data stays in fast on-chip SRAM.
  1 round trip to slow memory.
```

### Memory-Efficient Attention Kernel (FlashAttention)

#### The problem with standard attention

Standard self-attention computes:

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
```

The intermediate matrix `Q @ K^T` has shape `[seq_len, seq_len]`. For a sequence of 8,192 tokens:

```
8192 x 8192 x 2 bytes (FP16) = 128 MB  per head per batch item
x 32 heads x batch size = GBs of memory
```

This matrix must be **stored in GPU global memory (HBM)**, which is:
- **Slow** — HBM bandwidth is the bottleneck
- **Limited** — takes memory away from KV cache and model weights

#### GPU memory hierarchy

```
+----------------------------+
|   SRAM (on-chip)           |  ~20 MB     Very fast (19 TB/s)
|   (shared memory/L1)       |
+----------------------------+
|   HBM (off-chip)           |  ~80 GB     Slow (3 TB/s)
|   (global memory)          |
+----------------------------+
```

SRAM is ~1000x faster but ~4000x smaller than HBM.

#### How FlashAttention works

Instead of materializing the full `[seq_len, seq_len]` attention matrix in HBM, FlashAttention processes it in **tiles** that fit in SRAM:

```
Standard attention:
  1. Compute S = Q @ K^T         -> write full [N,N] matrix to HBM
  2. Read S, compute softmax(S)  -> write back to HBM
  3. Read softmax result, x V    -> write output to HBM
  Total HBM reads/writes: O(N^2)

FlashAttention:
  1. Load a TILE of Q, K, V into SRAM (fits in ~20MB)
  2. Compute attention for that tile entirely in SRAM
  3. Write only the final output tile to HBM
  4. Repeat for next tile
  Total HBM reads/writes: O(N^2/SRAM_size) -- much less
```

The trick for softmax across tiles is maintaining running statistics (max and sum) using the **online softmax** algorithm, so you never need the full row at once.

#### FlashAttention Results

| | Standard Attention | FlashAttention |
|---|---|---|
| **Memory** | O(N^2) — stores full attention matrix | O(N) — only tile-sized buffers |
| **HBM access** | Many round trips | Minimized |
| **Speed** | Bottlenecked by memory bandwidth | 2-4x faster |
| **Max sequence length** | Limited by GPU memory | Much longer sequences possible |

#### In practice

```python
# PyTorch 2.0+ has FlashAttention built in
import torch.nn.functional as F

# Automatically uses FlashAttention when possible
output = F.scaled_dot_product_attention(query, key, value)
```

---

## 5. VRAM

**VRAM** (Video RAM) is the **memory** on a GPU. It is not the GPU itself — it's one component of it.

Think of it like a computer:
- **CPU** = the processor (does computation)
- **RAM** = the memory (stores data the processor is working on)

A GPU has the same split:
- **GPU cores** = the processor (thousands of parallel compute units)
- **VRAM (HBM/GDDR)** = the memory (stores data the GPU cores are working on)

```
+-----------------------------------+
|           GPU Card                |
|                                   |
|  +-----------+   +-----------+   |
|  | GPU Cores |   |   VRAM    |   |
|  | (compute) |<->|  (memory) |   |
|  |           |   |           |   |
|  | Thousands |   | 24-80 GB  |   |
|  | of ALUs   |   | HBM/GDDR |   |
|  +-----------+   +-----------+   |
|                                   |
+-----------------------------------+
```

### What is stored in VRAM during LLM inference?

| What | Size example (LLaMA 70B, FP16) |
|------|-------------------------------|
| **Model weights** | ~140 GB |
| **KV cache** | 1-30+ GB (depends on batch size and sequence length) |
| **Activations** | A few GB |
| **Input/output buffers** | Small |

This is why large models need multiple GPUs — a single GPU maxes out at 24 GB (consumer) or 80 GB (data center).

### VRAM vs System RAM

- **System RAM**: Used by CPU. Cheap, large (64-512 GB common).
- **VRAM**: Used by GPU. Expensive, smaller (24-80 GB typical), but much higher bandwidth.

| | System RAM (DDR5) | VRAM (HBM3) |
|---|---|---|
| **Capacity** | 64-512 GB | 24-80 GB |
| **Bandwidth** | ~50 GB/s | ~3,000 GB/s |
| **Cost** | ~$5/GB | ~$50+/GB |
| **Used by** | CPU | GPU |

The GPU **cannot directly use system RAM** for computation (at full speed). Data must be in VRAM. This is why "out of VRAM" errors are so common.

### Common GPU VRAM sizes

| GPU | VRAM | Typical use |
|-----|------|-------------|
| RTX 4090 | 24 GB | Run 7B-13B models, fine-tune small models |
| A100 | 40 or 80 GB | Run up to ~30-65B models |
| H100 | 80 GB | Production LLM serving |
| Multiple H100s (8x) | 640 GB total | Run 70B+ models, large batch inference |

### Why VRAM is the main bottleneck for LLMs

A 70B parameter model in FP16 = 140 GB just for weights. A single 80 GB GPU can't hold it. Options:

- **Quantization**: Shrink to INT4 -> 70B x 0.5 bytes = ~35 GB. Now it fits on one 40GB GPU.
- **Tensor parallelism**: Split the model across multiple GPUs.
- **Offloading**: Store some weights in system RAM, move to VRAM as needed. Works but slow due to the PCIe bottleneck.

---

## 6. NVLink and InfiniBand

### NVLink

**NVLink** is a high-speed direct connection between GPUs (made by NVIDIA), bypassing the slow PCIe bus.

#### The problem NVLink solves

When a model is split across multiple GPUs (tensor parallelism), GPUs need to constantly exchange data. Through PCIe, this is slow:

```
Standard PCIe connection:
GPU 0 <---- PCIe (64 GB/s) ----> GPU 1

Each GPU has ~3,000 GB/s internal bandwidth.
Communicating through PCIe at 64 GB/s is a massive bottleneck.
```

#### NVLink solution

```
NVLink connection:
GPU 0 <---- NVLink (900 GB/s) ----> GPU 1

~14x faster than PCIe.
```

In a server like the NVIDIA DGX H100 (8 GPUs), all GPUs are connected via NVLink through an **NVSwitch**, and any GPU can talk to any other GPU at 900 GB/s.

#### Why this matters for LLMs

With tensor parallelism, each GPU holds a **slice** of every layer. At each layer, GPUs must synchronize via an **all-reduce** operation:

```
Without NVLink (PCIe):
  Layer 1: compute 5ms -> sync 20ms -> Layer 2: compute 5ms -> sync 20ms ...
  Communication dominates. Most time spent waiting.

With NVLink:
  Layer 1: compute 5ms -> sync 1.5ms -> Layer 2: compute 5ms -> sync 1.5ms ...
  Communication is fast. GPU cores stay busy.
```

#### NVLink generations

| Generation | Bandwidth (per GPU) | Used in |
|-----------|-------------------|---------|
| NVLink 3 | 600 GB/s | A100 |
| NVLink 4 | 900 GB/s | H100 |
| NVLink 5 | 1,800 GB/s | B200 |

### InfiniBand

**InfiniBand** is a high-speed **network** connection between separate machines (nodes/servers). NVLink connects GPUs within a single machine. InfiniBand connects machines to each other.

#### The hierarchy

```
Within one server (node):                Between servers:
+------------------------+               +-----------+     +-----------+
|  GPU <--NVLink--> GPU  |<--InfiniBand->|  Server   |<--->|  Server   |
|  GPU <--NVLink--> GPU  |    (network)  |    2      |     |    3      |
|  GPU <--NVLink--> GPU  |               +-----------+     +-----------+
|  GPU <--NVLink--> GPU  |
|     Server 1           |
+------------------------+
```

#### InfiniBand vs Ethernet

| | Standard Ethernet | InfiniBand (HDR/NDR) |
|---|---|---|
| **Bandwidth** | 25-100 Gbps | 200-400 Gbps |
| **Latency** | ~10-50 us | ~0.5-1 us |
| **CPU overhead** | High (kernel stack) | Near zero (RDMA) |
| **Designed for** | General networking | HPC / AI clusters |

#### RDMA — The key advantage

InfiniBand supports **RDMA (Remote Direct Memory Access)**. One machine's GPU can read/write directly into another machine's GPU memory **without involving the CPU or operating system**:

```
Standard Ethernet:
  GPU -> CPU -> OS kernel -> Network stack -> NIC -> wire -> NIC -> Network stack -> OS -> CPU -> GPU
  Many copies, high latency, CPU busy.

InfiniBand with RDMA (GPUDirect):
  GPU -> NIC -> wire -> NIC -> GPU
  Direct path. No CPU involvement. Minimal latency.
```

#### Parallelism strategy mapping

```
Within node:   Tensor Parallelism   (NVLink, heavy traffic)
Between nodes: Pipeline / Data Parallelism  (InfiniBand, less frequent traffic)
```

- **Tensor parallelism** (split layers across GPUs): Needs highest bandwidth -> NVLink within a node.
- **Pipeline parallelism** (split layers between nodes): Less frequent communication -> InfiniBand between nodes.
- **Data parallelism** (gradient sync): Periodic all-reduce -> InfiniBand between nodes.

### NVLink vs InfiniBand Summary

| | NVLink | InfiniBand |
|---|---|---|
| **Connects** | GPUs within a server | Servers to each other |
| **Bandwidth** | 900-1800 GB/s | 50 GB/s (400 Gbps) |
| **Scope** | Intra-node | Inter-node (cluster) |
| **Used for** | Tensor parallelism | Data/pipeline parallelism |
| **Key feature** | Direct GPU-to-GPU link | RDMA, low-latency networking |

---

## 7. PCIe

**PCIe** (Peripheral Component Interconnect Express) is the standard interface that connects expansion cards (GPUs, SSDs, network cards, etc.) to the CPU and motherboard. It's the "highway" inside your computer that components use to talk to each other.

### Lanes

PCIe bandwidth scales with the number of **lanes**. Each lane is an independent data path. GPUs typically use **x16** (the widest) for maximum bandwidth.

### PCIe generations

| Generation | Per lane (each direction) | x16 total (each direction) | Common era |
|-----------|--------------------------|---------------------------|------------|
| PCIe 3.0 | 1 GB/s | 16 GB/s | 2012-2017 |
| PCIe 4.0 | 2 GB/s | 32 GB/s | 2017-2022 |
| PCIe 5.0 | 4 GB/s | 64 GB/s | 2022-present |
| PCIe 6.0 | 8 GB/s | 128 GB/s | Coming soon |

### Why PCIe is a bottleneck for AI

Compare the bandwidths:

```
GPU internal (VRAM <-> GPU cores):    3,000 GB/s  (HBM3)
NVLink (GPU <-> GPU, same server):      900 GB/s
PCIe 5.0 x16 (GPU <-> CPU):              64 GB/s  <-- 47x slower than HBM
```

This creates real problems:

**1. Multi-GPU without NVLink**: Consumer GPUs (like RTX 4090) don't have NVLink. They must communicate through PCIe (64 GB/s) instead of NVLink (900 GB/s).

**2. CPU-GPU data transfer**: Loading data from system RAM to VRAM goes through PCIe. If the GPU processes data faster than PCIe can deliver it, the GPU sits idle ("data starved").

**3. CPU offloading**: Storing model weights in system RAM and fetching them back goes through PCIe:

```
70B model, weights in system RAM, offloading to GPU as needed:
140 GB / 64 GB/s = ~2.2 seconds just to load all weights once

vs. if weights are in VRAM:
140 GB / 3000 GB/s = ~0.047 seconds
```

### Speed comparison

```
HBM3 (GPU internal)  ================================  3,000 GB/s
NVLink 4             ==========================         900 GB/s
PCIe 5.0 x16         ====                                64 GB/s
InfiniBand NDR       ==                                  50 GB/s
Ethernet 100GbE      =                                   12.5 GB/s
```

---

## 8. Transfer Learning vs Fine-Tuning

These terms are closely related — fine-tuning is actually a **type** of transfer learning.

### Transfer Learning (the broad concept)

**Transfer learning** is the idea of taking knowledge learned from one task and applying it to a different task.

```
Task A (source)                    Task B (target)
-------------------                -------------------
Trained on massive                 Your specific task
general dataset                    with limited data

e.g. ImageNet (14M images)         e.g. Classifying 500 X-ray images
e.g. Internet text (trillions)     e.g. Legal document summarization
```

### Methods of Transfer Learning

#### 1. Feature Extraction (freeze everything)

Use the pretrained model as a fixed feature extractor. Only train a new head on top.

```
+---------------------+
|  New classifier head | <-- Only this is trained
+---------------------+
|                     |
|  Pretrained layers  | <-- All FROZEN (weights don't change)
|  (feature extractor)|
|                     |
+---------------------+
```

```python
model = torchvision.models.resnet50(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer with your own
model.fc = nn.Linear(2048, num_your_classes)
# Only model.fc will be trained
```

**When to use**: Very small dataset, or your task is similar to the original task.

#### 2. Fine-Tuning (unfreeze and retrain)

Start from pretrained weights, then **continue training** some or all layers on your data.

```
+---------------------+
|  New classifier head | <-- Trained
+---------------------+
|                     |
|  Pretrained layers  | <-- UNFROZEN, weights updated
|                     |    (with small learning rate)
+---------------------+
```

```python
model = torchvision.models.resnet50(pretrained=True)
model.fc = nn.Linear(2048, num_your_classes)

# All parameters are trainable
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # small LR
```

**When to use**: Moderate amount of data and want the model to adapt its internal representations.

#### 3. Partial Fine-Tuning

Freeze early layers (general features) and fine-tune later layers (task-specific features).

### Side-by-side comparison

| | Feature Extraction | Fine-Tuning |
|---|---|---|
| **Pretrained weights** | Frozen | Updated |
| **What's trained** | Only the new head | Some or all layers + new head |
| **Learning rate** | Normal (only new layers) | Small (to not destroy pretrained knowledge) |
| **Data needed** | Less | More |
| **Training time** | Fast | Slower |
| **Risk of overfitting** | Lower | Higher (more params to train) |
| **Performance ceiling** | Lower | Higher |

### In the context of LLMs

#### Full fine-tuning
Update **all** parameters of the model on your dataset. Requires enormous compute and memory. Most flexible, best results.

#### Parameter-efficient fine-tuning (PEFT)

Only update a **small subset** of parameters.

**LoRA (Low-Rank Adaptation)** — the most popular method:

```
Original weight matrix W:  [4096 x 4096] = 16M parameters  (Frozen)
LoRA adapter:  A [4096 x 16] x B [16 x 4096] = 131K parameters  (Trained)

Less than 1% of original parameters.
```

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# trainable: 6.5M / 7B total (0.09%)
```

#### Prompt tuning / in-context learning

Don't change the model at all. Just provide examples in the prompt.

### LLM methods spectrum

```
More parameters changed --------------------------------> Fewer parameters changed

Full            Partial         LoRA /        Prompt         In-context
fine-tune       fine-tune       Adapters      tuning         learning
(all params)    (top layers)    (<1% params)  (soft prompts) (zero params)

More data needed ----------------------------------------> Less data needed
Better adaptation ----------------------------------------> Less adaptation
More compute -------------------------------------------->  Less compute
```

---

## 9. Gradients in LLMs

A **gradient** is the measure of how much the loss (error) changes when you slightly adjust a weight. It tells the model **which direction** and **how much** to update each parameter to reduce the error.

### Core intuition

Imagine you're blindfolded on a hilly landscape, trying to reach the lowest valley (minimum loss). The gradient tells you which direction is downhill and how steep the slope is. You move opposite to the gradient (gradient descent).

### The math (simplified)

For a single weight `w`:

```
gradient = dLoss / dw

"How much does the loss change when I nudge this weight slightly?"
```

Update rule:

```
w_new = w_old - learning_rate x gradient
```

- If gradient is **positive**: increasing `w` increases loss -> decrease `w`
- If gradient is **negative**: increasing `w` decreases loss -> increase `w`
- If gradient is **near zero**: weight is already near a good value

### Backpropagation — How gradients are computed

An LLM has billions of weights organized in layers. Gradients flow **backward** from the loss through each layer using the **chain rule** of calculus.

```
Forward pass (compute loss):
Input -> Layer 1 -> Layer 2 -> ... -> Layer N -> Loss

Backward pass (compute gradients):
Input <- dL/dw1 <- dL/dw2 <- ... <- dL/dwN <- Loss
  gradients flow backward (backpropagation)
```

### What gradients look like in an LLM

A 7B parameter model has 7 billion weights. Backpropagation computes 7 billion gradients — one per weight.

### Memory implication

This is why training uses far more memory than inference:

| Component | Size (7B model, FP16) | Why needed |
|-----------|----------------------|------------|
| Weights | 14 GB | The model itself |
| Gradients | 14 GB | Direction to update each weight |
| Adam optimizer (m, v) | 28 GB | Momentum and variance per weight |
| Activations | Variable | Saved from forward pass, needed for backprop |
| **Total** | **~66+ GB** | **vs 14 GB for inference** |

### Gradient problems

#### Vanishing gradients

As gradients chain through many layers, they can shrink toward zero. Early layers barely learn.

**Solutions**: Residual connections (skip connections), layer normalization — both used in transformers.

```
Transformer residual connection:
output = LayerNorm(x + Attention(x))
                   ^
                   skip connection: gradient flows directly
                   through addition, avoiding vanishing
```

#### Exploding gradients

The opposite — gradients grow uncontrollably large, weights get wild updates.

**Solution**: Gradient clipping — cap the gradient magnitude:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Gradients in distributed training

When training across multiple GPUs, each GPU computes gradients on its own batch of data. These must be **averaged** across all GPUs before updating weights — this is the **all-reduce** operation.

### Gradients during fine-tuning vs inference

| | Training / Fine-tuning | Inference |
|---|---|---|
| Gradients computed? | Yes | No |
| Backpropagation? | Yes | No |
| Weights updated? | Yes | No |
| Memory for gradients? | Required | Not needed |

```python
# Training: gradients ON
model.train()
loss = model(input, labels)
loss.backward()          # compute gradients
optimizer.step()         # update weights using gradients

# Inference: gradients OFF
model.eval()
with torch.no_grad():   # skip gradient computation -> saves memory
    output = model(input)
```

---

## 10. High Performance Computing (HPC)

**HPC** is the use of aggregated computing power — many processors working together — to solve problems that are too large or too slow for a single machine.

### What makes a system "HPC"

Key components:
- **Many nodes** (servers) — hundreds to thousands
- **Fast interconnect** — InfiniBand, not regular Ethernet
- **Parallel file system** — storage that thousands of nodes can read/write simultaneously
- **Job scheduler** — software (like SLURM) that allocates resources to users

### Scale reference

| System type | Typical scale | Performance |
|---|---|---|
| Laptop | 1 CPU, maybe 1 GPU | ~1 TFLOPS |
| Workstation | 1-2 CPUs, 1-4 GPUs | ~10 TFLOPS |
| Small cluster | 10-50 nodes | ~1 PFLOPS |
| Supercomputer | 1,000-10,000+ nodes | 1-2,000+ PFLOPS |

*(TFLOPS = trillion floating point operations per second, PFLOPS = 1,000 TFLOPS)*

### Core HPC concepts

#### 1. Parallelism

```
Data parallelism:     Same operation on different chunks of data
Task parallelism:     Different operations run simultaneously
Pipeline parallelism: Different stages of a pipeline on different processors
```

#### 2. MPI (Message Passing Interface)

The standard programming model for HPC. Processes on different nodes communicate by sending messages.

#### 3. Job scheduling (SLURM)

HPC clusters are shared resources. Users submit jobs, and a scheduler allocates nodes.

#### 4. Parallel file systems

Thousands of nodes reading data simultaneously would overwhelm a normal file system. HPC uses distributed storage.

### HPC and AI/LLMs

Modern LLM training **is** HPC. The techniques are the same:

| HPC concept | LLM training equivalent |
|---|---|
| MPI all-reduce | Gradient synchronization across GPUs |
| Data parallelism | Each GPU trains on different data batches |
| Domain decomposition | Tensor parallelism (split layers across GPUs) |
| Pipeline processing | Pipeline parallelism (split model stages) |
| Job scheduler (SLURM) | Managing training runs on GPU clusters |
| InfiniBand / RDMA | Fast gradient exchange between nodes |
| Parallel I/O | Loading training data at scale |

### Traditional HPC vs AI workloads

| | Traditional HPC | AI / LLM training |
|---|---|---|
| **Workload** | Physics simulations, weather, molecular dynamics | Neural network training |
| **Precision** | FP64 (double precision) | FP16 / BF16 (half precision) |
| **Key hardware** | CPUs (historically) | GPUs / accelerators |
| **Communication** | MPI | NCCL + MPI |
| **Bottleneck** | Compute or memory bandwidth | Memory capacity (VRAM) + interconnect |

---

## 11. HPC for AI/LLMs Deep Dive

### MPI (Message Passing Interface)

MPI is the foundational communication standard in HPC. It defines how processes on different machines exchange data. Each process gets a **rank** (ID) and can send/receive data to/from any other process.

#### Key MPI operations

**Point-to-point** — one process sends to another.

**Collective operations** — all processes participate together:

- **Broadcast**: One process sends the same data to all others.
- **Scatter**: One process splits data and distributes pieces.
- **Gather**: Opposite of scatter. Collect pieces into one.
- **Reduce**: Combine data from all processes using an operation (sum, max, etc).
- **All-Reduce**: Reduce + broadcast result to everyone.

#### All-Reduce — the most important operation for LLM training

In data parallelism, each GPU computes gradients on its own data batch. **All-reduce** averages these gradients across all GPUs so every GPU has the same averaged gradient and updates weights identically.

```
Step 1: Each GPU computes local gradients
  GPU 0: grad = [0.1, -0.3, 0.5]    (from batch 0)
  GPU 1: grad = [0.2, -0.1, 0.3]    (from batch 1)
  GPU 2: grad = [0.0, -0.4, 0.6]    (from batch 2)
  GPU 3: grad = [0.3, -0.2, 0.4]    (from batch 3)

Step 2: All-reduce (average)
  All GPUs: avg_grad = [0.15, -0.25, 0.45]

Step 3: Each GPU updates weights with the same gradient
  All GPUs: w = w - lr x [0.15, -0.25, 0.45]
  -> Weights stay synchronized across all GPUs
```

#### MPI in PyTorch

```python
import torch.distributed as dist

dist.init_process_group(backend="mpi")

rank = dist.get_rank()
world_size = dist.get_world_size()

loss = model(data[rank])
loss.backward()

for param in model.parameters():
    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
    param.grad /= world_size

optimizer.step()
```

#### Ring All-Reduce algorithm

Naively, all-reduce would require every GPU to send its full gradient to every other GPU — O(N^2) communication. **Ring all-reduce** does it in O(N) by arranging GPUs in a ring:

1. **Reduce-Scatter phase**: Each GPU sends one chunk to the next, accumulating sums. After N-1 steps, each GPU holds the fully reduced version of one chunk.
2. **All-Gather phase**: Each GPU sends its completed chunk around the ring. After N-1 steps, every GPU has the full reduced result.

Total data transferred per GPU: `2 x (N-1)/N x data_size` — this is optimal and independent of the number of GPUs.

### NCCL (NVIDIA Collective Communication Library)

**NCCL** (pronounced "nickel") is NVIDIA's GPU-optimized replacement for MPI collectives. While MPI is general-purpose (CPU-centric), NCCL is specifically designed for GPU-to-GPU communication.

#### Why not just use MPI?

```
MPI all-reduce path:
  GPU VRAM -> CPU RAM -> MPI library -> Network -> CPU RAM -> GPU VRAM
  Multiple memory copies, CPU involvement at every step.

NCCL all-reduce path:
  GPU VRAM -> NVLink/PCIe/InfiniBand -> GPU VRAM
  Direct GPU-to-GPU. No CPU copies. GPU-initiated transfers.
```

#### NCCL topology awareness

NCCL automatically detects the hardware topology and chooses the optimal communication path — using NVLink for intra-node communication and InfiniBand for inter-node communication.

#### NCCL vs MPI

| | MPI | NCCL |
|---|---|---|
| **Designed for** | CPU processes | GPU processes |
| **Data location** | CPU memory | GPU memory (VRAM) |
| **GPU transfers** | Goes through CPU | Direct GPU-to-GPU |
| **Topology aware** | Manual configuration | Auto-detects NVLink, PCIe, IB |
| **Vendor** | Open standard (many implementations) | NVIDIA only |
| **Works with AMD?** | Yes | No (AMD has RCCL) |

#### Typical LLM training stack

```
Application layer:    PyTorch DDP / FSDP / DeepSpeed
Communication layer:  NCCL (GPU collectives)
Transport layer:      NVLink (intra-node) + InfiniBand RDMA (inter-node)
Hardware:             GPUs + NVSwitch + InfiniBand NICs
```

### Job Scheduler (SLURM)

**SLURM** (Simple Linux Utility for Resource Management) is the dominant job scheduler in HPC. It manages who gets to use which nodes and when.

#### SLURM commands

```bash
sbatch train.sh              # Submit a job
squeue                       # Check the queue
squeue -u $USER              # Check your jobs
scancel 1004                 # Cancel a job
srun --nodes=1 --gpus=4 --time=01:00:00 --pty bash  # Interactive session
sinfo                        # Check cluster status
```

#### A typical LLM training SLURM script

```bash
#!/bin/bash
#SBATCH --job-name=llama-70b-pretrain
#SBATCH --partition=gpu
#SBATCH --nodes=64                      # 64 servers
#SBATCH --gpus-per-node=8               # 8 GPUs per server = 512 GPUs total
#SBATCH --cpus-per-task=12              # CPU cores for data loading
#SBATCH --mem=480G                      # RAM per node
#SBATCH --time=168:00:00               # 7 days max
#SBATCH --output=logs/%j.out           # stdout
#SBATCH --error=logs/%j.err            # stderr
#SBATCH --exclusive                    # no sharing nodes

module load cuda/12.2
module load nccl/2.18
source activate training_env

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500
export NCCL_IB_DISABLE=0              # enable InfiniBand
export NCCL_NET_GDR_LEVEL=2           # enable GPUDirect RDMA

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=8 \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
        --model llama-70b \
        --batch_size 4 \
        --gradient_accumulation_steps 8 \
        --learning_rate 1.5e-4
```

#### Scheduling policies

- **FIFO**: First come, first served.
- **Fair-share**: Users/groups get proportional access based on historical usage.
- **Backfill**: While a large job waits, smaller jobs that will finish in time can run in the gaps.
- **Preemption**: High-priority jobs can pause/kill lower-priority jobs.

#### Fault tolerance with SLURM

LLM training runs for days/weeks. Hardware failures are inevitable at scale. SLURM integrates with checkpointing to save model state and auto-requeue failed jobs.

### Parallel File Systems

#### The problem

```
512 nodes, each with 8 GPUs, training an LLM.
Each GPU needs to read training data at ~1 GB/s.
Total read bandwidth needed: 512 x 8 x 1 GB/s = 4 TB/s

A single NFS server: ~1-10 GB/s  (400-4000x too slow)
```

#### Architecture (Lustre)

**Lustre** is the most common parallel file system in HPC.

- **MDS (Metadata Server)**: Stores file names, permissions, directory structure
- **MGS (Management Server)**: Stores config and cluster map
- **OSS (Object Storage Servers)**: Store actual file data, striped across many servers

#### How file striping works

A large file is split into chunks (stripes) distributed across many storage servers:

```
training_data.bin (10 TB file, stripe size = 1 MB)

OSS 0:   [chunk 0] [chunk 4] [chunk 8]  [chunk 12] ...
OSS 1:   [chunk 1] [chunk 5] [chunk 9]  [chunk 13] ...
OSS 2:   [chunk 2] [chunk 6] [chunk 10] [chunk 14] ...
OSS 3:   [chunk 3] [chunk 7] [chunk 11] [chunk 15] ...

When a node reads bytes 0-4MB:
  -> Reads from OSS 0, 1, 2, 3 in PARALLEL -> 4x bandwidth
```

With 100 OSS servers, you get up to 100x the bandwidth of a single server.

#### Configuring stripe settings for AI workloads

```bash
# Large files (training data): wide striping
lfs setstripe -c -1 /lustre/large_files/    # stripe across ALL OSTs

# Many small files (checkpoints): narrow striping
lfs setstripe -c 1 /lustre/checkpoints/     # 1 OST per file
```

#### Parallel file systems compared

| | Lustre | GPFS | BeeGFS | NFS (baseline) |
|---|---|---|---|---|
| **Max throughput** | TB/s | TB/s | Hundreds GB/s | ~10 GB/s |
| **Max capacity** | Exabytes | Exabytes | Petabytes | Terabytes |
| **Concurrent clients** | 100,000+ | 100,000+ | 10,000+ | ~100 |
| **Striping** | Yes | Yes | Yes | No |

#### Checkpoint storage

```
LLaMA 70B checkpoint:
  Model weights:        140 GB
  Optimizer states:     280 GB
  Other state:           10 GB
  Total per checkpoint: ~430 GB

  Without parallel FS: 430 GB / 10 GB/s  = 43 seconds (blocks training)
  With Lustre:         430 GB / 500 GB/s = ~0.9 seconds
```

### Full LLM Training Stack

```
Application:    Training script (PyTorch + DeepSpeed/FSDP)
Communication:  NCCL (GPU collectives)
Intra-node:     NVLink/NVSwitch (900 GB/s)
Inter-node:     InfiniBand RDMA (400 Gbps)
Storage:        Lustre parallel file system (TB/s)
Scheduling:     SLURM (node allocation, job queue, failure handling)
Monitoring:     Weights & Biases / TensorBoard
```

### A real training run lifecycle

1. **PREPARE**: Tokenized dataset on Lustre, submit job via `sbatch`
2. **QUEUE**: SLURM evaluates priority, waits for nodes
3. **LAUNCH**: SLURM allocates nodes, NCCL initializes, model weights broadcast
4. **TRAIN**: Forward/backward pass, NCCL all-reduce, optimizer step, periodic checkpoints
5. **FAILURE**: Hardware error detected, SLURM requeues, restart from last checkpoint
6. **COMPLETE**: Final model saved, nodes released, next queued job starts
