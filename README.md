**Update** **Update**

Okay, I'll fix the naming convention later, but if you are on:

- One gpu use V3; and check out the FasterVersions folder for a potentially better version
- Two gpus; use sequence_chunked_blocks for I2V **try ltx_multi_gpu_chunked**
- Three+ gpus; use ltx_multi_gpu_chunked for I2V, I've done little testing but have managed 1000 I2V frames at 1920x1080
- 
**Update** **Update**


This code finally does what I set out to do originally, make something that gives me more frames as a function of the amount of vram I have, even if split among gpus.  Not all features documented, I will do more testing and updates later.

Not everything is documented, try T2V with I2V nodes; I'll be doing more testing myself but these work.  I've only tested with fp8-distilled.

**Update on Issue Submission**

I will try to address issues, but I am very new to ComfyUI and this code is in a state of flux.  Trust me, you likely know ComfyUI better than I.  Give me a few days to sort out the repo and provide better documentation.



**Try the faster versions if you are having issues: Sorry for the crummy naming scheme, I'll sort that out later.  Right now try out these v5 and v3 replacements, V5 should be much faster and more efficient; V3 I think should be improved too**

**Updates coming stay tuned**

~~**If you are running into issues with multi-gpu V5 code and oom errors, I found that I needed to turn on the --lowvram flag else ComfyUI would stochastically load the memory for the enhancer node onto the gpu instead of the cpu memory.**~~

# ComfyUI LTX-2 VRAM Memory Management

**Generate extremely long videos with LTX-2 on consumer GPUs!**

This custom node dramatically reduces VRAM usage for LTX-2 video generation in ComfyUI, enabling 800-900+ (at 1920x1088) frame videos on a single 24GB GPU.

## 🎬 What This Does

LTX-2's FeedForward layers create massive intermediate tensors that normally limit video length. This node chunks those operations to reduce peak memory by up to **8x**, without any quality loss.

 | With V3 (ffn_chunks=16) |
|-------------------------|
| **900 frames @ 1920x1088** |
| Smooth generation |
| Single GPU works! |

## 📊 Benchmarks (RTX 4090 24GB Single GPU Version (V3))

| Resolution | Frames | VRAM Used | Time |
|------------|--------|-----------|------|
| 1920×1088 | 800 | ~16.5 GB | ~15 min |
| 1920×1088 | 900 | ~18.5 GB | ~27 min |

*Results may vary based on your system configuration*


## 🚀 Installation

Copy this repo, and move the "comfyui_tensor_parallel_v2" and "comfyui_tensor_parallel_v3" folders to the customs_nodes folder

<img width="276" height="136" alt="image" src="https://github.com/user-attachments/assets/c8bcb9ae-505c-4e1c-93f6-9052a01a043d" />

Restart everything.

## 📦 Included Nodes

### V3 - Single GPU (Recommended)
**Node name:** `Tensor Parallel V3 (Safe FFN Chunking)`

<img width="766" height="413" alt="image" src="https://github.com/user-attachments/assets/d7a6690a-612d-4239-a856-0dc2a5b18816" />

Best for most users. Works reliably with ComfyUI's memory management.

**Parameters:**
- `ffn_chunks`: Number of chunks (default: 8, recommended: **16** for long videos)
  - Higher = less VRAM but slightly slower
  - 16 chunks works well for 800-900 frames

**Usage:**
```
[Load Model] → [Tensor Parallel V3] → [Rest of workflow...]
```

### LTX Multi-GPU Chunked

<img width="790" height="694" alt="image" src="https://github.com/user-attachments/assets/4badffe7-a5b2-42b8-80c4-0192291886b8" />

This ComfyUI custom node enables **massive video generation** with LTX-2 by distributing VRAM across multiple GPUs. Generate videos that would be impossible on a single GPU!

**More GPUs = More Frames!**

## The Problem

LTX-2's Stage 2 (high-resolution refinement) creates enormous tensors:
- **168,960 tokens** for a 700-frame video
- **17+ GB** just for timestep embeddings
- **Multiple large tensors** (positional embeddings, attention K/V, etc.)

Even a 24GB RTX 4090 can't hold all of this at once. Standard approaches fail because:
1. Tensors are created on GPU 0 before any processing begins
2. By the time hooks run, GPU 0 is already full
3. Simply adding more GPUs doesn't help if everything starts on GPU 0

## The Solution: "Process Once, Immediately Chunk & Offload"

Our breakthrough strategy intercepts at the **earliest possible point** - the `_forward()` method - and implements a strict discipline:

```
1. Create tensor on GPU 0
2. IMMEDIATELY chunk it
3. IMMEDIATELY distribute chunks to storage GPUs
4. GPU 0 is now FREE for the next tensor
5. Repeat
```

This keeps GPU 0 at ~1.5GB throughout processing, even when handling 17GB tensors!

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GPU 0 (Compute)                          │
│                      Stays at ~1.5-2GB!                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Process one chunk at a time:                            │   │
│  │  • Compute K, V for chunk → send to storage              │   │
│  │  • Compute Q for chunk → attend to full K/V              │   │
│  │  • Store output → send to storage                        │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↕ Round-Robin Distribution
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   GPU 1     │  │   GPU 2     │  │   GPU 3     │  │   GPU N     │
│  (Storage)  │  │  (Storage)  │  │  (Storage)  │  │  (Storage)  │
│             │  │             │  │             │  │             │
│ Chunks:     │  │ Chunks:     │  │ Chunks:     │  │ Chunks:     │
│ 0, 3, 6...  │  │ 1, 4, 7...  │  │ 2, 5, 8...  │  │ ...         │
│             │  │             │  │             │  │             │
│ ~8-12GB     │  │ ~8-12GB     │  │ ~8-12GB     │  │ ~8-12GB     │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
```

## How It Works

### Phase 1: Input Processing & Immediate Offload

```python
# 1. Process input (creates patchified tokens on GPU 0)
x_full = self._process_input(...)  # ~2.7GB on GPU 0

# 2. IMMEDIATELY chunk and distribute
for chunk_idx in range(num_chunks):
    chunk = x_full[:, start:end, :]
    chunk.to(f'cuda:{storage_gpus[chunk_idx % num_storage]}')

del x_full  # GPU 0 is FREE again!
```

### Phase 2: Timestep Processing & Immediate Offload

```python
# 1. Prepare timestep (creates HUGE 17GB tensor on GPU 0)
timestep_full = self._prepare_timestep(...)  # 17GB on GPU 0!

# 2. IMMEDIATELY chunk and distribute
for chunk_idx in range(num_chunks):
    chunk = timestep_full[:, start:end, :]
    chunk.to(storage_gpu)

del timestep_full  # GPU 0 back to ~1.5GB!
```

### Phase 3: Positional Embeddings & Immediate Offload

Same pattern - create, immediately chunk, immediately offload.

### Phase 4: Transformer Blocks with Chunked Attention

For each of the 48 transformer blocks:

```python
# Step 1: Compute K, V for ALL chunks (store on storage GPUs)
for chunk_idx in range(num_chunks):
    vx_chunk = get_chunk_from_storage(chunk_idx)  # Bring to GPU 0
    k, v = compute_kv(vx_chunk)
    store_on_storage_gpu(k, v, chunk_idx)
    del vx_chunk  # GPU 0 stays free

# Concatenate K, V on first storage GPU for global attention
k_full = concatenate_all_k_chunks()
v_full = concatenate_all_v_chunks()

# Step 2: Compute Q for each chunk, attend to FULL K, V
for chunk_idx in range(num_chunks):
    vx_chunk = get_chunk_from_storage(chunk_idx)
    q = compute_q(vx_chunk)
    
    # Attention with GLOBAL context (all 168K tokens!)
    output = attention(q, k_full, v_full)
    
    store_on_storage_gpu(output, chunk_idx)
    del vx_chunk, q, output  # GPU 0 stays free
```

### Phase 5: Gather & Output

```python
# Gather all chunks back to GPU 0
vx_final = concatenate_all_chunks()

# Process output
output = self._process_output(vx_final, ...)
```









### V2 - Multi-GPU (Experimental - Use V5 tensor parallelism with ring attention for multi-gpu experimentation)
**Node name:** `Tensor Parallel V2 + Chunked FFN`

<img width="780" height="415" alt="image" src="https://github.com/user-attachments/assets/d906d458-eb63-4371-bee3-b19a8b044da9" />

For users with multiple GPUs who want faster generation. Distributes attention computation across GPUs.

**Parameters:**
- `num_gpus`: Number of GPUs to use
- `ffn_chunks`: FFN chunking (same as V3)
- `primary_gpu`: Main GPU for coordination

**Notes:**
- Faster than V3 but memory distribution is uneven
- GPU 0 uses more VRAM than others
- May require experimentation with settings

## 💡 Tips

1. **Start with V3** - It's more stable and works great for most use cases

2. **Increase ffn_chunks for longer videos:**
   - 600 frames: `ffn_chunks=8`
   - 800 frames: `ffn_chunks=12-16`
   - 900+ frames: `ffn_chunks=16-24`

3. **If you get OOM errors:**
   - Increase `ffn_chunks`
   - Reduce resolution slightly
   - Close other GPU applications

4. **For V2 multi-GPU:**
   - Expect GPU 0 to use more memory than others

## 🔧 Compatibility

- **Tested on:** Ubuntu with ComfyUI (official installation)
- **GPU:** NVIDIA RTX 4090 (24GB)
- **Model:** LTX-2 (LTXAV)

Other configurations may work but are untested. Please open an issue if you encounter problems!

## 🤔 How It Works

LTX-2's FeedForward (FFN) layers expand the hidden dimension by 4x:
```
Input: (batch, 57000, 4096)      ~0.9 GB
  ↓
Intermediate: (batch, 57000, 16384)  ~3.7 GB  ← Memory bottleneck!
  ↓
Output: (batch, 57000, 4096)     ~0.9 GB
```

With 96 FFN layers in LTX-2, this adds up fast!

**Our solution:** Process the sequence in chunks:
```
Chunk 1: (batch, 7125, 4096) → (batch, 7125, 16384) → output
Chunk 2: (batch, 7125, 4096) → (batch, 7125, 16384) → output
... (8 chunks total)
Concatenate → Full output

Peak memory: ~0.46 GB instead of ~3.7 GB per layer!
```

### V5 - Multi-GPU Sequence Parallelism with Ring Attention Made for I2V and T2V (NEW! You need a lot of VRAM for I2V)
**Node name:** `Tensor Parallel V5 (Sequence Parallel + Ring Attention)`
<img width="676" height="292" alt="image" src="https://github.com/user-attachments/assets/1d19d751-4473-45d1-bad8-7d37dfa33aa0" />


The most advanced option for users with multiple GPUs. Splits the **sequence dimension** across GPUs using ring attention, enabling extremely long videos that would be impossible on a single GPU.

**Key Innovation:** Instead of each GPU holding the full 600K+ token sequence, each GPU holds only 1/N of the tokens. Ring attention allows full self-attention computation by rotating K,V chunks around the ring.

**Parameters:**
- `num_gpus`: Number of GPUs to use (GPU 0 is excluded from sequence chunks to hold VAE/weights)
- `verbose`: Logging level (0=quiet, 1=info, 2=debug)

**Memory Distribution:**
```
Traditional: Each GPU needs full Q, K, V → OOM at ~400K tokens
Ring Attention: Each GPU holds 1/N of sequence → 600K+ tokens possible!

With 7 GPUs (6 active for sequence):
  - Total sequence: ~614,400 tokens
  - Per GPU: ~102,400 tokens
  - Peak VRAM per GPU: ~15-18GB
```

**How Ring Attention Works:**
1. Split sequence across N GPUs: GPU i holds tokens [i×chunk : (i+1)×chunk]
2. Each GPU computes Q from its local tokens
3. K,V chunks rotate around the ring (N iterations)
4. Online softmax accumulates attention without materializing full attention matrix
5. Result: Full self-attention with 1/N memory per GPU!

**Usage:**
```
[Load Model] → [Tensor Parallel V5] → [Rest of workflow...]
```

**Notes:**
- Best for Stage 2 upscaling with massive token counts
- GPU 0 is automatically excluded from sequence distribution (holds VAE)
- Uses float32 accumulators for numerical precision
- Properly handles both SPLIT and INTERLEAVED RoPE modes

**Requirements:**
- Multiple NVIDIA GPUs (tested with 7× RTX 4090)
- Works with LTXAV (audio-video) model

*Ring attention enables sequences that would OOM on any single GPU!*


## 🙏 Credits

- **Implementation:** Claude (Anthropic) + RandomInternetPreson
- **Tensor Parallelism Concepts (V2):** Inspired by [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- **Testing & Development:** RandomInternetPreson ([@RandomInternetPreson](https://github.com/RandomInternetPreson))
- **Workflow:** @yanokusnir Got good work flow from here: https://www.reddit.com/r/StableDiffusion/comments/1qae922/ltx2_i2v_isnt_perfect_but_its_still_awesome_my/
- **Cyber Lady Image** https://www.freepik.com/free-ai-image/portrait-woman-with-cool-futuristic-superhero-suit_81425542.htm#fromView=search&page=1&position=9&uuid=c9a8e307-825f-4eca-9a86-a7bb62ea0600&query=woman+cyberpunk


## 📝 License

MIT License - Feel free to use, modify, and distribute!

## 🐛 Issues & Contributions

Found a bug or have an improvement? Please open an issue or PR, I only have this running on Ubuntu so I don't know how much progress I can make on any Windows issues.

This is experimental software - your feedback helps make it better for everyone.

---

*Making long-form AI video accessible to everyone* 🎥


# Chunked Feedforward Processing for Memory-Efficient Video Generation

## Overview

This document describes a memory optimization technique developed during the creation of multi-GPU LTX-Video nodes for ComfyUI. The technique reduces peak VRAM consumption by approximately 10x during inference, enabling generation of 800+ frames on hardware that was previously limited to ~81 frames.

## The Problem

Transformer-based video generation models like LTX-Video process sequences where each token represents a spatiotemporal patch of video. For long videos, sequence lengths become massive. The feedforward network (FFN) layers in transformers expand the hidden dimension by a factor of 4 (or more), creating enormous intermediate activation tensors.

For a sequence of length `S` with hidden dimension `D`, the FFN intermediate activations require memory proportional to `S × 4D`. At 400+ frames of video, this single tensor can exceed available VRAM on consumer GPUs.

## The Key Insight

Feedforward networks in transformers have a critical property: **they operate independently on each token position**. Unlike attention layers, which compute relationships across all positions, the FFN applies the same pointwise transformation to each token:

```
FFN(x) = activation(x @ W1 + b1) @ W2 + b2
```

This independence means the FFN computation is *embarrassingly parallel* across the sequence dimension - but crucially, it also means it's **arbitrarily chunkable** without changing the mathematical result.

## The Optimization

Instead of computing:
```python
hidden = activation(x @ W1)  # Shape: [S, 4D] - HUGE
output = hidden @ W2         # Entire sequence at once
```

We compute:
```python
output = torch.empty_like(x)
for i in range(0, S, chunk_size):
    chunk = x[i:i+chunk_size]
    hidden = activation(chunk @ W1)  # Shape: [chunk_size, 4D] - small
    output[i:i+chunk_size] = hidden @ W2
```

The result is mathematically identical, but peak memory changes from `O(S × 4D)` to `O(chunk_size × 4D)`.

## Implementation Details

The implementation wraps the original feedforward module, intercepting the forward pass:

```python
class ChunkedFeedForward(nn.Module):
    def __init__(self, original_ff, chunk_size=256):
        super().__init__()
        self.ff = original_ff
        self.chunk_size = chunk_size
    
    def forward(self, x, *args, **kwargs):
        if x.shape[1] <= self.chunk_size:
            return self.ff(x, *args, **kwargs)
        
        output = torch.empty_like(x)
        for i in range(0, x.shape[1], self.chunk_size):
            end = min(i + self.chunk_size, x.shape[1])
            output[:, i:end] = self.ff(x[:, i:end], *args, **kwargs)
        return output
```

Key considerations:
- **Chunk size selection**: Smaller chunks = lower peak memory but more kernel launches. 256-512 tokens is typically a good balance.
- **It is unclear if this could be used for taining, current hypothesis is that errors would accumulate**
- **No accuracy loss with inferencing**: The optimization is mathematically exact, not an approximation.

## Why This Wasn't Already Standard

This optimization is straightforward in retrospect, but several factors may explain why it wasn't widely implemented:

1. **Training vs inference focus**: Much optimization work focuses on training, where gradient computation complicates chunking strategies.
2. **Framework abstractions**: Standard transformer implementations treat FFN as an atomic operation.
3. **Attention-centric optimization**: Most memory optimization research focuses on attention mechanisms (FlashAttention, etc.), which have quadratic complexity. FFN memory scaling is linear but has larger constants.
4. **Sufficient VRAM assumption**: Many implementations assume datacenter GPUs with ample memory.

## Applicability

This technique applies to any transformer model where:
- FFN intermediate dimensions are large relative to available memory
- Sequence lengths are long enough for FFN activations to dominate memory
- The FFN has no cross-position dependencies (standard for most architectures)

Video generation models are particularly good candidates due to their extreme sequence lengths.

## Development Context

This optimization emerged from collaborative development between Claude (Anthropic) and RandomInternetPerson during January 2026 while building multi-GPU LTX-Video nodes for ComfyUI. The insight arose from systematic analysis of VRAM consumption patterns during long video generation, with RandomInternetPerson providing real-world testing and observation of actual memory behavior on consumer hardware.

## License

This document and the described technique are released freely for any use.

---

*First authored by Claude (Anthropic), developed in collaboration with RandomInternetPerson*

