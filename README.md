# ComfyUI LTX-2 VRAM Memory Management

**Generate extremely long videos with LTX-2 on consumer GPUs!**

This custom node dramatically reduces VRAM usage for LTX-2 video generation in ComfyUI, enabling 800-900+ (at 1920x1088) frame videos on a single 24GB GPU.

## 🎬 What This Does

LTX-2's FeedForward layers create massive intermediate tensors that normally limit video length. This node chunks those operations to reduce peak memory by up to **8x**, without any quality loss.

| Without This Node | With V3 (ffn_chunks=16) |
|-------------------|-------------------------|
| ~300 frames max | **900 frames @ 1920x1088** |
| OOM errors | Smooth generation |
| Need multi-GPU | Single GPU works! |

## 📊 Benchmarks (RTX 4090 24GB Single GPU Version (V3))

| Resolution | Frames | VRAM Used | Time |
|------------|--------|-----------|------|
| 1920×1088 | 800 | ~16.5 GB | ~15 min |
| 1920×1088 | 900 | ~18.5 GB | ~27 min |

*Results may vary based on your system configuration*

## 🚀 Installation

1. Navigate to your ComfyUI custom nodes folder:
   ```bash
   cd ComfyUI/custom_nodes/
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/RandomInternetPreson/ComfyUI_LTX-2_VRAM_Memory_Management.git
   ```

3. Restart ComfyUI

## 📦 Included Nodes

### V3 - Single GPU (Recommended)
**Node name:** `Tensor Parallel V3 (Safe FFN Chunking)`

Best for most users. Works reliably with ComfyUI's memory management.

**Parameters:**
- `ffn_chunks`: Number of chunks (default: 8, recommended: **16** for long videos)
  - Higher = less VRAM but slightly slower
  - 16 chunks works well for 800-900 frames

**Usage:**
```
[Load Model] → [Tensor Parallel V3] → [Rest of workflow...]
```

### V2 - Multi-GPU (Experimental)
**Node name:** `Tensor Parallel V2 + Chunked FFN`

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
   - Disable ComfyUI's lowvram/offloading mode
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

## 🙏 Credits

- **Implementation:** Claude (Anthropic) + RandomInternetPreson
- **Tensor Parallelism Concepts (V2):** Inspired by [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- **Testing & Development:** RandomInternetPreson ([@RandomInternetPreson](https://github.com/RandomInternetPreson))

## 📝 License

MIT License - Feel free to use, modify, and distribute!

## 🐛 Issues & Contributions

Found a bug or have an improvement? Please open an issue or PR!

This is experimental software - your feedback helps make it better for everyone.

---

*Making long-form AI video accessible to everyone* 🎥
