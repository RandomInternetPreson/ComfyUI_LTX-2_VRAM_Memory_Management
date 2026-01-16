"""
Tensor Parallel Attention for LTX-2 (v2)
========================================

This version properly parallelizes attention across GPUs using CUDA streams.

The key insight: Split Q, K, V by attention heads, run attention in parallel
on different GPUs, then gather results. Each GPU only needs 1/N of the 
attention memory!

Memory for attention scores: batch × heads × seq × seq × 4 bytes
For 500 frames (S ≈ 25,000):
  1 GPU (32 heads): 32 × 25000² × 4 = 80GB 
  4 GPUs (8 heads): 8 × 25000² × 4 = 20GB per GPU ✓
  7 GPUs (5 heads): 5 × 25000² × 4 = 12GB per GPU ✓✓

Credits:
- Tensor parallelism: Megatron-LM (NVIDIA)  
- Implementation: Claude (Anthropic) with RandomInternetPreson
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
import math
from einops import rearrange

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TensorParallelV2")


def apply_rotary_emb_heads(q_heads, k_heads, pe, head_indices, total_heads):
    """
    Apply rotary positional embeddings to a subset of heads.
    
    PE is computed for all heads, we need to select the right slice.
    """
    if pe is None:
        return q_heads, k_heads
    
    # PE shape depends on implementation - handle common cases
    # Usually: (batch, seq, heads, dim) or (batch, heads, seq, dim)
    
    cos_freqs, sin_freqs = pe if isinstance(pe, tuple) else (pe, pe)
    
    # Apply RoPE to Q
    q_rot = _apply_rope_to_subset(q_heads, cos_freqs, sin_freqs, head_indices, total_heads)
    k_rot = _apply_rope_to_subset(k_heads, cos_freqs, sin_freqs, head_indices, total_heads)
    
    return q_rot, k_rot


def _apply_rope_to_subset(x, cos, sin, head_indices, total_heads):
    """Apply rotary embeddings, handling head subsetting."""
    # x shape: (batch, num_heads_subset, seq, head_dim)
    # For now, assume PE applies uniformly or handle slicing
    
    # If PE is per-head, slice it
    if cos.dim() >= 3 and cos.shape[-2] == total_heads:
        # PE has head dimension - slice it
        cos = cos[..., head_indices, :]
        sin = sin[..., head_indices, :]
    
    # Standard RoPE application
    # Interleaved version (LTX-2 style)
    t_dup = rearrange(x, "... (d r) -> ... d r", r=2)
    t1, t2 = t_dup.unbind(dim=-1)
    t_dup_rot = torch.stack((-t2, t1), dim=-1)
    x_rot = rearrange(t_dup_rot, "... d r -> ... (d r)")
    
    # Broadcast cos/sin to match x shape if needed
    if cos.dim() < x.dim():
        cos = cos.unsqueeze(1)  # Add heads dim
        sin = sin.unsqueeze(1)
    
    return x * cos + x_rot * sin


class TensorParallelCrossAttention(nn.Module):
    """
    Drop-in replacement for CrossAttention that splits heads across GPUs.
    
    Runs attention computation in parallel using CUDA streams.
    """
    
    def __init__(
        self,
        original_attn: nn.Module,
        gpu_ids: List[int],
        primary_gpu: int = 0,
        verbose: int = 1,
    ):
        super().__init__()
        self.gpu_ids = gpu_ids
        self.num_gpus = len(gpu_ids)
        self.primary_gpu = primary_gpu
        self.primary_device = torch.device(f'cuda:{primary_gpu}')
        self.verbose = verbose
        
        # Copy original attention attributes
        self.heads = original_attn.heads
        self.dim_head = original_attn.dim_head
        self.attn_precision = getattr(original_attn, 'attn_precision', None)
        
        # Keep original layers on primary GPU (these are small)
        self.to_q = original_attn.to_q
        self.to_k = original_attn.to_k
        self.to_v = original_attn.to_v
        self.q_norm = original_attn.q_norm
        self.k_norm = original_attn.k_norm
        self.to_out = original_attn.to_out
        
        # Calculate head distribution
        self._distribute_heads()
        
        # Create CUDA streams for parallel execution
        self.streams = {
            gpu_id: torch.cuda.Stream(device=f'cuda:{gpu_id}')
            for gpu_id in gpu_ids
        }
        
        if verbose >= 1:
            logger.info(f"  TensorParallel: {self.heads} heads across {self.num_gpus} GPUs")
            for gpu_id, (start, end) in self.head_ranges.items():
                logger.info(f"    GPU {gpu_id}: heads {start}-{end-1} ({end-start} heads)")
    
    def _distribute_heads(self):
        """Calculate which heads go to which GPU."""
        base_heads = self.heads // self.num_gpus
        extra = self.heads % self.num_gpus
        
        self.head_ranges = {}
        self.head_counts = {}
        current = 0
        
        for i, gpu_id in enumerate(self.gpu_ids):
            count = base_heads + (1 if i < extra else 0)
            self.head_ranges[gpu_id] = (current, current + count)
            self.head_counts[gpu_id] = count
            current += count
    
    def forward(self, x, context=None, mask=None, pe=None, k_pe=None, transformer_options={}):
        """
        Tensor-parallel forward pass.
        
        Handles both:
        - Self-attention: Q, K, V all from x (same seq_len)
        - Cross-attention: Q from x, K/V from context (different seq_len!)
        """
        batch_size, seq_len_q, _ = x.shape
        
        # Step 1: Compute Q from x
        q = self.to_q(x)
        
        # K, V come from context (cross-attention) or x (self-attention)
        is_cross_attention = context is not None
        kv_input = context if is_cross_attention else x
        seq_len_kv = kv_input.shape[1]
        
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)
        
        # Step 2: Apply norms
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Step 3: Apply RoPE (only for self-attention typically)
        if pe is not None and not is_cross_attention:
            q = self._apply_rope(q, pe)
            k = self._apply_rope(k, pe if k_pe is None else k_pe)
        
        # Reshape for head splitting
        # Q: (batch, seq_q, heads*dim) -> (batch, seq_q, heads, dim)
        # K, V: (batch, seq_kv, heads*dim) -> (batch, seq_kv, heads, dim)
        q = q.view(batch_size, seq_len_q, self.heads, self.dim_head)
        k = k.view(batch_size, seq_len_kv, self.heads, self.dim_head)
        v = v.view(batch_size, seq_len_kv, self.heads, self.dim_head)
        
        # Step 4: Split by heads and distribute to GPUs
        q_splits = {}
        k_splits = {}
        v_splits = {}
        
        for gpu_id in self.gpu_ids:
            head_start, head_end = self.head_ranges[gpu_id]
            device = torch.device(f'cuda:{gpu_id}')
            
            q_splits[gpu_id] = q[:, :, head_start:head_end, :].to(device, non_blocking=True)
            k_splits[gpu_id] = k[:, :, head_start:head_end, :].to(device, non_blocking=True)
            v_splits[gpu_id] = v[:, :, head_start:head_end, :].to(device, non_blocking=True)
        
        # Free the full Q, K, V from primary GPU memory
        del q, k, v
        torch.cuda.empty_cache()
        
        # Step 5: Run attention in PARALLEL on all GPUs
        partial_outputs = {}
        
        for gpu_id in self.gpu_ids:
            stream = self.streams[gpu_id]
            device = torch.device(f'cuda:{gpu_id}')
            
            with torch.cuda.stream(stream):
                q_slice = q_splits[gpu_id]
                k_slice = k_splits[gpu_id]
                v_slice = v_splits[gpu_id]
                
                # Reshape for attention: (batch, seq, heads, dim) -> (batch, heads, seq, dim)
                q_slice = q_slice.transpose(1, 2).contiguous()
                k_slice = k_slice.transpose(1, 2).contiguous()
                v_slice = v_slice.transpose(1, 2).contiguous()
                
                # Scaled dot-product attention
                scale = 1.0 / math.sqrt(self.dim_head)
                
                # Use PyTorch's efficient attention if available
                if hasattr(F, 'scaled_dot_product_attention'):
                    attn_out = F.scaled_dot_product_attention(
                        q_slice, k_slice, v_slice,
                        attn_mask=mask.to(device) if mask is not None else None,
                        scale=scale,
                    )
                else:
                    # Manual attention
                    # Q: (batch, heads, seq_q, dim)
                    # K: (batch, heads, seq_kv, dim)
                    # Attention: (batch, heads, seq_q, seq_kv)
                    attn_scores = torch.matmul(q_slice, k_slice.transpose(-2, -1)) * scale
                    
                    if mask is not None:
                        mask_gpu = mask.to(device, non_blocking=True)
                        attn_scores = attn_scores.masked_fill(mask_gpu == 0, float('-inf'))
                    
                    attn_probs = F.softmax(attn_scores, dim=-1)
                    del attn_scores
                    
                    attn_out = torch.matmul(attn_probs, v_slice)
                    del attn_probs
                
                del q_slice, k_slice, v_slice
                
                # Output: (batch, heads, seq_q, dim) -> (batch, seq_q, heads, dim)
                attn_out = attn_out.transpose(1, 2).contiguous()
                
                partial_outputs[gpu_id] = attn_out
        
        del q_splits, k_splits, v_splits
        
        # Step 6: Synchronize and gather
        for stream in self.streams.values():
            stream.synchronize()
        
        gathered = []
        for gpu_id in self.gpu_ids:
            gathered.append(partial_outputs[gpu_id].to(self.primary_device, non_blocking=True))
        
        del partial_outputs
        torch.cuda.synchronize(self.primary_device)
        
        # Concatenate: (batch, seq_q, heads_subset, dim) -> (batch, seq_q, all_heads, dim)
        full_output = torch.cat(gathered, dim=2)
        del gathered
        
        # Reshape: (batch, seq_q, heads, dim) -> (batch, seq_q, heads*dim)
        full_output = full_output.view(batch_size, seq_len_q, -1)
        
        return self.to_out(full_output)
    
    def _apply_rope(self, x, pe):
        """Apply rotary positional embedding matching LTX-2's format."""
        if pe is None:
            return x
        
        # LTX-2 format: pe = (cos_freqs, sin_freqs, [optional split_flag])
        cos_freqs = pe[0]
        sin_freqs = pe[1]
        split_pe = pe[2] if len(pe) > 2 else False
        
        if split_pe:
            return self._apply_split_rope(x, cos_freqs, sin_freqs)
        else:
            return self._apply_interleaved_rope(x, cos_freqs, sin_freqs)
    
    def _apply_interleaved_rope(self, x, cos_freqs, sin_freqs):
        """Interleaved RoPE (LTX-2 default)."""
        t_dup = rearrange(x, "... (d r) -> ... d r", r=2)
        t1, t2 = t_dup.unbind(dim=-1)
        t_dup_rot = torch.stack((-t2, t1), dim=-1)
        x_rot = rearrange(t_dup_rot, "... d r -> ... (d r)")
        
        return x * cos_freqs + x_rot * sin_freqs
    
    def _apply_split_rope(self, x, cos, sin):
        """Split RoPE variant."""
        needs_reshape = False
        if x.ndim != 4 and cos.ndim == 4:
            B, H, T, _ = cos.shape
            x = x.reshape(B, T, H, -1).swapaxes(1, 2)
            needs_reshape = True
        
        split_x = rearrange(x, "... (d r) -> ... d r", d=2)
        first_half = split_x[..., :1, :]
        second_half = split_x[..., 1:, :]
        
        output = split_x * cos.unsqueeze(-2)
        first_out = output[..., :1, :]
        second_out = output[..., 1:, :]
        first_out.addcmul_(-sin.unsqueeze(-2), second_half)
        second_out.addcmul_(sin.unsqueeze(-2), first_half)
        
        output = rearrange(output, "... d r -> ... (d r)")
        return output.swapaxes(1, 2).reshape(B, T, -1) if needs_reshape else output


class ChunkedFFN(nn.Module):
    """
    Wraps a FeedForward module to run in chunked mode.
    
    Instead of processing all tokens at once (huge intermediate activation),
    processes in chunks to reduce peak memory.
    
    For 51000 tokens with 4x expansion:
      Full: (1, 51000, 16384) = 3.3GB intermediate
      Chunked (6 chunks): (1, 8500, 16384) = 0.55GB intermediate
    
    Uses in-place output to avoid creating multiple full-size tensors.
    """
    
    def __init__(
        self,
        original_ffn: nn.Module,
        num_chunks: int = 6,
        verbose: int = 1,
    ):
        super().__init__()
        self.original_ffn = original_ffn
        self.num_chunks = num_chunks
        self.verbose = verbose
    
    def forward(self, x):
        """Chunked FFN forward - reduces peak memory with in-place output."""
        batch_size, seq_len, hidden = x.shape
        
        # Calculate chunk size
        chunk_size = (seq_len + self.num_chunks - 1) // self.num_chunks
        
        # Pre-allocate output tensor (same size as input for FFN)
        output = torch.empty_like(x)
        
        # Process in chunks, writing directly to output
        for i in range(0, seq_len, chunk_size):
            end_idx = min(i + chunk_size, seq_len)
            chunk = x[:, i:end_idx, :]
            
            # Run FFN on this chunk and write directly to output
            out_chunk = self.original_ffn(chunk)
            output[:, i:end_idx, :] = out_chunk
            
            # Explicitly delete intermediate to free memory
            del out_chunk
        
        return output


class TensorParallelPipelineV2:
    """
    Manages tensor parallelism for a model's attention layers.
    Version 2 with proper CUDA stream parallelism.
    Now also parallelizes FFN layers!
    """
    
    _instances: Dict[int, 'TensorParallelPipelineV2'] = {}
    
    def __init__(
        self,
        model: nn.Module,
        num_gpus: int = None,
        primary_gpu: int = 0,
        parallelize_ffn: bool = True,
        ffn_chunks: int = 8,
        verbose: int = 1,
    ):
        self.model = model
        self.model_id = id(model)
        self.verbose = verbose
        self.primary_gpu = primary_gpu
        self.parallelize_ffn = parallelize_ffn
        self.ffn_chunks = ffn_chunks
        
        available = torch.cuda.device_count()
        self.num_gpus = min(num_gpus or available, available)
        self.gpu_ids = list(range(self.num_gpus))
        
        self.replaced_modules: Dict[str, nn.Module] = {}
        self.original_modules: Dict[str, nn.Module] = {}
        
        TensorParallelPipelineV2._instances[self.model_id] = self
    
    @classmethod
    def get_instance(cls, model: nn.Module) -> Optional['TensorParallelPipelineV2']:
        return cls._instances.get(id(model))
    
    def analyze_and_setup(self) -> Dict[str, Any]:
        """Find and replace CrossAttention and FFN modules."""
        info = {
            "num_gpus": self.num_gpus,
            "attention_modules_found": 0,
            "attention_modules_replaced": 0,
            "ffn_modules_found": 0,
            "ffn_modules_replaced": 0,
        }
        
        if self.verbose >= 1:
            logger.info(f"TensorParallelV2: Analyzing model...")
            logger.info(f"  Using {self.num_gpus} GPUs: {self.gpu_ids}")
        
        # Find CrossAttention modules
        attention_modules = {}
        for name, module in self.model.named_modules():
            # Check for CrossAttention-like modules
            if hasattr(module, 'to_q') and hasattr(module, 'to_k') and hasattr(module, 'to_v'):
                if hasattr(module, 'heads'):
                    attention_modules[name] = module
        
        info["attention_modules_found"] = len(attention_modules)
        
        if self.verbose >= 1:
            logger.info(f"  Found {len(attention_modules)} attention modules")
        
        # Replace with tensor-parallel versions
        for name, module in attention_modules.items():
            try:
                tp_attn = TensorParallelCrossAttention(
                    original_attn=module,
                    gpu_ids=self.gpu_ids,
                    primary_gpu=self.primary_gpu,
                    verbose=self.verbose,
                )
                
                self.original_modules[name] = module
                self.replaced_modules[name] = tp_attn
                self._replace_module(name, tp_attn)
                
                info["attention_modules_replaced"] += 1
                
            except Exception as e:
                if self.verbose >= 1:
                    logger.warning(f"  Could not parallelize attention {name}: {e}")
        
        # Find and replace FFN modules (if enabled)
        if self.parallelize_ffn:
            ffn_modules = {}
            for name, module in self.model.named_modules():
                # Look for FeedForward-like modules with net attribute
                if hasattr(module, 'net') and isinstance(module.net, nn.Sequential):
                    # Check if it's a FFN (ends with .ff or contains feed_forward)
                    if name.endswith('.ff') or 'ff' in name.split('.')[-1]:
                        ffn_modules[name] = module
            
            info["ffn_modules_found"] = len(ffn_modules)
            
            if self.verbose >= 1:
                logger.info(f"  Found {len(ffn_modules)} FFN modules")
            
            for name, module in ffn_modules.items():
                try:
                    # Use ChunkedFFN - doesn't require cloning, works with offloaded weights
                    chunked_ffn = ChunkedFFN(
                        original_ffn=module,
                        num_chunks=self.ffn_chunks,  # Configurable for more aggressive chunking
                        verbose=self.verbose,
                    )
                    
                    self.original_modules[name] = module
                    self.replaced_modules[name] = chunked_ffn
                    self._replace_module(name, chunked_ffn)
                    
                    info["ffn_modules_replaced"] += 1
                    
                except Exception as e:
                    if self.verbose >= 1:
                        logger.warning(f"  Could not wrap FFN {name}: {e}")
        
        if self.verbose >= 1:
            logger.info(f"TensorParallelV2: Setup complete!")
            logger.info(f"  Attention: {info['attention_modules_replaced']} (head-parallel across GPUs)")
            if self.parallelize_ffn:
                logger.info(f"  FFN: {info['ffn_modules_replaced']} (chunked into {self.ffn_chunks} pieces)")
            
            # Estimate memory savings
            if attention_modules:
                first_attn = list(attention_modules.values())[0]
                heads = getattr(first_attn, 'heads', 32)
                seq_estimate = 50000  # 700+ frames
                
                single_attn = heads * seq_estimate * seq_estimate * 4
                parallel_attn = (heads // self.num_gpus) * seq_estimate * seq_estimate * 4
                
                logger.info(f"  Memory estimate (S={seq_estimate}):")
                logger.info(f"    Attention: {single_attn / (1024**3):.1f}GB → {parallel_attn / (1024**3):.1f}GB per GPU")
                
                # FFN estimate: seq × hidden × 4 (expansion) × 4 bytes
                ffn_single = seq_estimate * 4096 * 4 * 4
                ffn_chunked = ffn_single // self.ffn_chunks
                logger.info(f"    FFN peak: {ffn_single / (1024**3):.1f}GB → {ffn_chunked / (1024**3):.1f}GB ({self.ffn_chunks} chunks)")
        
        return info
    
    def _replace_module(self, path: str, new_module: nn.Module):
        """Replace a module in the model hierarchy."""
        parts = path.split('.')
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)
    
    def cleanup(self):
        """Restore original modules."""
        for path, original in self.original_modules.items():
            self._replace_module(path, original)
        
        self.replaced_modules.clear()
        self.original_modules.clear()
        TensorParallelPipelineV2._instances.pop(self.model_id, None)
        
        if self.verbose >= 1:
            logger.info("TensorParallelV2: Cleanup complete")


# Node definitions for ComfyUI
class TensorParallelAttentionV2:
    """
    EXPERIMENTAL: True parallel computation across GPUs using CUDA streams.
    
    Parallelizes both:
    - Attention: Split by heads (each GPU computes subset of heads)
    - FFN: Chunked execution (reduces peak memory)
    
    For very long videos (1000+ frames), increase ffn_chunks to reduce
    peak FFN memory even further.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        num_gpus = torch.cuda.device_count()
        return {
            "required": {
                "model": ("MODEL",),
                "num_gpus": ("INT", {"default": min(num_gpus, 7), "min": 2, "max": num_gpus}),
                "primary_gpu": ("INT", {"default": 0, "min": 0, "max": num_gpus - 1}),
                "parallelize_ffn": ("BOOLEAN", {"default": True}),
                "ffn_chunks": ("INT", {
                    "default": 8,
                    "min": 4,
                    "max": 32,
                    "step": 2,
                    "tooltip": "More chunks = less FFN memory but slower. Try 12-16 for 1000+ frames."
                }),
                "verbose": ("INT", {"default": 1, "min": 0, "max": 2}),
            },
        }
    
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "setup"
    CATEGORY = "multigpu/experimental"
    
    def setup(self, model, num_gpus, primary_gpu, parallelize_ffn, ffn_chunks, verbose):
        lines = ["Tensor Parallel V2 + Chunked FFN"]
        lines.append("=" * 50)
        
        try:
            # Get diffusion model
            target = model
            if hasattr(model, 'model'):
                target = model.model
                if hasattr(target, 'diffusion_model'):
                    target = target.diffusion_model
            
            # Check existing
            existing = TensorParallelPipelineV2.get_instance(target)
            if existing:
                lines.append("Already active!")
                return (model, "\n".join(lines))
            
            # Setup
            pipeline = TensorParallelPipelineV2(
                model=target,
                num_gpus=num_gpus,
                primary_gpu=primary_gpu,
                parallelize_ffn=parallelize_ffn,
                ffn_chunks=ffn_chunks,
                verbose=verbose,
            )
            
            info = pipeline.analyze_and_setup()
            
            lines.append(f"GPUs: {num_gpus}")
            lines.append(f"Attention modules: {info['attention_modules_replaced']} (head-parallel)")
            if parallelize_ffn:
                lines.append(f"FFN modules: {info.get('ffn_modules_replaced', 0)} (chunked into {ffn_chunks} pieces)")
            lines.append("")
            lines.append("Memory distribution:")
            lines.append(f"  - Attention: split across {num_gpus} GPUs by heads")
            if parallelize_ffn:
                lines.append(f"  - FFN: {ffn_chunks} chunks (more = less peak memory)")
            
        except Exception as e:
            lines.append(f"ERROR: {e}")
            import traceback
            lines.append(traceback.format_exc())
        
        return (model, "\n".join(lines))


NODE_CLASS_MAPPINGS = {
    "TensorParallelAttentionV2": TensorParallelAttentionV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorParallelAttentionV2": "Tensor Parallel V2 + Chunked FFN",
}
