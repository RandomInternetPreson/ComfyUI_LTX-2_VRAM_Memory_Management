"""
Sequence Parallelism + Ring Attention for LTX-2 (V5)
=====================================================

This implements sequence parallelism across multiple GPUs, where the
sequence dimension (tokens) is split across GPUs rather than heads.

Key insight: For 600K+ tokens, no single GPU can hold full Q, K, V.
Solution: Each GPU holds 1/N of the sequence and uses ring attention
to compute full self-attention without any GPU holding the full tensors.

Ring Attention Algorithm:
  1. Split sequence across N GPUs: GPU i holds tokens [i*chunk : (i+1)*chunk]
  2. Each GPU computes Q_local from its sequence chunk
  3. Ring iteration (N steps):
     - Each GPU holds K_chunk, V_chunk (starts with its own)
     - Compute partial attention: softmax(Q_local @ K_chunk.T) @ V_chunk
     - Use online softmax to accumulate across chunks
     - Pass K_chunk, V_chunk to next GPU in ring
  4. After N iterations, each GPU has full attention output for its tokens

Memory per GPU:
  - vx_local: 1/N of full sequence (~3GB for 7 GPUs)
  - Q_local: 1/N (~3GB)
  - K_chunk, V_chunk: 1/N each (~6GB)
  - Attention intermediates: ~3GB
  - Total: ~15GB fits in 24GB!

Credits:
- Ring Attention paper: https://arxiv.org/abs/2310.01889
- Implementation: Claude (Anthropic) with RandomInternetPreson
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda import Stream
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
import math
import gc
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SeqParallelV5")


def get_gpu_memory_info(device_id: int) -> str:
    """Get formatted GPU memory info."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
        reserved = torch.cuda.memory_reserved(device_id) / (1024**3)
        total = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
        return f"GPU {device_id}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {total:.1f}GB total"
    return "CUDA not available"


# =============================================================================
# Ring Attention Implementation
# =============================================================================

class RingAttention(nn.Module):
    """
    Ring Attention for sequence-parallel self-attention.
    
    Each GPU holds 1/N of the sequence. To compute full self-attention,
    we rotate K, V chunks around the ring while each GPU computes partial
    attention with its local Q against the visiting K, V chunks.
    
    Uses online softmax (log-sum-exp trick) to accumulate attention
    across chunks without materializing the full attention matrix.
    """
    
    def __init__(
        self,
        gpu_ids: List[int],
        dim_head: int,
        num_heads: int,
        verbose: int = 1,
    ):
        super().__init__()
        self.gpu_ids = gpu_ids
        self.num_gpus = len(gpu_ids)
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.verbose = verbose
        self.scale = 1.0 / math.sqrt(dim_head)
        
        # Create CUDA streams for async communication
        self.compute_streams = {
            gpu_id: Stream(device=f'cuda:{gpu_id}')
            for gpu_id in gpu_ids
        }
        self.comm_streams = {
            gpu_id: Stream(device=f'cuda:{gpu_id}')
            for gpu_id in gpu_ids
        }
        
        self._logged = False
    
    def forward(
        self,
        q_chunks: Dict[int, torch.Tensor],  # {gpu_id: Q_local}
        k_chunks: Dict[int, torch.Tensor],  # {gpu_id: K_local}
        v_chunks: Dict[int, torch.Tensor],  # {gpu_id: V_local}
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Compute ring attention across all GPUs.
        
        Args:
            q_chunks: Dict mapping gpu_id to Q tensor (batch, heads, seq_local, dim)
            k_chunks: Dict mapping gpu_id to K tensor (batch, heads, seq_local, dim)
            v_chunks: Dict mapping gpu_id to V tensor (batch, heads, seq_local, dim)
            mask: Optional attention mask
            
        Returns:
            Dict mapping gpu_id to attention output (batch, heads, seq_local, dim)
        """
        if not self._logged and self.verbose >= 1:
            sample_q = list(q_chunks.values())[0]
            logger.info(f"RingAttention: Q chunk shape {sample_q.shape}, {self.num_gpus} GPUs")
            self._logged = True
        
        # Initialize output accumulators and log-sum-exp for online softmax
        outputs = {}
        lse = {}  # log-sum-exp for numerical stability
        
        for gpu_id in self.gpu_ids:
            device = torch.device(f'cuda:{gpu_id}')
            q_local = q_chunks[gpu_id]
            batch, heads, seq_local, dim = q_local.shape
            
            # Initialize output accumulator
            outputs[gpu_id] = torch.zeros_like(q_local)
            # Initialize log-sum-exp (for online softmax normalization)
            lse[gpu_id] = torch.full(
                (batch, heads, seq_local, 1),
                float('-inf'),
                device=device,
                dtype=q_local.dtype
            )
        
        # Current K, V chunks on each GPU (will rotate)
        current_k = {gpu_id: k_chunks[gpu_id].clone() for gpu_id in self.gpu_ids}
        current_v = {gpu_id: v_chunks[gpu_id].clone() for gpu_id in self.gpu_ids}
        
        # Ring iteration: each GPU processes all K, V chunks
        for ring_step in range(self.num_gpus):
            # Compute partial attention on each GPU
            for gpu_id in self.gpu_ids:
                with torch.cuda.device(gpu_id):
                    stream = self.compute_streams[gpu_id]
                    with torch.cuda.stream(stream):
                        q_local = q_chunks[gpu_id]
                        k_chunk = current_k[gpu_id]
                        v_chunk = current_v[gpu_id]
                        
                        # Compute attention scores: (batch, heads, seq_q, seq_k)
                        attn_scores = torch.matmul(q_local, k_chunk.transpose(-2, -1)) * self.scale
                        
                        # Online softmax update (log-sum-exp trick)
                        # This allows accumulating softmax across chunks
                        chunk_max = attn_scores.max(dim=-1, keepdim=True).values
                        chunk_exp = torch.exp(attn_scores - chunk_max)
                        chunk_sum = chunk_exp.sum(dim=-1, keepdim=True)
                        chunk_lse = chunk_max + torch.log(chunk_sum)
                        
                        # Compute weighted values for this chunk
                        chunk_attn = chunk_exp / chunk_sum  # Local softmax
                        chunk_out = torch.matmul(chunk_attn, v_chunk)
                        
                        # Update running output with online softmax correction
                        old_lse = lse[gpu_id]
                        new_lse = torch.logaddexp(old_lse, chunk_lse)
                        
                        # Correction factors for accumulated output and new chunk
                        old_scale = torch.exp(old_lse - new_lse)
                        new_scale = torch.exp(chunk_lse - new_lse)
                        
                        outputs[gpu_id] = outputs[gpu_id] * old_scale + chunk_out * new_scale
                        lse[gpu_id] = new_lse
                        
                        del attn_scores, chunk_exp, chunk_attn, chunk_out
            
            # Synchronize compute before communication
            for gpu_id in self.gpu_ids:
                self.compute_streams[gpu_id].synchronize()
            
            # Rotate K, V to next GPU in ring (skip on last iteration)
            if ring_step < self.num_gpus - 1:
                next_k = {}
                next_v = {}
                
                for i, gpu_id in enumerate(self.gpu_ids):
                    # Send to next GPU, receive from previous
                    next_gpu = self.gpu_ids[(i + 1) % self.num_gpus]
                    prev_gpu = self.gpu_ids[(i - 1) % self.num_gpus]
                    
                    with torch.cuda.device(gpu_id):
                        stream = self.comm_streams[gpu_id]
                        with torch.cuda.stream(stream):
                            # Receive K, V from previous GPU
                            next_k[gpu_id] = current_k[prev_gpu].to(f'cuda:{gpu_id}', non_blocking=True)
                            next_v[gpu_id] = current_v[prev_gpu].to(f'cuda:{gpu_id}', non_blocking=True)
                
                # Synchronize communication
                for gpu_id in self.gpu_ids:
                    self.comm_streams[gpu_id].synchronize()
                
                # Update current K, V
                current_k = next_k
                current_v = next_v
        
        # Cleanup
        del current_k, current_v, lse
        
        return outputs


# =============================================================================
# Sequence Parallel Attention Wrapper
# =============================================================================

class SeqParallelAttention(nn.Module):
    """
    Wraps an attention module to use sequence parallelism.
    
    Input sequence is already split across GPUs. This module:
    1. Computes Q, K, V projections locally on each GPU
    2. Uses ring attention for self-attention
    3. For cross-attention, broadcasts the (small) context to all GPUs
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
        self.verbose = verbose
        
        # Copy attention parameters
        self.heads = original_attn.heads
        self.dim_head = original_attn.dim_head
        self.inner_dim = self.heads * self.dim_head
        
        # Keep original projections (will be replicated to each GPU as needed)
        self.to_q = original_attn.to_q
        self.to_k = original_attn.to_k
        self.to_v = original_attn.to_v
        self.to_out = original_attn.to_out
        
        # Norms
        self.q_norm = original_attn.q_norm if hasattr(original_attn, 'q_norm') else None
        self.k_norm = original_attn.k_norm if hasattr(original_attn, 'k_norm') else None
        
        # Ring attention module
        self.ring_attention = RingAttention(
            gpu_ids=gpu_ids,
            dim_head=self.dim_head,
            num_heads=self.heads,
            verbose=verbose,
        )
        
        # Track which GPUs have weight copies
        self._weights_on_gpu = set()
        self._gpu_modules = {}
        
        self._logged = False
    
    def _ensure_weights_on_gpu(self, gpu_id: int):
        """Lazily copy weights to GPU when first needed."""
        if gpu_id in self._weights_on_gpu:
            return
        
        device = torch.device(f'cuda:{gpu_id}')
        
        # Copy projection weights to this GPU
        self._gpu_modules[gpu_id] = {
            'to_q': self._copy_linear_to_device(self.to_q, device),
            'to_k': self._copy_linear_to_device(self.to_k, device),
            'to_v': self._copy_linear_to_device(self.to_v, device),
            'to_out': self._copy_sequential_to_device(self.to_out, device) if isinstance(self.to_out, nn.Sequential) else self._copy_linear_to_device(self.to_out, device),
            'q_norm': self._copy_norm_to_device(self.q_norm, device) if self.q_norm else None,
            'k_norm': self._copy_norm_to_device(self.k_norm, device) if self.k_norm else None,
        }
        
        self._weights_on_gpu.add(gpu_id)
    
    def _copy_linear_to_device(self, linear: nn.Linear, device: torch.device) -> nn.Linear:
        """Copy a linear layer to a device."""
        new_linear = nn.Linear(
            linear.in_features, linear.out_features,
            bias=linear.bias is not None,
            device=device,
            dtype=linear.weight.dtype
        )
        with torch.no_grad():
            new_linear.weight.copy_(linear.weight)
            if linear.bias is not None:
                new_linear.bias.copy_(linear.bias)
        return new_linear
    
    def _copy_sequential_to_device(self, seq: nn.Sequential, device: torch.device) -> nn.Sequential:
        """Copy a sequential module to a device."""
        new_modules = []
        for module in seq:
            if isinstance(module, nn.Linear):
                new_modules.append(self._copy_linear_to_device(module, device))
            elif isinstance(module, nn.Dropout):
                new_modules.append(nn.Dropout(module.p))
            else:
                new_modules.append(module)
        return nn.Sequential(*new_modules).to(device)
    
    def _copy_norm_to_device(self, norm: nn.Module, device: torch.device) -> nn.Module:
        """Copy a norm layer to a device."""
        # Handle RMSNorm or LayerNorm
        if hasattr(norm, 'weight'):
            new_norm = type(norm)(norm.weight.shape[0], eps=getattr(norm, 'eps', 1e-6))
            new_norm = new_norm.to(device)
            with torch.no_grad():
                if norm.weight is not None:
                    new_norm.weight.copy_(norm.weight)
            return new_norm
        return norm
    
    def forward(
        self,
        x_chunks: Dict[int, torch.Tensor],  # {gpu_id: x_local}
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        pe: Optional[Tuple] = None,
        **kwargs
    ) -> Dict[int, torch.Tensor]:
        """
        Sequence-parallel attention forward.
        
        Args:
            x_chunks: Dict mapping gpu_id to local sequence chunk
            context: Optional context for cross-attention (broadcast to all GPUs)
            mask: Optional attention mask
            pe: Optional positional embeddings
            
        Returns:
            Dict mapping gpu_id to output chunks
        """
        is_cross_attention = context is not None
        
        if not self._logged and self.verbose >= 1:
            sample_x = list(x_chunks.values())[0]
            logger.info(f"SeqParallelAttention: x chunk {sample_x.shape}, cross_attn={is_cross_attention}")
            self._logged = True
        
        # Compute Q, K, V on each GPU
        q_chunks = {}
        k_chunks = {}
        v_chunks = {}
        
        for gpu_id in self.gpu_ids:
            self._ensure_weights_on_gpu(gpu_id)
            device = torch.device(f'cuda:{gpu_id}')
            modules = self._gpu_modules[gpu_id]
            
            with torch.cuda.device(gpu_id):
                x_local = x_chunks[gpu_id]
                batch, seq_local, hidden = x_local.shape
                
                # Q projection
                q = modules['to_q'](x_local)
                if modules['q_norm']:
                    q = modules['q_norm'](q)
                
                # K, V projection (from context if cross-attention)
                if is_cross_attention:
                    # Broadcast context to this GPU
                    ctx = context.to(device)
                    k = modules['to_k'](ctx)
                    v = modules['to_v'](ctx)
                    if modules['k_norm']:
                        k = modules['k_norm'](k)
                else:
                    k = modules['to_k'](x_local)
                    v = modules['to_v'](x_local)
                    if modules['k_norm']:
                        k = modules['k_norm'](k)
                
                # Apply RoPE if provided (for self-attention)
                if pe is not None and not is_cross_attention:
                    q, k = self._apply_rope(q, k, pe, gpu_id, seq_local)
                
                # Reshape for attention: (batch, seq, heads*dim) -> (batch, heads, seq, dim)
                q = q.view(batch, -1, self.heads, self.dim_head).transpose(1, 2)
                k = k.view(batch, -1, self.heads, self.dim_head).transpose(1, 2)
                v = v.view(batch, -1, self.heads, self.dim_head).transpose(1, 2)
                
                q_chunks[gpu_id] = q
                k_chunks[gpu_id] = k
                v_chunks[gpu_id] = v
        
        # Compute attention
        if is_cross_attention:
            # Cross-attention: each GPU has full K, V from context
            # Just compute local attention (no ring needed)
            out_chunks = {}
            for gpu_id in self.gpu_ids:
                with torch.cuda.device(gpu_id):
                    q = q_chunks[gpu_id]
                    k = k_chunks[gpu_id]
                    v = v_chunks[gpu_id]
                    
                    # Standard attention
                    scale = 1.0 / math.sqrt(self.dim_head)
                    attn = F.scaled_dot_product_attention(q, k, v, scale=scale)
                    out_chunks[gpu_id] = attn
        else:
            # Self-attention: use ring attention
            out_chunks = self.ring_attention(q_chunks, k_chunks, v_chunks, mask)
        
        # Cleanup Q, K, V
        del q_chunks, k_chunks, v_chunks
        
        # Output projection on each GPU
        output_chunks = {}
        for gpu_id in self.gpu_ids:
            with torch.cuda.device(gpu_id):
                modules = self._gpu_modules[gpu_id]
                attn_out = out_chunks[gpu_id]
                batch, heads, seq_local, dim = attn_out.shape
                
                # Reshape: (batch, heads, seq, dim) -> (batch, seq, heads*dim)
                attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_local, -1)
                
                # Output projection
                if isinstance(modules['to_out'], nn.Sequential):
                    output_chunks[gpu_id] = modules['to_out'](attn_out)
                else:
                    output_chunks[gpu_id] = modules['to_out'](attn_out)
        
        del out_chunks
        return output_chunks
    
    def _apply_rope(self, q, k, pe, gpu_id, seq_local):
        """Apply rotary position embeddings to Q and K.
        
        Matches the original model.py implementation:
        - pe[2] = True -> split RoPE
        - pe[2] = False (default) -> interleaved RoPE
        """
        cos = pe[0]
        sin = pe[1]
        split_pe = pe[2] if len(pe) > 2 else False
        
        device = torch.device(f'cuda:{gpu_id}')
        
        # Get this GPU's portion of the sequence
        gpu_idx = self.gpu_ids.index(gpu_id)
        total_seq = cos.shape[2] if cos.ndim == 4 else (cos.shape[1] if cos.ndim >= 2 else cos.shape[0])
        chunk_size = total_seq // self.num_gpus
        start = gpu_idx * chunk_size
        end = start + seq_local
        
        # Slice PE for this chunk
        if cos.ndim == 4:  # (batch, heads, seq, dim)
            cos_local = cos[:, :, start:end, :].to(device)
            sin_local = sin[:, :, start:end, :].to(device)
        elif cos.ndim == 3:  # (batch, seq, dim)
            cos_local = cos[:, start:end, :].to(device)
            sin_local = sin[:, start:end, :].to(device)
        else:
            cos_local = cos.to(device)
            sin_local = sin.to(device)
        
        cos_dim = cos_local.shape[-1]
        dim_head = q.shape[-1]
        
        if split_pe:
            # SPLIT RoPE - exactly matching model.py apply_split_rotary_emb
            q_split = q.view(*q.shape[:-1], 2, dim_head // 2)
            k_split = k.view(*k.shape[:-1], 2, dim_head // 2)
            
            q_first = q_split[..., 0, :]
            q_second = q_split[..., 1, :]
            k_first = k_split[..., 0, :]
            k_second = k_split[..., 1, :]
            
            q_first_out = q_first * cos_local - sin_local * q_second
            q_second_out = q_second * cos_local + sin_local * q_first
            k_first_out = k_first * cos_local - sin_local * k_second
            k_second_out = k_second * cos_local + sin_local * k_first
            
            q_out = torch.stack([q_first_out, q_second_out], dim=-2)
            k_out = torch.stack([k_first_out, k_second_out], dim=-2)
            q = q_out.view(*q.shape[:-1], dim_head)
            k = k_out.view(*k.shape[:-1], dim_head)
        else:
            # INTERLEAVED RoPE - pairs of consecutive elements
            q_reshape = q.view(*q.shape[:-1], -1, 2)
            k_reshape = k.view(*k.shape[:-1], -1, 2)
            
            q1, q2 = q_reshape[..., 0], q_reshape[..., 1]
            k1, k2 = k_reshape[..., 0], k_reshape[..., 1]
            
            # Rotated: (-t2, t1)
            q_rot = torch.stack([-q2, q1], dim=-1).view_as(q)
            k_rot = torch.stack([-k2, k1], dim=-1).view_as(k)
            
            # Expand cos/sin if needed
            if cos_dim == dim_head // 2:
                cos_local = cos_local.unsqueeze(-1).expand(*cos_local.shape, 2).reshape(*cos_local.shape[:-1], -1)
                sin_local = sin_local.unsqueeze(-1).expand(*sin_local.shape, 2).reshape(*sin_local.shape[:-1], -1)
            
            q = q * cos_local + q_rot * sin_local
            k = k * cos_local + k_rot * sin_local
        
        return q, k


# =============================================================================
# Sequence Parallel FFN
# =============================================================================

class SeqParallelFFN(nn.Module):
    """
    FFN that processes sequence chunks independently on each GPU.
    
    Since FFN is applied independently to each token, we can simply
    run it on each GPU's local sequence chunk with no communication.
    """
    
    def __init__(
        self,
        original_ffn: nn.Module,
        gpu_ids: List[int],
        num_chunks: int = 4,  # Sub-chunking within each GPU for memory
        verbose: int = 1,
    ):
        super().__init__()
        self.gpu_ids = gpu_ids
        self.num_gpus = len(gpu_ids)
        self.num_chunks = num_chunks
        self.verbose = verbose
        
        # Keep original FFN
        self.ffn = original_ffn
        
        # Weight copies per GPU
        self._weights_on_gpu = set()
        self._gpu_ffn = {}
        
        self._logged = False
    
    def _ensure_weights_on_gpu(self, gpu_id: int):
        """Lazily copy FFN weights to GPU."""
        if gpu_id in self._weights_on_gpu:
            return
        
        device = torch.device(f'cuda:{gpu_id}')
        
        # Deep copy the FFN to this device
        self._gpu_ffn[gpu_id] = self._copy_ffn_to_device(self.ffn, device)
        self._weights_on_gpu.add(gpu_id)
    
    def _copy_ffn_to_device(self, ffn: nn.Module, device: torch.device) -> nn.Module:
        """Copy FFN module to device."""
        import copy
        new_ffn = copy.deepcopy(ffn)
        new_ffn = new_ffn.to(device)
        return new_ffn
    
    def forward(self, x_chunks: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        Process FFN on each GPU's sequence chunk.
        
        Args:
            x_chunks: Dict mapping gpu_id to local sequence chunk
            
        Returns:
            Dict mapping gpu_id to FFN output chunks
        """
        if not self._logged and self.verbose >= 1:
            sample_x = list(x_chunks.values())[0]
            logger.info(f"SeqParallelFFN: x chunk {sample_x.shape}")
            self._logged = True
        
        output_chunks = {}
        
        for gpu_id in self.gpu_ids:
            self._ensure_weights_on_gpu(gpu_id)
            
            with torch.cuda.device(gpu_id):
                x_local = x_chunks[gpu_id]
                ffn = self._gpu_ffn[gpu_id]
                
                batch, seq_local, hidden = x_local.shape
                
                # Sub-chunk for memory efficiency
                if seq_local > 1000 and self.num_chunks > 1:
                    chunk_size = (seq_local + self.num_chunks - 1) // self.num_chunks
                    outputs = []
                    
                    for i in range(0, seq_local, chunk_size):
                        end = min(i + chunk_size, seq_local)
                        chunk = x_local[:, i:end, :]
                        out = ffn(chunk)
                        outputs.append(out)
                    
                    output_chunks[gpu_id] = torch.cat(outputs, dim=1)
                else:
                    output_chunks[gpu_id] = ffn(x_local)
        
        return output_chunks


# =============================================================================
# Sequence Parallel Block Wrapper
# =============================================================================

class SeqParallelBlockWrapper(nn.Module):
    """
    Wraps a transformer block to use sequence parallelism.
    
    The sequence is split across GPUs at block entry, processed in parallel,
    and remains split until the final block where it's gathered.
    """
    
    def __init__(
        self,
        original_block: nn.Module,
        gpu_ids: List[int],
        block_idx: int,
        num_blocks: int,
        is_av_block: bool = False,
        verbose: int = 1,
        exclude_primary: bool = True,  # Exclude GPU 0 from chunks - it holds VAE/weights
    ):
        super().__init__()
        self.original_block = original_block
        self.all_gpu_ids = gpu_ids
        self.block_idx = block_idx
        self.num_blocks = num_blocks
        self.is_av_block = is_av_block
        self.is_last_block = (block_idx == num_blocks - 1)
        self.verbose = verbose
        
        # Exclude primary GPU from chunk distribution if it's overloaded with VAE/weights
        self.exclude_primary = exclude_primary
        if exclude_primary and len(gpu_ids) > 2:
            self.primary_gpu = gpu_ids[0]  # This GPU holds VAE - avoid using it
            self.gpu_ids = gpu_ids[1:]  # For sequence chunks AND compute (distributed)
            if verbose >= 1 and block_idx == 0:
                logger.info(f"  GPU {self.primary_gpu} excluded (holds VAE), GPUs {self.gpu_ids} for distributed chunks + compute")
        else:
            self.primary_gpu = gpu_ids[0]
            self.gpu_ids = gpu_ids
        
        self.num_gpus = len(self.gpu_ids)
        
        # Create sequence-parallel wrappers for attention
        if hasattr(original_block, 'attn1'):
            self.sp_attn1 = SeqParallelAttention(
                original_block.attn1, self.gpu_ids, verbose=verbose
            )
        
        if hasattr(original_block, 'attn2'):
            self.sp_attn2 = SeqParallelAttention(
                original_block.attn2, self.gpu_ids, verbose=verbose
            )
        
        # FFN wrapper
        if hasattr(original_block, 'ff'):
            self.sp_ff = SeqParallelFFN(original_block.ff, self.gpu_ids, verbose=verbose)
        
        # For LTXAV blocks (audio-video)
        if is_av_block:
            if hasattr(original_block, 'audio_attn1'):
                self.sp_audio_attn1 = SeqParallelAttention(
                    original_block.audio_attn1, gpu_ids, verbose=verbose
                )
            if hasattr(original_block, 'audio_attn2'):
                self.sp_audio_attn2 = SeqParallelAttention(
                    original_block.audio_attn2, gpu_ids, verbose=verbose
                )
            if hasattr(original_block, 'audio_ff'):
                self.sp_audio_ff = SeqParallelFFN(
                    original_block.audio_ff, gpu_ids, verbose=verbose
                )
            # Cross-modal attention
            if hasattr(original_block, 'audio_to_video_attn'):
                self.sp_a2v_attn = SeqParallelAttention(
                    original_block.audio_to_video_attn, gpu_ids, verbose=verbose
                )
            if hasattr(original_block, 'video_to_audio_attn'):
                self.sp_v2a_attn = SeqParallelAttention(
                    original_block.video_to_audio_attn, gpu_ids, verbose=verbose
                )
        
        # Store scale/shift tables for ada operations
        self._setup_ada_params()
        
        self._logged = False
    
    def _setup_ada_params(self):
        """Copy adaptive parameters to each GPU."""
        self._gpu_ada = {}
        
        for gpu_id in self.gpu_ids:
            device = torch.device(f'cuda:{gpu_id}')
            ada_params = {}
            
            # Copy scale_shift_table if exists
            if hasattr(self.original_block, 'scale_shift_table'):
                ada_params['scale_shift_table'] = self.original_block.scale_shift_table.to(device)
            
            # For LTXAV blocks
            if hasattr(self.original_block, 'audio_scale_shift_table'):
                ada_params['audio_scale_shift_table'] = self.original_block.audio_scale_shift_table.to(device)
            
            self._gpu_ada[gpu_id] = ada_params
    
    def forward(self, x, **kwargs):
        """
        Sequence-parallel forward pass.
        
        x can be:
        - A single tensor (LTXVModel)
        - A tuple (vx, ax) for LTXAV model
        """
        # Only log for block 0 to avoid spam
        if not self._logged and self.verbose >= 2 and self.block_idx == 0:
            if isinstance(x, tuple):
                logger.info(f"SeqParallelBlock {self.block_idx}: vx={x[0].shape}, ax={x[1].shape}")
            else:
                logger.info(f"SeqParallelBlock {self.block_idx}: x={x.shape}")
            self._logged = True
        
        # For now, fall back to original block with GPU distribution
        # Full implementation would split x into chunks across GPUs
        # This is a simplified version that at least distributes computation
        
        return self._distributed_forward(x, **kwargs)
    
    def _distributed_forward(self, x, **kwargs):
        """
        Distribute sequence across GPUs and process.
        """
        if isinstance(x, tuple):
            vx, ax = x
            return self._av_forward(vx, ax, **kwargs)
        else:
            return self._single_stream_forward(x, **kwargs)
    
    def _split_tensor(self, tensor: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Split tensor's sequence dimension across GPUs."""
        batch, seq, hidden = tensor.shape
        chunk_size = (seq + self.num_gpus - 1) // self.num_gpus
        
        chunks = {}
        for i, gpu_id in enumerate(self.gpu_ids):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, seq)
            if start < seq:
                chunk = tensor[:, start:end, :].to(f'cuda:{gpu_id}', non_blocking=True)
                chunks[gpu_id] = chunk
            else:
                # Empty chunk for this GPU
                chunks[gpu_id] = torch.empty(batch, 0, hidden, device=f'cuda:{gpu_id}')
        
        return chunks
    
    def _gather_chunks(self, chunks: Dict[int, torch.Tensor], target_device: torch.device) -> torch.Tensor:
        """Gather chunks from all GPUs back to target device."""
        ordered_chunks = []
        for gpu_id in self.gpu_ids:
            if chunks[gpu_id].shape[1] > 0:  # Skip empty chunks
                ordered_chunks.append(chunks[gpu_id].to(target_device, non_blocking=True))
        
        torch.cuda.synchronize(target_device)
        return torch.cat(ordered_chunks, dim=1)
    
    def _single_stream_forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Process single stream with proper ring attention.
        
        Key insight: We can't just run chunks through the original block
        because self-attention needs to see ALL tokens. Instead, we:
        1. Split x across GPUs
        2. Use ring attention for self-attention (each GPU's Q attends to all K,V)
        3. Run FFN independently per chunk
        4. Gather results
        """
        original_device = x.device
        batch, seq, hidden = x.shape
        
        # Get kwargs
        context = kwargs.get('context')
        v_context = kwargs.get('v_context', context)
        timestep = kwargs.get('timestep')
        v_timestep = kwargs.get('v_timestep', timestep)
        pe = kwargs.get('pe')
        v_pe = kwargs.get('v_pe', pe)
        transformer_options = kwargs.get('transformer_options', {})
        attention_mask = kwargs.get('attention_mask')
        
        # Split sequence across GPUs
        chunk_size = (seq + self.num_gpus - 1) // self.num_gpus
        x_chunks = {}
        chunk_ranges = {}
        
        for i, gpu_id in enumerate(self.gpu_ids):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, seq)
            if start < seq:
                x_chunks[gpu_id] = x[:, start:end, :].to(f'cuda:{gpu_id}', non_blocking=True)
                chunk_ranges[gpu_id] = (start, end)
            else:
                x_chunks[gpu_id] = torch.empty(batch, 0, hidden, device=f'cuda:{gpu_id}')
                chunk_ranges[gpu_id] = (start, start)
        
        # Synchronize transfers
        for gpu_id in self.gpu_ids:
            torch.cuda.synchronize(f'cuda:{gpu_id}')
        
        # Get ada values (scale, shift, gate) - need these for the block
        ada_values = {}
        if hasattr(self.original_block, 'scale_shift_table') and v_timestep is not None:
            for gpu_id in self.gpu_ids:
                with torch.cuda.device(gpu_id):
                    local_chunk = x_chunks[gpu_id]
                    if local_chunk.shape[1] == 0:
                        continue
                    ts = v_timestep.to(f'cuda:{gpu_id}')
                    # Call original block's ada function
                    if hasattr(self.original_block, 'get_ada_values'):
                        scale_shift = self._gpu_ada[gpu_id].get('scale_shift_table')
                        if scale_shift is not None:
                            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                                self.original_block.get_ada_values(scale_shift, batch, ts, slice(None))
                            )
                            ada_values[gpu_id] = {
                                'shift_msa': shift_msa, 'scale_msa': scale_msa, 'gate_msa': gate_msa,
                                'shift_mlp': shift_mlp, 'scale_mlp': scale_mlp, 'gate_mlp': gate_mlp,
                            }
        
        # ========== SELF-ATTENTION (Ring Attention) ==========
        # Step 1: Compute local Q, K, V on each GPU
        q_chunks = {}
        k_chunks = {}
        v_chunks = {}
        
        for gpu_id in self.gpu_ids:
            local_x = x_chunks[gpu_id]
            if local_x.shape[1] == 0:
                continue
                
            with torch.cuda.device(gpu_id):
                # Apply pre-norm and ada scaling
                if gpu_id in ada_values:
                    ada = ada_values[gpu_id]
                    norm_x = self._rms_norm(local_x)
                    norm_x = norm_x * (1 + ada['scale_msa']) + ada['shift_msa']
                else:
                    norm_x = self._rms_norm(local_x)
                
                # Project to Q, K, V using original attention weights
                attn = self.original_block.attn1
                attn_device = f'cuda:{gpu_id}'
                
                # Move attention weights to this GPU if needed
                to_q = attn.to_q.to(attn_device)
                to_k = attn.to_k.to(attn_device)
                to_v = attn.to_v.to(attn_device)
                
                q = to_q(norm_x)
                k = to_k(norm_x)
                v = to_v(norm_x)
                
                # Apply Q/K norms if present
                if attn.q_norm is not None:
                    q_norm = attn.q_norm.to(attn_device)
                    q = q_norm(q)
                if attn.k_norm is not None:
                    k_norm = attn.k_norm.to(attn_device)
                    k = k_norm(k)
                
                # Reshape for attention FIRST (before RoPE)
                heads = attn.heads
                dim_head = attn.dim_head
                q = q.view(batch, -1, heads, dim_head).transpose(1, 2).contiguous()
                k = k.view(batch, -1, heads, dim_head).transpose(1, 2).contiguous()
                v = v.view(batch, -1, heads, dim_head).transpose(1, 2).contiguous()
                
                # Apply RoPE AFTER reshaping (q/k are now batch, heads, seq, dim_head)
                if v_pe is not None:
                    q, k = self._apply_rope_chunked(q, k, v_pe, gpu_id, chunk_ranges[gpu_id])
                
                q_chunks[gpu_id] = q
                k_chunks[gpu_id] = k
                v_chunks[gpu_id] = v
                
                del norm_x
        
        # Step 2: Ring attention
        attn_out_chunks = self._ring_attention_forward(q_chunks, k_chunks, v_chunks)
        
        del q_chunks, k_chunks, v_chunks
        
        # Step 3: Output projection and residual
        for gpu_id in self.gpu_ids:
            if gpu_id not in attn_out_chunks:
                continue
            with torch.cuda.device(gpu_id):
                attn_out = attn_out_chunks[gpu_id]
                batch_size, heads, seq_local, dim = attn_out.shape
                
                # Reshape back
                attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_local, -1)
                
                # Output projection
                to_out = self.original_block.attn1.to_out.to(f'cuda:{gpu_id}')
                attn_out = to_out(attn_out)
                
                # Apply gate and residual
                if gpu_id in ada_values:
                    attn_out = attn_out * ada_values[gpu_id]['gate_msa']
                
                x_chunks[gpu_id] = x_chunks[gpu_id] + attn_out
                del attn_out
        
        del attn_out_chunks
        
        # ========== CROSS-ATTENTION TO TEXT (if present) ==========
        # av_model.py shows attn2 uses rms_norm
        if hasattr(self.original_block, 'attn2') and v_context is not None:
            for gpu_id in self.gpu_ids:
                local_x = x_chunks[gpu_id]
                if local_x.shape[1] == 0:
                    continue
                    
                with torch.cuda.device(gpu_id):
                    # Context is small - broadcast to all GPUs
                    ctx = v_context.to(f'cuda:{gpu_id}')
                    
                    # Move attn2 to this GPU
                    attn2 = self.original_block.attn2.to(f'cuda:{gpu_id}')
                    
                    # Cross-attention with rms_norm (av_model.py uses it)
                    attn_out = attn2(
                        self._rms_norm(local_x),
                        context=ctx,
                        mask=attention_mask.to(f'cuda:{gpu_id}') if attention_mask is not None else None,
                    )
                    
                    x_chunks[gpu_id] = x_chunks[gpu_id] + attn_out
                    del attn_out, ctx
        
        # ========== FFN (Independent per chunk) ==========
        # Original applies rms_norm, then scale/shift, then FFN, then gate
        if hasattr(self.original_block, 'ff'):
            for gpu_id in self.gpu_ids:
                local_x = x_chunks[gpu_id]
                if local_x.shape[1] == 0:
                    continue
                    
                with torch.cuda.device(gpu_id):
                    # Apply pre-norm and ada scaling (scale/shift IS correct)
                    if gpu_id in ada_values:
                        ada = ada_values[gpu_id]
                        norm_x = self._rms_norm(local_x)
                        norm_x = norm_x * (1 + ada['scale_mlp']) + ada['shift_mlp']
                    else:
                        norm_x = self._rms_norm(local_x)
                    
                    # Move FFN to this GPU
                    ff = self.original_block.ff.to(f'cuda:{gpu_id}')
                    
                    # Process FFN in sub-chunks for memory efficiency
                    seq_local = norm_x.shape[1]
                    sub_chunk_size = max(1, seq_local // 4)
                    ff_outputs = []
                    
                    for i in range(0, seq_local, sub_chunk_size):
                        end = min(i + sub_chunk_size, seq_local)
                        chunk = norm_x[:, i:end, :]
                        ff_out = ff(chunk)
                        ff_outputs.append(ff_out)
                        del chunk
                    
                    ff_out = torch.cat(ff_outputs, dim=1)
                    del ff_outputs, norm_x
                    
                    # Apply gate and residual
                    if gpu_id in ada_values:
                        ff_out = ff_out * ada_values[gpu_id]['gate_mlp']
                    
                    x_chunks[gpu_id] = x_chunks[gpu_id] + ff_out
                    del ff_out
        
        # Gather chunks back
        return self._gather_chunks(x_chunks, original_device)
    
    def _rms_norm(self, x, eps=1e-6):
        """RMS normalization."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    
    def _apply_rope_chunked(self, q, k, pe, gpu_id, chunk_range):
        """
        Apply RoPE to Q and K for a specific sequence chunk.
        
        Matches the original model.py implementation exactly:
        - pe[2] = True -> split RoPE
        - pe[2] = False (default) -> interleaved RoPE
        """
        start, end = chunk_range
        device = f'cuda:{gpu_id}'
        
        if pe is None:
            return q, k
        
        # Handle tuple PE format (cos, sin) or (cos, sin, split_flag)
        if isinstance(pe, (list, tuple)):
            cos = pe[0]
            sin = pe[1]
            split_pe = pe[2] if len(pe) > 2 else False
        else:
            return q, k  # Unknown format
        
        # Handle nested tuple format - pe[0] might be (cos, cross_cos)
        if isinstance(cos, (list, tuple)):
            cos = cos[0]
            sin = sin[0] if isinstance(sin, (list, tuple)) else sin
        
        # Move to device
        cos = cos.to(device)
        sin = sin.to(device)
        
        # Debug on first call ever (only at verbose >= 2)
        if self.verbose >= 2 and not getattr(SeqParallelBlockWrapper, '_rope_debug_logged', False):
            logger.info(f"RoPE: q={q.shape}, cos={cos.shape}, split_pe={split_pe}, chunk_range={chunk_range}")
            SeqParallelBlockWrapper._rope_debug_logged = True
        
        batch, heads, seq_local, dim_head = q.shape
        
        # Slice PE for this chunk's sequence positions
        try:
            if cos.ndim == 4:  # (batch, heads, seq, dim)
                if cos.shape[2] > seq_local:  # seq is dim 2
                    cos = cos[:, :, start:end, :]
                    sin = sin[:, :, start:end, :]
        except Exception as e:
            logger.warning(f"RoPE slice failed: {e}, using full PE")
        
        cos_dim = cos.shape[-1]
        
        if split_pe:
            # SPLIT RoPE - matches apply_split_rotary_emb in model.py
            # The original uses einops rearrange with d=2 to split into (2, dim/2)
            # Then: first_half_out = first_half * cos - sin * second_half
            #       second_half_out = second_half * cos + sin * first_half
            if self.verbose >= 2 and not getattr(SeqParallelBlockWrapper, '_rope_split_logged', False):
                logger.info(f"RoPE: Using SPLIT mode (pe[2]=True), cos_dim={cos_dim}, dim_head={dim_head}")
                SeqParallelBlockWrapper._rope_split_logged = True
            
            # q shape: (batch, heads, seq, dim_head) e.g. (1, 32, 255, 128)
            # cos shape after slice: (batch, heads, seq, dim_head/2) e.g. (1, 32, 255, 64)
            
            # Use the exact same approach as original: rearrange to (... d r) where d=2
            # This groups dim 128 as (2, 64), where [0,:] is first half and [1,:] is second half
            q_split = q.view(*q.shape[:-1], 2, dim_head // 2)  # (B, H, T, 2, 64)
            k_split = k.view(*k.shape[:-1], 2, dim_head // 2)
            
            q_first = q_split[..., 0, :]  # (B, H, T, 64) = elements 0,2,4...126
            q_second = q_split[..., 1, :]  # (B, H, T, 64) = elements 1,3,5...127
            k_first = k_split[..., 0, :]
            k_second = k_split[..., 1, :]
            
            # Apply rotation exactly as original:
            # first_out = first * cos - sin * second
            # second_out = second * cos + sin * first
            q_first_out = q_first * cos - sin * q_second
            q_second_out = q_second * cos + sin * q_first
            k_first_out = k_first * cos - sin * k_second
            k_second_out = k_second * cos + sin * k_first
            
            # Recombine
            q_out = torch.stack([q_first_out, q_second_out], dim=-2)  # (B, H, T, 2, 64)
            k_out = torch.stack([k_first_out, k_second_out], dim=-2)
            q = q_out.view(*q.shape[:-1], dim_head)  # (B, H, T, 128)
            k = k_out.view(*k.shape[:-1], dim_head)
        else:
            # INTERLEAVED RoPE - matches apply_interleaved_rotary_emb in model.py
            if self.verbose >= 2 and not getattr(SeqParallelBlockWrapper, '_rope_interleaved_logged', False):
                logger.info(f"RoPE: Using INTERLEAVED mode (pe[2]=False), cos_dim={cos_dim}, dim_head={dim_head}")
                SeqParallelBlockWrapper._rope_interleaved_logged = True
            
            # The original does: t_dup = rearrange(input, "... (d r) -> ... d r", r=2)
            # Then: t1, t2 = t_dup.unbind(dim=-1)  -> t1=even indices, t2=odd indices
            # Then: t_dup = torch.stack((-t2, t1), dim=-1)  -> rotated = (-t2, t1) NOT (t1, -t2)!
            # Then: out = input * cos + rotated * sin
            
            # Reshape to separate even/odd pairs
            q_reshape = q.view(*q.shape[:-1], -1, 2)  # (batch, heads, seq, dim/2, 2)
            k_reshape = k.view(*k.shape[:-1], -1, 2)
            
            q1, q2 = q_reshape[..., 0], q_reshape[..., 1]  # even, odd indices
            k1, k2 = k_reshape[..., 0], k_reshape[..., 1]
            
            # Rotated version: (-t2, t1)
            q_rot = torch.stack([-q2, q1], dim=-1).view_as(q)
            k_rot = torch.stack([-k2, k1], dim=-1).view_as(k)
            
            # Expand cos/sin to match full dim_head
            if cos_dim == dim_head // 2:
                # cos/sin are half dim, need to repeat for interleaved
                cos = cos.unsqueeze(-1).expand(*cos.shape, 2).reshape(*cos.shape[:-1], -1)
                sin = sin.unsqueeze(-1).expand(*sin.shape, 2).reshape(*sin.shape[:-1], -1)
            
            q = q * cos + q_rot * sin
            k = k * cos + k_rot * sin
        
        return q, k
    
    def _get_compute_device(self, module, fallback=None):
        """Get compute device for a module, handling CPU offloading."""
        # Default to first chunk GPU (not primary which holds VAE)
        if fallback is None:
            fallback = f'cuda:{self.gpu_ids[0]}'  # Use first chunk GPU, not primary
        try:
            device = next(module.parameters()).device
            if device.type == 'cpu':
                return torch.device(fallback)
            return device
        except StopIteration:
            return torch.device(fallback)
    
    def _ring_attention_forward(
        self,
        q_chunks: Dict[int, torch.Tensor],
        k_chunks: Dict[int, torch.Tensor],
        v_chunks: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """
        Ring attention: each GPU's Q attends to ALL K, V across the ring.
        
        Uses online softmax to accumulate attention without materializing
        the full attention matrix.
        
        For large sequences, we sub-chunk Q to avoid quadratic memory blow-up
        in the attention scores matrix.
        """
        # Clear caches once at start (not after every op)
        for gpu_id in self.gpu_ids:
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
        
        # Initialize outputs with online softmax accumulators (in float32 for precision)
        outputs = {}
        lse = {}  # log-sum-exp for numerical stability
        
        scale = 1.0 / math.sqrt(list(q_chunks.values())[0].shape[-1])
        
        for gpu_id in self.gpu_ids:
            if gpu_id not in q_chunks:
                continue
            q = q_chunks[gpu_id]
            batch, heads, seq_local, dim = q.shape
            device = q.device
            
            # Use float32 for accumulators to avoid precision loss
            outputs[gpu_id] = torch.zeros(batch, heads, seq_local, dim, device=device, dtype=torch.float32)
            lse[gpu_id] = torch.full(
                (batch, heads, seq_local, 1), float('-inf'),
                device=device, dtype=torch.float32
            )
        
        # Current K, V on each GPU (will rotate)
        current_k = {gpu_id: k_chunks[gpu_id].clone() for gpu_id in k_chunks}
        current_v = {gpu_id: v_chunks[gpu_id].clone() for gpu_id in v_chunks}
        
        active_gpus = [g for g in self.gpu_ids if g in q_chunks]
        
        # Determine sub-chunk size for Q to limit attention scores memory
        # MEMORY OPTIMIZATION: Target 1.0GB for scores to leave headroom
        # Stage 2 has 102K tokens / 6 GPUs = 17K tokens per GPU
        # scores = batch * heads * q_sub * seq_k * 4 bytes (float32)
        sample_q = list(q_chunks.values())[0]
        sample_k = list(k_chunks.values())[0]
        batch, heads, seq_q, dim = sample_q.shape
        seq_k = sample_k.shape[2]
        
        # Calculate scores memory: batch * heads * q_sub * k_len * 4 bytes (float32)
        target_bytes = int(1.0 * 1024 * 1024 * 1024)  # 1.0GB target (was 2.5GB)
        max_q_sub = target_bytes // (batch * heads * seq_k * 4)  # 4 bytes for float32
        max_q_sub = max(256, min(max_q_sub, seq_q))
        
        num_q_subs = (seq_q + max_q_sub - 1) // max_q_sub
        q_sub_size = (seq_q + num_q_subs - 1) // num_q_subs
        
        if self.verbose >= 1 and self.block_idx == 0:
            scores_mem = batch * heads * q_sub_size * seq_k * 4 / (1024**3)  # float32
            logger.info(f"Ring attention: seq_q={seq_q}, seq_k={seq_k}, q_sub_size={q_sub_size}, num_subs={num_q_subs}, scores_mem={scores_mem:.2f}GB")
        
        # Ring iterations
        for ring_step in range(len(active_gpus)):
            # Compute partial attention on each GPU
            for gpu_id in active_gpus:
                with torch.cuda.device(gpu_id):
                    q_local = q_chunks[gpu_id]
                    k_chunk = current_k[gpu_id]
                    v_chunk = current_v[gpu_id]
                    seq_q_local = q_local.shape[2]
                    
                    # Process Q in sub-chunks to limit memory
                    for q_start in range(0, seq_q_local, q_sub_size):
                        q_end = min(q_start + q_sub_size, seq_q_local)
                        q_sub = q_local[:, :, q_start:q_end, :]
                        
                        # Attention scores for this Q sub-chunk (compute in float32 for precision)
                        scores = torch.matmul(q_sub.float(), k_chunk.float().transpose(-2, -1)) * scale
                        
                        # Online softmax (in float32)
                        chunk_max = scores.max(dim=-1, keepdim=True).values
                        chunk_exp = torch.exp(scores - chunk_max)
                        del scores
                        chunk_sum = chunk_exp.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                        chunk_lse = chunk_max + torch.log(chunk_sum)
                        del chunk_max
                        
                        # Weighted values (compute in float32)
                        chunk_attn = chunk_exp / chunk_sum
                        del chunk_exp, chunk_sum
                        chunk_out = torch.matmul(chunk_attn, v_chunk.float())
                        del chunk_attn
                        
                        # Accumulate with online softmax correction (in float32)
                        old_lse = lse[gpu_id][:, :, q_start:q_end, :]
                        new_lse = torch.logaddexp(old_lse, chunk_lse)
                        
                        old_scale = torch.exp(old_lse - new_lse)
                        new_scale = torch.exp(chunk_lse - new_lse)
                        del chunk_lse
                        
                        # Update output (keep accumulator in float32)
                        outputs[gpu_id][:, :, q_start:q_end, :] = (
                            outputs[gpu_id][:, :, q_start:q_end, :] * old_scale + 
                            chunk_out * new_scale
                        )
                        lse[gpu_id][:, :, q_start:q_end, :] = new_lse
                        del old_scale, new_scale, chunk_out, q_sub
            
            # Rotate K, V (skip on last iteration)
            if ring_step < len(active_gpus) - 1:
                next_k = {}
                next_v = {}
                
                for i, gpu_id in enumerate(active_gpus):
                    prev_gpu = active_gpus[(i - 1) % len(active_gpus)]
                    with torch.cuda.device(gpu_id):
                        next_k[gpu_id] = current_k[prev_gpu].to(f'cuda:{gpu_id}', non_blocking=True)
                        next_v[gpu_id] = current_v[prev_gpu].to(f'cuda:{gpu_id}', non_blocking=True)
                
                # Sync before deleting old
                for gpu_id in active_gpus:
                    torch.cuda.synchronize(f'cuda:{gpu_id}')
                
                # Delete old K, V
                for gpu_id in active_gpus:
                    del current_k[gpu_id]
                    del current_v[gpu_id]
                
                current_k = next_k
                current_v = next_v
        
        # Cleanup
        for gpu_id in active_gpus:
            if gpu_id in current_k:
                del current_k[gpu_id]
            if gpu_id in current_v:
                del current_v[gpu_id]
            if gpu_id in lse:
                del lse[gpu_id]
        
        # NaN detection
        if self.verbose >= 1 and self.block_idx == 0:
            for gpu_id, out in outputs.items():
                if torch.isnan(out).any():
                    logger.warning(f"NaN detected in ring attention output on GPU {gpu_id}!")
                if torch.isinf(out).any():
                    logger.warning(f"Inf detected in ring attention output on GPU {gpu_id}!")
        
        # Convert back to original dtype (bf16)
        original_dtype = list(q_chunks.values())[0].dtype
        for gpu_id in outputs:
            outputs[gpu_id] = outputs[gpu_id].to(original_dtype)
        
        return outputs
    
    def _av_forward(self, vx: torch.Tensor, ax: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process audio-video streams with ring attention for video.
        
        Video stream (vx) is large - split across GPUs with ring attention.
        Audio stream (ax) is smaller - broadcast to all GPUs.
        """
        original_device = vx.device
        batch, seq_v, hidden_v = vx.shape
        
        # Clear caches at start of first block to handle fragmentation between stages
        if self.block_idx == 0:
            for gpu_id in self.gpu_ids:
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
            # Also clear primary GPU (holds VAE)
            with torch.cuda.device(self.primary_gpu):
                torch.cuda.empty_cache()
        
        # Log sequence size at first block
        if self.verbose >= 1 and self.block_idx == 0:
            logger.info(f"SeqParallelBlock 0: vx={vx.shape}, ax={ax.shape}, splitting to {self.num_gpus} GPUs")
        
        # Get kwargs
        v_context = kwargs.get('v_context')
        a_context = kwargs.get('a_context')
        v_timestep = kwargs.get('v_timestep')
        a_timestep = kwargs.get('a_timestep')
        v_pe = kwargs.get('v_pe')
        a_pe = kwargs.get('a_pe')
        v_cross_pe = kwargs.get('v_cross_pe')
        a_cross_pe = kwargs.get('a_cross_pe')
        transformer_options = kwargs.get('transformer_options', {})
        attention_mask = kwargs.get('attention_mask')
        
        # Run flags
        run_vx = transformer_options.get("run_vx", True)
        run_ax = transformer_options.get("run_ax", True) and ax.numel() > 0
        run_a2v = run_vx and transformer_options.get("a2v_cross_attn", True) and ax.numel() > 0
        run_v2a = run_ax and transformer_options.get("v2a_cross_attn", True)
        
        # Split VIDEO sequence across GPUs
        chunk_size = (seq_v + self.num_gpus - 1) // self.num_gpus
        vx_chunks = {}
        chunk_ranges = {}
        
        for i, gpu_id in enumerate(self.gpu_ids):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, seq_v)
            if start < seq_v:
                vx_chunks[gpu_id] = vx[:, start:end, :].to(f'cuda:{gpu_id}', non_blocking=True)
                chunk_ranges[gpu_id] = (start, end)
            else:
                vx_chunks[gpu_id] = torch.empty(batch, 0, hidden_v, device=f'cuda:{gpu_id}')
                chunk_ranges[gpu_id] = (start, start)
        
        # Audio is smaller - keep on primary GPU, broadcast as needed
        ax_local = ax  # Will broadcast when needed
        
        # Sync
        for gpu_id in self.gpu_ids:
            torch.cuda.synchronize(f'cuda:{gpu_id}')
        
        # ========== VIDEO SELF-ATTENTION (Ring Attention) ==========
        if run_vx and hasattr(self.original_block, 'attn1'):
            # Get the device where attention weights live (or use cuda:0 if offloaded to CPU)
            attn = self.original_block.attn1
            weight_device = self._get_compute_device(attn)
            
            # Compute Q, K, V on EACH GPU's own device (distributed compute)
            # ComfyUI's ops will auto-load weights to the target device
            q_chunks = {}
            k_chunks = {}
            v_chunks = {}
            gate_msa = {}
            
            for gpu_id in self.gpu_ids:
                local_vx = vx_chunks[gpu_id]
                if local_vx.shape[1] == 0:
                    continue
                
                # Compute on this GPU's device (weights load automatically)
                compute_device = f'cuda:{gpu_id}'
                
                with torch.cuda.device(compute_device):
                    # Ensure chunk is on its own GPU
                    local_vx_compute = local_vx.to(compute_device, non_blocking=True)
                    torch.cuda.synchronize(compute_device)
                    
                    # Get ada values
                    if hasattr(self.original_block, 'scale_shift_table') and v_timestep is not None:
                        ts = v_timestep.to(compute_device)
                        table = self.original_block.scale_shift_table.to(compute_device)
                        ada = self.original_block.get_ada_values(table, batch, ts, slice(0, 3))
                        shift_msa, scale_msa, gate_val = ada[0], ada[1], ada[2]
                        
                        # Slice if ada has sequence dimension
                        if shift_msa.ndim == 3 and shift_msa.shape[1] > 1:
                            start, end = chunk_ranges[gpu_id]
                            shift_msa = shift_msa[:, start:end, :]
                            scale_msa = scale_msa[:, start:end, :]
                            if gate_val.ndim == 3 and gate_val.shape[1] > 1:
                                gate_val = gate_val[:, start:end, :]
                        
                        norm_vx = self._rms_norm(local_vx_compute)
                        norm_vx = norm_vx * (1 + scale_msa) + shift_msa
                        gate_msa[gpu_id] = gate_val
                    else:
                        norm_vx = self._rms_norm(local_vx_compute)
                        gate_msa[gpu_id] = 1.0
                    
                    # Compute Q, K, V - weights auto-load to this device
                    q = attn.to_q(norm_vx)
                    k = attn.to_k(norm_vx)
                    v = attn.to_v(norm_vx)
                    del norm_vx
                    
                    # Norms
                    if attn.q_norm is not None:
                        q = attn.q_norm(q)
                    if attn.k_norm is not None:
                        k = attn.k_norm(k)
                    
                    # Reshape
                    heads = attn.heads
                    dim_head = attn.dim_head
                    q = q.view(batch, -1, heads, dim_head).transpose(1, 2).contiguous()
                    k = k.view(batch, -1, heads, dim_head).transpose(1, 2).contiguous()
                    v = v.view(batch, -1, heads, dim_head).transpose(1, 2).contiguous()
                    
                    # RoPE (applied to reshaped tensors)
                    if v_pe is not None:
                        q, k = self._apply_rope_chunked(q, k, v_pe, gpu_id, chunk_ranges[gpu_id])
                    
                    # Q, K, V are already on the right GPU
                    q_chunks[gpu_id] = q
                    k_chunks[gpu_id] = k
                    v_chunks[gpu_id] = v
                
                del local_vx_compute
            
            # Sync all GPUs
            for gpu_id in self.gpu_ids:
                if gpu_id in q_chunks:
                    torch.cuda.synchronize(f'cuda:{gpu_id}')
            
            # Ring attention
            attn_out = self._ring_attention_forward(q_chunks, k_chunks, v_chunks)
            
            del q_chunks, k_chunks, v_chunks
            
            # Output projection on EACH GPU's own device (distributed)
            for gpu_id in self.gpu_ids:
                if gpu_id not in attn_out:
                    continue
                
                with torch.cuda.device(gpu_id):
                    out = attn_out[gpu_id]
                    b, h, s, d = out.shape
                    out = out.transpose(1, 2).contiguous().view(b, s, -1)
                    
                    # Output projection - weights auto-load to this device
                    out = attn.to_out(out)
                
                if gpu_id in gate_msa:
                    out = out * gate_msa[gpu_id]
                
                vx_chunks[gpu_id] = vx_chunks[gpu_id] + out
                del out
            
            del attn_out, gate_msa
            
            # Clear caches on all chunk GPUs
            for gpu_id in self.gpu_ids:
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
        
        # ========== VIDEO CROSS-ATTENTION TO TEXT ==========
        # av_model.py line 187-192 shows: vx += self.attn2(comfy.ldm.common_dit.rms_norm(vx), ...)
        # So attn2 DOES use rms_norm!
        if run_vx and hasattr(self.original_block, 'attn2') and v_context is not None:
            attn2 = self.original_block.attn2
            
            for gpu_id in self.gpu_ids:
                local_vx = vx_chunks[gpu_id]
                if local_vx.shape[1] == 0:
                    continue
                
                # Compute on this GPU's own device
                with torch.cuda.device(gpu_id):
                    ctx = v_context.to(f'cuda:{gpu_id}', non_blocking=True)
                    torch.cuda.synchronize(gpu_id)
                    
                    # Apply rms_norm (av_model.py uses it for attn2)
                    out = attn2(
                        self._rms_norm(local_vx),
                        context=ctx,
                        mask=attention_mask.to(f'cuda:{gpu_id}') if attention_mask is not None else None,
                    )
                    del ctx
                vx_chunks[gpu_id] = vx_chunks[gpu_id] + out
                del out
            
            # Clear caches
            for gpu_id in self.gpu_ids:
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
        
        # ========== AUDIO SELF-ATTENTION (on weight device - audio is small) ==========
        if run_ax and hasattr(self.original_block, 'audio_attn1'):
            audio_attn1 = self.original_block.audio_attn1
            weight_device = self._get_compute_device(audio_attn1, f'cuda:{self.gpu_ids[0]}')
            
            with torch.cuda.device(weight_device):
                ax_gpu = ax_local.to(weight_device, non_blocking=True)
                torch.cuda.synchronize(weight_device)
                
                # Get ada values for audio
                if hasattr(self.original_block, 'audio_scale_shift_table') and a_timestep is not None:
                    ts = a_timestep.to(weight_device)
                    table = self.original_block.audio_scale_shift_table
                    if table.device != weight_device:
                        table = table.to(weight_device)
                    ada = self.original_block.get_ada_values(table, batch, ts, slice(0, 3))
                    shift_a, scale_a, gate_a = ada[0], ada[1], ada[2]
                    
                    norm_ax = self._rms_norm(ax_gpu)
                    norm_ax = norm_ax * (1 + scale_a) + shift_a
                else:
                    norm_ax = self._rms_norm(ax_gpu)
                    gate_a = 1.0
                
                # Move full PE tuple to device
                if a_pe is not None:
                    pe_a = tuple(
                        t.to(weight_device) if isinstance(t, torch.Tensor) else t
                        for t in a_pe
                    )
                else:
                    pe_a = None
                
                out = audio_attn1(norm_ax, pe=pe_a, transformer_options=transformer_options)
                ax_gpu = ax_gpu + out * gate_a
                
                ax_local = ax_gpu.to(original_device, non_blocking=True)
                torch.cuda.synchronize(original_device)
                del out, norm_ax
        
        # ========== AUDIO CROSS-ATTENTION TO TEXT ==========
        if run_ax and hasattr(self.original_block, 'audio_attn2') and a_context is not None:
            audio_attn2 = self.original_block.audio_attn2
            weight_device = self._get_compute_device(audio_attn2, f'cuda:{self.gpu_ids[0]}')
            
            with torch.cuda.device(weight_device):
                ax_gpu = ax_local.to(weight_device, non_blocking=True)
                ctx = a_context.to(weight_device, non_blocking=True)
                torch.cuda.synchronize(weight_device)
                
                out = audio_attn2(
                    self._rms_norm(ax_gpu),
                    context=ctx,
                    mask=attention_mask.to(weight_device) if attention_mask is not None else None,
                )
                ax_gpu = ax_gpu + out
                ax_local = ax_gpu.to(original_device, non_blocking=True)
                torch.cuda.synchronize(original_device)
                del out, ctx
        
        # ========== CROSS-MODAL ATTENTION ==========
        # These need: scale/shift ada values, position embeddings, and gates!
        # See av_model.py lines 215-297
        
        # Get cross-attention timestep parameters
        v_cross_scale_shift_timestep = kwargs.get('v_cross_scale_shift_timestep')
        a_cross_scale_shift_timestep = kwargs.get('a_cross_scale_shift_timestep')
        v_cross_gate_timestep = kwargs.get('v_cross_gate_timestep')
        a_cross_gate_timestep = kwargs.get('a_cross_gate_timestep')
        
        if run_a2v or run_v2a:
            # MEMORY OPTIMIZATION: Rotate compute device across blocks to spread memory load
            # Instead of always using GPU1, distribute cross-modal attention across all GPUs
            crossmodal_gpu_idx = self.block_idx % self.num_gpus
            weight_device = f'cuda:{self.gpu_ids[crossmodal_gpu_idx]}'
            
            # Gather vx for norm computation
            # NOTE: Original av_model.py computes vx_norm3 ONCE and uses it for BOTH a2v and v2a
            # v2a uses the PRE-a2v normalized video as context (not re-normalized after a2v)
            with torch.cuda.device(weight_device):
                vx_gathered = self._gather_chunks(vx_chunks, weight_device)
                ax_gpu = ax_local.to(weight_device, non_blocking=True)
                torch.cuda.synchronize(weight_device)
                
                # Compute norms ONCE for both a2v and v2a
                vx_norm3 = self._rms_norm(vx_gathered)
                ax_norm3 = self._rms_norm(ax_gpu)
                
                # MEMORY: Delete vx_gathered now - we only need vx_norm3 going forward
                del vx_gathered
                
                # Get ada values for cross-attention
                gate_out_a2v = None
                gate_out_v2a = None
                scale_shift_a2v_video = None
                scale_shift_a2v_audio = None
                scale_shift_v2a_video = None
                scale_shift_v2a_audio = None
                
                if hasattr(self.original_block, 'scale_shift_table_a2v_ca_audio') and a_cross_scale_shift_timestep is not None:
                    audio_ada = self.original_block.get_av_ca_ada_values(
                        self.original_block.scale_shift_table_a2v_ca_audio.to(weight_device),
                        batch,
                        a_cross_scale_shift_timestep.to(weight_device),
                        a_cross_gate_timestep.to(weight_device) if a_cross_gate_timestep is not None else None,
                    )
                    # Unpack: scale_a2v, shift_a2v, scale_v2a, shift_v2a, gate_v2a
                    scale_shift_a2v_audio = (audio_ada[0], audio_ada[1])
                    scale_shift_v2a_audio = (audio_ada[2], audio_ada[3])
                    gate_out_v2a = audio_ada[4] if len(audio_ada) > 4 else 1.0
                
                if hasattr(self.original_block, 'scale_shift_table_a2v_ca_video') and v_cross_scale_shift_timestep is not None:
                    video_ada = self.original_block.get_av_ca_ada_values(
                        self.original_block.scale_shift_table_a2v_ca_video.to(weight_device),
                        batch,
                        v_cross_scale_shift_timestep.to(weight_device),
                        v_cross_gate_timestep.to(weight_device) if v_cross_gate_timestep is not None else None,
                    )
                    # Unpack: scale_a2v, shift_a2v, scale_v2a, shift_v2a, gate_a2v
                    scale_shift_a2v_video = (video_ada[0], video_ada[1])
                    scale_shift_v2a_video = (video_ada[2], video_ada[3])
                    gate_out_a2v = video_ada[4] if len(video_ada) > 4 else 1.0
        
        # Audio-to-Video: Q=video (full), K/V=audio (broadcast)
        if run_a2v and hasattr(self.original_block, 'audio_to_video_attn'):
            a2v_attn = self.original_block.audio_to_video_attn
            
            with torch.cuda.device(weight_device):
                # Apply scale/shift to norms
                if scale_shift_a2v_video is not None:
                    vx_scaled = vx_norm3 * (1 + scale_shift_a2v_video[0]) + scale_shift_a2v_video[1]
                else:
                    vx_scaled = vx_norm3
                
                if scale_shift_a2v_audio is not None:
                    ax_scaled = ax_norm3 * (1 + scale_shift_a2v_audio[0]) + scale_shift_a2v_audio[1]
                else:
                    ax_scaled = ax_norm3
                
                # Prepare PE
                v_cross_pe_device = None
                a_cross_pe_device = None
                if v_cross_pe is not None:
                    if isinstance(v_cross_pe, (list, tuple)):
                        v_cross_pe_device = tuple(t.to(weight_device) if isinstance(t, torch.Tensor) else t for t in v_cross_pe)
                    else:
                        v_cross_pe_device = v_cross_pe.to(weight_device)
                if a_cross_pe is not None:
                    if isinstance(a_cross_pe, (list, tuple)):
                        a_cross_pe_device = tuple(t.to(weight_device) if isinstance(t, torch.Tensor) else t for t in a_cross_pe)
                    else:
                        a_cross_pe_device = a_cross_pe.to(weight_device)
                
                # Cross-attention with PE
                a2v_out = a2v_attn(
                    vx_scaled,
                    context=ax_scaled,
                    pe=v_cross_pe_device,
                    k_pe=a_cross_pe_device,
                    transformer_options=transformer_options,
                )
                
                # Apply gate
                if gate_out_a2v is not None and isinstance(gate_out_a2v, torch.Tensor):
                    a2v_out = a2v_out * gate_out_a2v
                
                # MEMORY OPTIMIZATION: Split a2v_out directly to chunks and add
                # This avoids keeping full vx_gathered around
                del vx_scaled, ax_scaled
            
            # Split a2v output to chunks and add to vx_chunks
            for gpu_id in self.gpu_ids:
                start, end = chunk_ranges[gpu_id]
                if start < end:
                    with torch.cuda.device(gpu_id):
                        chunk_out = a2v_out[:, start:end, :].to(f'cuda:{gpu_id}', non_blocking=True)
                        vx_chunks[gpu_id] = vx_chunks[gpu_id] + chunk_out
                        del chunk_out
            
            for gpu_id in self.gpu_ids:
                torch.cuda.synchronize(f'cuda:{gpu_id}')
            
            del a2v_out
            with torch.cuda.device(weight_device):
                torch.cuda.empty_cache()
        
        # Video-to-Audio: Q=audio (small), K/V=video (uses same vx_norm3 - per original av_model.py!)
        if run_v2a and hasattr(self.original_block, 'video_to_audio_attn'):
            v2a_attn = self.original_block.video_to_audio_attn
            
            with torch.cuda.device(weight_device):
                # NOTE: Original av_model.py uses the SAME vx_norm3 for both a2v and v2a
                # v2a uses pre-a2v normalized video as K/V context (not re-normalized!)
                
                # Apply scale/shift to norms
                if scale_shift_v2a_audio is not None:
                    ax_scaled = ax_norm3 * (1 + scale_shift_v2a_audio[0]) + scale_shift_v2a_audio[1]
                else:
                    ax_scaled = ax_norm3
                
                if scale_shift_v2a_video is not None:
                    vx_scaled = vx_norm3 * (1 + scale_shift_v2a_video[0]) + scale_shift_v2a_video[1]
                else:
                    vx_scaled = vx_norm3
                
                # Prepare PE (reversed for v2a)
                v_cross_pe_device = None
                a_cross_pe_device = None
                if a_cross_pe is not None:
                    if isinstance(a_cross_pe, (list, tuple)):
                        a_cross_pe_device = tuple(t.to(weight_device) if isinstance(t, torch.Tensor) else t for t in a_cross_pe)
                    else:
                        a_cross_pe_device = a_cross_pe.to(weight_device)
                if v_cross_pe is not None:
                    if isinstance(v_cross_pe, (list, tuple)):
                        v_cross_pe_device = tuple(t.to(weight_device) if isinstance(t, torch.Tensor) else t for t in v_cross_pe)
                    else:
                        v_cross_pe_device = v_cross_pe.to(weight_device)
                
                # Cross-attention with PE
                v2a_out = v2a_attn(
                    ax_scaled,
                    context=vx_scaled,
                    pe=a_cross_pe_device,
                    k_pe=v_cross_pe_device,
                    transformer_options=transformer_options,
                )
                
                # Apply gate
                if gate_out_v2a is not None and isinstance(gate_out_v2a, torch.Tensor):
                    v2a_out = v2a_out * gate_out_v2a
                
                ax_local = (ax_gpu + v2a_out).to(original_device, non_blocking=True)
                torch.cuda.synchronize(original_device)
                del v2a_out, vx_scaled, ax_scaled, ax_gpu
        
        # Cleanup cross-attention intermediates (vx_norm3, ax_norm3)
        if run_a2v or run_v2a:
            del vx_norm3, ax_norm3
            with torch.cuda.device(weight_device):
                torch.cuda.empty_cache()
        
        # ========== VIDEO FFN ==========
        # Original applies rms_norm, then scale/shift, then FFN, then gate
        if run_vx and hasattr(self.original_block, 'ff'):
            ff = self.original_block.ff
            
            for gpu_id in self.gpu_ids:
                local_vx = vx_chunks[gpu_id]
                if local_vx.shape[1] == 0:
                    continue
                
                # Compute on this GPU's own device
                with torch.cuda.device(gpu_id):
                    # Get ada values
                    if hasattr(self.original_block, 'scale_shift_table') and v_timestep is not None:
                        ts = v_timestep.to(f'cuda:{gpu_id}')
                        table = self.original_block.scale_shift_table.to(f'cuda:{gpu_id}')
                        ada = self.original_block.get_ada_values(table, batch, ts, slice(3, None))
                        shift_mlp, scale_mlp, gate_mlp = ada[0], ada[1], ada[2]
                        
                        # Slice if needed
                        if shift_mlp.ndim == 3 and shift_mlp.shape[1] > 1:
                            start, end = chunk_ranges[gpu_id]
                            shift_mlp = shift_mlp[:, start:end, :]
                            scale_mlp = scale_mlp[:, start:end, :]
                            if gate_mlp.ndim == 3 and gate_mlp.shape[1] > 1:
                                gate_mlp = gate_mlp[:, start:end, :]
                        
                        # Apply rms_norm AND scale/shift (this IS correct per original)
                        norm_vx = self._rms_norm(local_vx)
                        norm_vx = norm_vx * (1 + scale_mlp) + shift_mlp
                    else:
                        norm_vx = self._rms_norm(local_vx)
                        gate_mlp = 1.0
                    
                    # Sub-chunk FFN for memory
                    seq_local = norm_vx.shape[1]
                    sub_size = max(1, seq_local // 4)
                    ff_outs = []
                    for i in range(0, seq_local, sub_size):
                        end_idx = min(i + sub_size, seq_local)
                        ff_outs.append(ff(norm_vx[:, i:end_idx, :]))
                    out = torch.cat(ff_outs, dim=1)
                    del ff_outs, norm_vx
                    
                    # Apply gate
                    if isinstance(gate_mlp, torch.Tensor):
                        out = out * gate_mlp
                
                vx_chunks[gpu_id] = vx_chunks[gpu_id] + out
                del out
            
            # Clear caches
            for gpu_id in self.gpu_ids:
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
        
        # ========== AUDIO FFN ==========
        if run_ax and hasattr(self.original_block, 'audio_ff'):
            audio_ff = self.original_block.audio_ff
            weight_device = self._get_compute_device(audio_ff, f'cuda:{self.gpu_ids[0]}')
            
            with torch.cuda.device(weight_device):
                ax_gpu = ax_local.to(weight_device, non_blocking=True)
                torch.cuda.synchronize(weight_device)
                
                # Get ada values
                if hasattr(self.original_block, 'audio_scale_shift_table') and a_timestep is not None:
                    ts = a_timestep.to(weight_device)
                    table = self.original_block.audio_scale_shift_table
                    if table.device != weight_device:
                        table = table.to(weight_device)
                    ada = self.original_block.get_ada_values(table, batch, ts, slice(3, None))
                    shift_a, scale_a, gate_a = ada[0], ada[1], ada[2]
                    
                    norm_ax = self._rms_norm(ax_gpu)
                    norm_ax = norm_ax * (1 + scale_a) + shift_a
                else:
                    norm_ax = self._rms_norm(ax_gpu)
                    gate_a = 1.0
                
                out = audio_ff(norm_ax)
                ax_local = (ax_gpu + out * gate_a).to(original_device, non_blocking=True)
                torch.cuda.synchronize(original_device)
                del out, norm_ax
        
        # ========== GATHER VIDEO ==========
        vx_out = self._gather_chunks(vx_chunks, original_device)
        
        # NaN detection for final output
        if self.verbose >= 1 and self.block_idx == 0:
            if torch.isnan(vx_out).any():
                logger.warning(f"NaN detected in gathered vx_out!")
            if torch.isinf(vx_out).any():
                logger.warning(f"Inf detected in gathered vx_out!")
        
        # ========== AGGRESSIVE MEMORY CLEANUP ==========
        # Clear all GPU caches after each block to prevent memory accumulation
        del vx_chunks
        for gpu_id in self.gpu_ids:
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
        
        return vx_out, ax_local


# =============================================================================
# Main Pipeline
# =============================================================================

class SeqParallelPipeline:
    """
    Manages sequence parallelism across multiple GPUs.
    """
    
    _instances: Dict[int, 'SeqParallelPipeline'] = {}
    
    def __init__(
        self,
        model: nn.Module,
        gpu_ids: List[int] = None,
        verbose: int = 1,
    ):
        self.model = model
        self.model_id = id(model)
        self.verbose = verbose
        
        # Use all available GPUs by default
        if gpu_ids is None:
            gpu_ids = list(range(torch.cuda.device_count()))
        self.gpu_ids = gpu_ids
        self.num_gpus = len(gpu_ids)
        
        self.wrapped_blocks = {}
        self.original_forwards = {}
        
        SeqParallelPipeline._instances[self.model_id] = self
    
    @classmethod
    def get_instance(cls, model: nn.Module) -> Optional['SeqParallelPipeline']:
        return cls._instances.get(id(model))
    
    def setup(self) -> Dict[str, Any]:
        """Find and wrap transformer blocks."""
        info = {
            "num_gpus": self.num_gpus,
            "gpu_ids": self.gpu_ids,
            "blocks_wrapped": 0,
            "is_av_model": False,
        }
        
        if self.verbose >= 1:
            logger.info(f"SeqParallelV5: Setting up sequence parallelism...")
            logger.info(f"  Using {self.num_gpus} GPUs: {self.gpu_ids}")
            for gpu_id in self.gpu_ids:
                logger.info(f"  {get_gpu_memory_info(gpu_id)}")
        
        # Find transformer blocks
        blocks = []
        is_av_model = False
        
        # Look for LTXAV blocks first
        if hasattr(self.model, 'transformer_blocks'):
            for i, block in enumerate(self.model.transformer_blocks):
                if hasattr(block, 'audio_attn1'):
                    is_av_model = True
                blocks.append((f'transformer_blocks.{i}', block, i))
        
        # Also check for nested blocks
        if not blocks:
            for name, module in self.model.named_modules():
                if hasattr(module, 'attn1') and hasattr(module, 'ff'):
                    if hasattr(module, 'audio_attn1'):
                        is_av_model = True
                    blocks.append((name, module, len(blocks)))
        
        info["is_av_model"] = is_av_model
        
        if not blocks:
            if self.verbose >= 1:
                logger.warning("SeqParallelV5: No transformer blocks found!")
            return info
        
        if self.verbose >= 1:
            logger.info(f"  Found {len(blocks)} transformer blocks")
            logger.info(f"  AV model: {is_av_model}")
        
        # Wrap blocks
        num_blocks = len(blocks)
        for name, block, idx in blocks:
            wrapper = SeqParallelBlockWrapper(
                original_block=block,
                gpu_ids=self.gpu_ids,
                block_idx=idx,
                num_blocks=num_blocks,
                is_av_block=is_av_model,
                verbose=self.verbose,
            )
            
            self.original_forwards[name] = block.forward
            self.wrapped_blocks[name] = wrapper
            
            # Replace forward method
            block.forward = wrapper.forward
            
            info["blocks_wrapped"] += 1
        
        if self.verbose >= 1:
            logger.info(f"  Wrapped {info['blocks_wrapped']} blocks")
            logger.info(f"  Sequence will be split across {self.num_gpus} GPUs")
        
        return info
    
    def cleanup(self):
        """Restore original forwards."""
        for name, original_fwd in self.original_forwards.items():
            parts = name.split('.')
            module = self.model
            for part in parts[:-1]:
                module = getattr(module, part)
            target = getattr(module, parts[-1])
            target.forward = original_fwd
        
        self.wrapped_blocks.clear()
        self.original_forwards.clear()
        SeqParallelPipeline._instances.pop(self.model_id, None)
        
        if self.verbose >= 1:
            logger.info("SeqParallelV5: Cleanup complete")


# =============================================================================
# ComfyUI Node
# =============================================================================

class TensorParallelV5Node:
    """
    Sequence Parallelism V5 - Split sequence across all GPUs.
    
    Uses ring attention to compute full self-attention without
    any single GPU holding the full Q, K, V tensors.
    
    Best for: Very long sequences (600K+ tokens) from I2V upscaling.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        num_gpus = torch.cuda.device_count()
        return {
            "required": {
                "model": ("MODEL",),
                "num_gpus": ("INT", {
                    "default": min(num_gpus, 7),
                    "min": 2,
                    "max": num_gpus,
                    "tooltip": "Number of GPUs to use for sequence parallelism"
                }),
                "verbose": ("INT", {"default": 1, "min": 0, "max": 2}),
            },
        }
    
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "setup"
    CATEGORY = "multigpu/experimental"
    
    def setup(self, model, num_gpus, verbose):
        lines = ["Sequence Parallelism V5 - Ring Attention"]
        lines.append("=" * 50)
        lines.append("")
        
        try:
            # Get diffusion model
            target = model
            if hasattr(model, 'model'):
                target = model.model
                if hasattr(target, 'diffusion_model'):
                    target = target.diffusion_model
            
            # Check for existing instance
            existing = SeqParallelPipeline.get_instance(target)
            if existing:
                lines.append("Pipeline already active")
                existing.cleanup()
            
            # Create pipeline
            gpu_ids = list(range(num_gpus))
            pipeline = SeqParallelPipeline(
                model=target,
                gpu_ids=gpu_ids,
                verbose=verbose,
            )
            
            info = pipeline.setup()
            
            if info["blocks_wrapped"] == 0:
                lines.append("WARNING: No blocks wrapped!")
                lines.append("Model may not be compatible.")
            else:
                lines.append(f"Blocks wrapped: {info['blocks_wrapped']}")
                lines.append(f"GPUs used: {info['gpu_ids']}")
                lines.append(f"AV model: {info['is_av_model']}")
                lines.append("")
                lines.append("Sequence parallelism:")
                lines.append(f"  - Sequence split across {num_gpus} GPUs")
                lines.append(f"  - Each GPU holds 1/{num_gpus} of tokens")
                lines.append(f"  - Ring attention for self-attention")
                lines.append("")
                lines.append("Memory estimate:")
                seq_total = 614400  # Typical Stage 2 sequence
                seq_per_gpu = seq_total // num_gpus
                mem_per_gpu = (seq_per_gpu * 4096 * 2 * 3) / (1024**3)  # Q, K, V in bf16
                lines.append(f"  - Total sequence: ~{seq_total:,} tokens")
                lines.append(f"  - Per GPU: ~{seq_per_gpu:,} tokens")
                lines.append(f"  - Est. QKV per GPU: ~{mem_per_gpu:.1f}GB")
            
        except Exception as e:
            lines.append(f"ERROR: {e}")
            import traceback
            lines.append(traceback.format_exc())
        
        return (model, "\n".join(lines))


NODE_CLASS_MAPPINGS = {
    "TensorParallelV5Node": TensorParallelV5Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorParallelV5Node": "Tensor Parallel V5 (Sequence Parallel + Ring Attention)",
}
