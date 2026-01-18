"""
Sequence Chunked Block - Core Implementation
============================================

Hooks BasicTransformerBlock.forward() to process the sequence in chunks,
storing intermediate results on a secondary GPU.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple, Any, Callable
import logging
import math
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SeqChunkedBlock")


# ============================================================================
# Configuration
# ============================================================================

class SequenceChunkConfig:
    """Configuration for sequence-chunked block processing."""
    
    def __init__(
        self,
        storage_gpu: int = 1,
        compute_gpu: int = 0,
        chunk_size: int = 10000,
        min_seq_length: int = 30000,
        enabled: bool = True,
        verbose: int = 1,
        offload_intermediates: bool = True,
    ):
        self.storage_gpu = storage_gpu  # GPU to store chunks during processing
        self.compute_gpu = compute_gpu  # GPU for computation (where weights stream)
        self.chunk_size = chunk_size    # Tokens per chunk
        self.min_seq_length = min_seq_length  # Min sequence to trigger chunking
        self.enabled = enabled
        self.verbose = verbose
        self.offload_intermediates = offload_intermediates  # Store chunks on storage GPU
        
        # Stats
        self.blocks_chunked = 0
        self.total_chunks_processed = 0
        self.peak_memory_saved_mb = 0
        self.last_seq_len = 0
        self.last_num_chunks = 0
    
    @property
    def compute_device(self) -> str:
        return f'cuda:{self.compute_gpu}'
    
    @property
    def storage_device(self) -> str:
        return f'cuda:{self.storage_gpu}'


# Global state
_config: Optional[SequenceChunkConfig] = None
_original_block_forward: Optional[Callable] = None
_hook_installed: bool = False
_original_av_block_forward: Optional[Callable] = None
_av_hook_installed: bool = False


# ============================================================================
# Chunked Block Forward Pass (for BasicTransformerBlock - video only)
# ============================================================================

def chunked_block_forward(
    self,  # BasicTransformerBlock instance
    x: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    timestep: Optional[torch.Tensor] = None,
    pe: Optional[Tuple] = None,
    transformer_options: Dict = {},
) -> torch.Tensor:
    """
    Process transformer block in sequence chunks with GLOBAL self-attention.
    
    CRITICAL: Self-attention (attn1) needs each token to attend to ALL tokens.
    We achieve this by:
    1. Computing K and V for the FULL sequence (stored on storage GPU)
    2. Computing Q only for each chunk
    3. Running attention on storage GPU: Q_chunk @ K_full @ V_full
    
    This preserves the global attention pattern while reducing peak memory on GPU0.
    
    For cross-attention (attn2) and FFN, we can chunk freely since:
    - Cross-attn: context (text) is small and same for all chunks
    - FFN: Operates element-wise, no cross-token dependencies
    """
    global _config, _original_block_forward
    
    if _config is None or not _config.enabled:
        return _original_block_forward(self, x, context, attention_mask, timestep, pe, transformer_options)
    
    batch_size, seq_len, hidden_dim = x.shape
    
    # Only chunk if sequence is long enough
    if seq_len < _config.min_seq_length:
        if _config.verbose >= 2:
            logger.info(f"[CHUNK] Skipping: seq_len={seq_len} < min={_config.min_seq_length}")
        return _original_block_forward(self, x, context, attention_mask, timestep, pe, transformer_options)
    
    # Try chunked processing, fall back to original on error
    try:
        return _chunked_block_forward_impl(
            self, x, context, attention_mask, timestep, pe, transformer_options,
            batch_size, seq_len, hidden_dim
        )
    except Exception as e:
        logger.error(f"[CHUNK] Error in chunked forward, falling back to original: {e}")
        import traceback
        traceback.print_exc()
        # Clear any partial state
        torch.cuda.empty_cache()
        return _original_block_forward(self, x, context, attention_mask, timestep, pe, transformer_options)


def _chunked_block_forward_impl(
    self,
    x: torch.Tensor,
    context: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    timestep: Optional[torch.Tensor],
    pe: Optional[Tuple],
    transformer_options: Dict,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
) -> torch.Tensor:
    """Implementation of chunked forward pass."""
    global _config
    
    chunk_size = _config.chunk_size
    num_chunks = math.ceil(seq_len / chunk_size)
    
    if _config.verbose >= 1:
        x_size_mb = x.numel() * x.element_size() / 1024**2
        chunk_size_mb = (batch_size * chunk_size * hidden_dim * x.element_size()) / 1024**2
        logger.info(f"[CHUNK] Processing seq_len={seq_len} in {num_chunks} chunks of {chunk_size}")
        logger.info(f"[CHUNK]   Full tensor: {x_size_mb:.1f}MB, Per chunk: {chunk_size_mb:.1f}MB")
    
    # Determine devices
    compute_device = _config.compute_device
    storage_device = _config.storage_device
    
    import comfy.ldm.common_dit
    import comfy.ldm.modules.attention as attn_module
    from comfy.ldm.lightricks.model import apply_rotary_emb
    
    # Compute scale/shift values ONCE (they're the same for all chunks)
    # These have shape (batch, 1, dim) and broadcast across sequence
    scale_shift_table = self.scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
    timestep_reshaped = timestep.reshape(x.shape[0], timestep.shape[1], self.scale_shift_table.shape[0], -1)
    all_params = (scale_shift_table + timestep_reshaped).unbind(dim=2)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = all_params
    
    # Track memory
    if _config.verbose >= 2:
        mem_before = torch.cuda.memory_allocated(_config.compute_gpu) / 1024**2
        logger.info(f"[CHUNK] GPU{_config.compute_gpu} memory before: {mem_before:.1f}MB")
    
    # ================================================================
    # PHASE 1: Compute K and V for FULL sequence (for global self-attention)
    # ================================================================
    if _config.verbose >= 1:
        logger.info(f"[CHUNK] Phase 1: Computing full K, V for global self-attention")
    
    # Normalize and modulate full x for attn1 input
    x_norm = comfy.ldm.common_dit.rms_norm(x)
    x_norm = torch.addcmul(x_norm, x_norm, scale_msa)
    x_norm = x_norm.add_(shift_msa)
    
    # Compute K and V for full sequence
    attn1 = self.attn1
    k_full = attn1.to_k(x_norm)
    v_full = attn1.to_v(x_norm)
    k_full = attn1.k_norm(k_full)
    
    # Apply RoPE to K
    if pe is not None:
        k_full = apply_rotary_emb(k_full, pe)
    
    # Move K, V to storage GPU
    k_storage = k_full.to(storage_device, non_blocking=True)
    v_storage = v_full.to(storage_device, non_blocking=True)
    torch.cuda.synchronize(storage_device)
    del k_full, v_full
    
    if _config.verbose >= 1:
        k_size_mb = k_storage.numel() * k_storage.element_size() / 1024**2
        logger.info(f"[CHUNK]   K, V stored on GPU{_config.storage_gpu}: {k_size_mb:.1f}MB each")
    
    # Move full x to storage GPU for reading chunks
    x_storage = x.to(storage_device, non_blocking=True)
    x_norm_storage = x_norm.to(storage_device, non_blocking=True)
    torch.cuda.synchronize(storage_device)
    del x, x_norm
    torch.cuda.empty_cache()
    
    if _config.verbose >= 2:
        mem_after_kv = torch.cuda.memory_allocated(_config.compute_gpu) / 1024**2
        logger.info(f"[CHUNK] GPU{_config.compute_gpu} after K,V offload: {mem_after_kv:.1f}MB")
    
    # ================================================================
    # PHASE 2: Process each chunk with global K, V
    # ================================================================
    if _config.verbose >= 1:
        logger.info(f"[CHUNK] Phase 2: Processing {num_chunks} chunks with global attention")
    
    output_chunks = []
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, seq_len)
        
        if _config.verbose >= 2:
            logger.info(f"[CHUNK] Chunk {chunk_idx+1}/{num_chunks}: tokens {start_idx}-{end_idx}")
        
        # ============================================================
        # SELF-ATTENTION (attn1): Q_chunk @ K_full @ V_full on storage GPU
        # ============================================================
        
        # Get x_norm chunk from storage (already normalized and modulated)
        x_norm_chunk = x_norm_storage[:, start_idx:end_idx, :].to(compute_device, non_blocking=True)
        torch.cuda.synchronize(compute_device)
        
        # Compute Q for this chunk only
        q_chunk = attn1.to_q(x_norm_chunk)
        q_chunk = attn1.q_norm(q_chunk)
        del x_norm_chunk
        
        # Apply RoPE to Q chunk
        if pe is not None:
            cos_freqs, sin_freqs = pe[0], pe[1]
            split_pe = pe[2] if len(pe) > 2 else False
            if cos_freqs.dim() == 4:
                cos_chunk = cos_freqs[:, :, start_idx:end_idx, :]
                sin_chunk = sin_freqs[:, :, start_idx:end_idx, :]
            elif cos_freqs.dim() == 3:
                cos_chunk = cos_freqs[:, start_idx:end_idx, :]
                sin_chunk = sin_freqs[:, start_idx:end_idx, :]
            else:
                cos_chunk, sin_chunk = cos_freqs, sin_freqs
            pe_chunk = (cos_chunk, sin_chunk, split_pe) if len(pe) > 2 else (cos_chunk, sin_chunk)
            q_chunk = apply_rotary_emb(q_chunk, pe_chunk)
        
        # Move Q to storage GPU for attention
        q_storage = q_chunk.to(storage_device, non_blocking=True)
        torch.cuda.synchronize(storage_device)
        del q_chunk
        
        # Run attention on storage GPU
        with torch.cuda.device(storage_device):
            attn_out_storage = attn_module.optimized_attention(
                q_storage, k_storage, v_storage,
                attn1.heads,
                attn_precision=attn1.attn_precision,
                transformer_options=transformer_options
            )
        del q_storage
        
        # Move attention output back to compute GPU
        attn_out = attn_out_storage.to(compute_device, non_blocking=True)
        torch.cuda.synchronize(compute_device)
        del attn_out_storage
        
        # Get x chunk for residual and subsequent ops
        x_chunk = x_storage[:, start_idx:end_idx, :].to(compute_device, non_blocking=True)
        torch.cuda.synchronize(compute_device)
        
        # Apply output projection and gated residual
        attn_out = attn1.to_out(attn_out)
        
        # Handle gate_msa shape - slice if needed
        if gate_msa.shape[1] > 1:
            gate_chunk = gate_msa[:, start_idx:end_idx, :]
        else:
            gate_chunk = gate_msa
        
        x_chunk = x_chunk + attn_out * gate_chunk
        del attn_out
        
        # ============================================================
        # CROSS-ATTENTION (attn2): Uses text context (small, same for all chunks)
        # ============================================================
        x_chunk = x_chunk + self.attn2(
            comfy.ldm.common_dit.rms_norm(x_chunk),
            context=context,
            mask=attention_mask,
            transformer_options=transformer_options,
        )
        
        # ============================================================
        # FFN: Element-wise, chunk independently
        # ============================================================
        y = comfy.ldm.common_dit.rms_norm(x_chunk)
        
        # Handle mlp scale/shift/gate shapes
        if scale_mlp.shape[1] > 1:
            scale_mlp_chunk = scale_mlp[:, start_idx:end_idx, :]
            shift_mlp_chunk = shift_mlp[:, start_idx:end_idx, :]
            gate_mlp_chunk = gate_mlp[:, start_idx:end_idx, :]
        else:
            scale_mlp_chunk = scale_mlp
            shift_mlp_chunk = shift_mlp
            gate_mlp_chunk = gate_mlp
        
        y = y * (1 + scale_mlp_chunk) + shift_mlp_chunk
        x_chunk = x_chunk + self.ff(y) * gate_mlp_chunk
        del y
        
        # Store chunk result on storage GPU
        chunk_result = x_chunk.to(storage_device, non_blocking=True)
        torch.cuda.synchronize(storage_device)
        output_chunks.append(chunk_result)
        del x_chunk
        torch.cuda.empty_cache()
        
        if _config.verbose >= 2:
            mem_chunk = torch.cuda.memory_allocated(_config.compute_gpu) / 1024**2
            logger.info(f"[CHUNK]   Chunk {chunk_idx+1} done, GPU{_config.compute_gpu} mem: {mem_chunk:.1f}MB")
    
    # ================================================================
    # PHASE 3: Concatenate chunks and return
    # ================================================================
    if _config.verbose >= 1:
        logger.info(f"[CHUNK] Phase 3: Concatenating {len(output_chunks)} chunks")
    
    # Cleanup K, V
    del k_storage, v_storage, x_storage, x_norm_storage
    
    # Concatenate on storage GPU then move to compute
    output_storage = torch.cat(output_chunks, dim=1)
    del output_chunks
    
    output = output_storage.to(compute_device, non_blocking=True)
    torch.cuda.synchronize(compute_device)
    del output_storage
    torch.cuda.empty_cache()
    
    # Update stats
    _config.blocks_chunked += 1
    _config.total_chunks_processed += num_chunks
    _config.last_seq_len = seq_len
    _config.last_num_chunks = num_chunks
    
    if _config.verbose >= 1:
        out_size_mb = output.numel() * output.element_size() / 1024**2
        logger.info(f"[CHUNK] Block complete: {tuple(output.shape)} ({out_size_mb:.1f}MB)")
    
    return output


# ============================================================================
# Chunked AV Block Forward Pass (for BasicAVTransformerBlock - audio+video)
# ============================================================================

def chunked_av_block_forward(
    self,  # BasicAVTransformerBlock instance
    x: Tuple[torch.Tensor, torch.Tensor],
    v_context=None,
    a_context=None,
    attention_mask=None,
    v_timestep=None,
    a_timestep=None,
    v_pe=None,
    a_pe=None,
    v_cross_pe=None,
    a_cross_pe=None,
    v_cross_scale_shift_timestep=None,
    a_cross_scale_shift_timestep=None,
    v_cross_gate_timestep=None,
    a_cross_gate_timestep=None,
    transformer_options=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process AV transformer block in sequence chunks.
    
    Video is chunked, audio is processed normally (it's small).
    """
    global _config, _original_av_block_forward
    
    if _config is None or not _config.enabled:
        return _original_av_block_forward(
            self, x, v_context, a_context, attention_mask,
            v_timestep, a_timestep, v_pe, a_pe, v_cross_pe, a_cross_pe,
            v_cross_scale_shift_timestep, a_cross_scale_shift_timestep,
            v_cross_gate_timestep, a_cross_gate_timestep, transformer_options
        )
    
    if transformer_options is None:
        transformer_options = {}
    
    vx, ax = x
    batch_size, v_seq_len, v_hidden = vx.shape
    _, a_seq_len, a_hidden = ax.shape if ax.numel() > 0 else (batch_size, 0, 0)
    
    # Only chunk if video sequence is long enough
    if v_seq_len < _config.min_seq_length:
        if _config.verbose >= 2:
            logger.info(f"[CHUNK-AV] Skipping: v_seq_len={v_seq_len} < min={_config.min_seq_length}")
        return _original_av_block_forward(
            self, x, v_context, a_context, attention_mask,
            v_timestep, a_timestep, v_pe, a_pe, v_cross_pe, a_cross_pe,
            v_cross_scale_shift_timestep, a_cross_scale_shift_timestep,
            v_cross_gate_timestep, a_cross_gate_timestep, transformer_options
        )
    
    try:
        return _chunked_av_block_forward_impl(
            self, vx, ax, v_context, a_context, attention_mask,
            v_timestep, a_timestep, v_pe, a_pe, v_cross_pe, a_cross_pe,
            v_cross_scale_shift_timestep, a_cross_scale_shift_timestep,
            v_cross_gate_timestep, a_cross_gate_timestep, transformer_options,
            batch_size, v_seq_len, v_hidden, a_seq_len, a_hidden
        )
    except Exception as e:
        logger.error(f"[CHUNK-AV] Error in chunked forward, falling back: {e}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
        return _original_av_block_forward(
            self, x, v_context, a_context, attention_mask,
            v_timestep, a_timestep, v_pe, a_pe, v_cross_pe, a_cross_pe,
            v_cross_scale_shift_timestep, a_cross_scale_shift_timestep,
            v_cross_gate_timestep, a_cross_gate_timestep, transformer_options
        )


def _chunked_av_block_forward_impl(
    self,
    vx: torch.Tensor,
    ax: torch.Tensor,
    v_context, a_context, attention_mask,
    v_timestep, a_timestep, v_pe, a_pe, v_cross_pe, a_cross_pe,
    v_cross_scale_shift_timestep, a_cross_scale_shift_timestep,
    v_cross_gate_timestep, a_cross_gate_timestep, transformer_options,
    batch_size: int, v_seq_len: int, v_hidden: int, a_seq_len: int, a_hidden: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Implementation of chunked AV block forward."""
    global _config
    
    chunk_size = _config.chunk_size
    num_chunks = math.ceil(v_seq_len / chunk_size)
    
    compute_device = _config.compute_device
    storage_device = _config.storage_device
    
    run_vx = transformer_options.get("run_vx", True)
    run_ax = transformer_options.get("run_ax", True) and ax.numel() > 0
    run_a2v = run_vx and transformer_options.get("a2v_cross_attn", True) and ax.numel() > 0
    run_v2a = run_ax and transformer_options.get("v2a_cross_attn", True)
    
    if _config.verbose >= 1:
        vx_size_mb = vx.numel() * vx.element_size() / 1024**2
        ax_size_mb = ax.numel() * ax.element_size() / 1024**2 if ax.numel() > 0 else 0
        logger.info(f"[CHUNK-AV] Video: {v_seq_len} tokens ({vx_size_mb:.1f}MB), Audio: {a_seq_len} tokens ({ax_size_mb:.1f}MB)")
        logger.info(f"[CHUNK-AV] Processing video in {num_chunks} chunks of {chunk_size}")
    
    import comfy.ldm.common_dit
    import comfy.ldm.modules.attention as attn_module
    from comfy.ldm.lightricks.model import apply_rotary_emb
    
    # ================================================================
    # PHASE 0: Move vx to storage GPU FIRST to free GPU0 memory
    # ================================================================
    if _config.verbose >= 1:
        logger.info(f"[CHUNK-AV] Phase 0: Moving video to storage GPU to free compute GPU")
    
    vx_storage = vx.to(storage_device, non_blocking=True)
    torch.cuda.synchronize(storage_device)
    del vx
    torch.cuda.empty_cache()
    
    if _config.verbose >= 2:
        mem_after = torch.cuda.memory_allocated(_config.compute_gpu) / 1024**2
        logger.info(f"[CHUNK-AV] GPU{_config.compute_gpu} after vx offload: {mem_after:.1f}MB")
    
    # ================================================================
    # PHASE 1: Compute K and V in CHUNKS (to avoid OOM on full tensor)
    # ================================================================
    
    v_attn1 = self.attn1
    k_chunks_storage = []
    v_chunks_storage = []
    
    if run_vx:
        if _config.verbose >= 1:
            logger.info(f"[CHUNK-AV] Phase 1: Computing K, V in {num_chunks} chunks")
        
        # Get video scale/shift/gate values (compute once, slice per chunk later)
        # These may have shape (batch, seq_len, dim) for LTXAV!
        vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
            self.scale_shift_table, batch_size, v_timestep, slice(0, 3)
        )
        vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
            self.scale_shift_table, batch_size, v_timestep, slice(3, 6)
        )
        
        # Move ada values to storage for later use
        vshift_msa_storage = vshift_msa.to(storage_device, non_blocking=True)
        vscale_msa_storage = vscale_msa.to(storage_device, non_blocking=True)
        vgate_msa_storage = vgate_msa.to(storage_device, non_blocking=True)
        vshift_mlp_storage = vshift_mlp.to(storage_device, non_blocking=True)
        vscale_mlp_storage = vscale_mlp.to(storage_device, non_blocking=True)
        vgate_mlp_storage = vgate_mlp.to(storage_device, non_blocking=True)
        torch.cuda.synchronize(storage_device)
        del vshift_msa, vscale_msa, vgate_msa, vshift_mlp, vscale_mlp, vgate_mlp
        torch.cuda.empty_cache()
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, v_seq_len)
            
            # Get vx chunk from storage
            vx_chunk = vx_storage[:, start_idx:end_idx, :].to(compute_device, non_blocking=True)
            
            # Get ada values for this chunk (handle both broadcasted and per-token)
            if vscale_msa_storage.shape[1] > 1:
                vscale_chunk = vscale_msa_storage[:, start_idx:end_idx, :].to(compute_device, non_blocking=True)
                vshift_chunk = vshift_msa_storage[:, start_idx:end_idx, :].to(compute_device, non_blocking=True)
            else:
                vscale_chunk = vscale_msa_storage.to(compute_device, non_blocking=True)
                vshift_chunk = vshift_msa_storage.to(compute_device, non_blocking=True)
            torch.cuda.synchronize(compute_device)
            
            # Normalize and modulate
            vx_norm_chunk = comfy.ldm.common_dit.rms_norm(vx_chunk) * (1 + vscale_chunk) + vshift_chunk
            del vscale_chunk, vshift_chunk
            
            # Compute K and V for this chunk
            k_chunk = v_attn1.to_k(vx_norm_chunk)
            v_chunk = v_attn1.to_v(vx_norm_chunk)
            k_chunk = v_attn1.k_norm(k_chunk)
            
            # Apply RoPE to K chunk
            if v_pe is not None:
                cos_freqs, sin_freqs = v_pe[0], v_pe[1]
                split_pe = v_pe[2] if len(v_pe) > 2 else False
                if cos_freqs.dim() == 4:
                    cos_chunk = cos_freqs[:, :, start_idx:end_idx, :]
                    sin_chunk = sin_freqs[:, :, start_idx:end_idx, :]
                elif cos_freqs.dim() == 3:
                    cos_chunk = cos_freqs[:, start_idx:end_idx, :]
                    sin_chunk = sin_freqs[:, start_idx:end_idx, :]
                else:
                    cos_chunk, sin_chunk = cos_freqs, sin_freqs
                pe_chunk = (cos_chunk, sin_chunk, split_pe) if len(v_pe) > 2 else (cos_chunk, sin_chunk)
                k_chunk = apply_rotary_emb(k_chunk, pe_chunk)
            
            # Move K, V chunks to storage GPU
            k_chunk_storage = k_chunk.to(storage_device, non_blocking=True)
            v_chunk_storage = v_chunk.to(storage_device, non_blocking=True)
            torch.cuda.synchronize(storage_device)
            k_chunks_storage.append(k_chunk_storage)
            v_chunks_storage.append(v_chunk_storage)
            
            del vx_chunk, vx_norm_chunk, k_chunk, v_chunk
            torch.cuda.empty_cache()
            
            if _config.verbose >= 2:
                logger.info(f"[CHUNK-AV] K,V chunk {chunk_idx+1}/{num_chunks} computed and stored")
        
        # Concatenate K and V on storage GPU
        vk_storage = torch.cat(k_chunks_storage, dim=1)
        vv_storage = torch.cat(v_chunks_storage, dim=1)
        del k_chunks_storage, v_chunks_storage
        
        if _config.verbose >= 1:
            vk_size_mb = vk_storage.numel() * vk_storage.element_size() / 1024**2
            logger.info(f"[CHUNK-AV] Full K, V on GPU{_config.storage_gpu}: {vk_size_mb:.1f}MB each")
    
    # Process audio normally (it's small)
    if run_ax:
        ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
            self.audio_scale_shift_table, ax.shape[0], a_timestep, slice(0, 3)
        )
        ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
            self.audio_scale_shift_table, ax.shape[0], a_timestep, slice(3, 6)
        )
        
        # Audio self-attention
        norm_ax = comfy.ldm.common_dit.rms_norm(ax) * (1 + ascale_msa) + ashift_msa
        ax = ax + self.audio_attn1(norm_ax, pe=a_pe, transformer_options=transformer_options) * agate_msa
        
        # Audio cross-attention with text
        ax = ax + self.audio_attn2(
            comfy.ldm.common_dit.rms_norm(ax),
            context=a_context,
            mask=attention_mask,
            transformer_options=transformer_options,
        )
        del norm_ax
    
    # ================================================================
    # PRE-COMPUTE CROSS-ATTENTION ADA VALUES (for a2v)
    # ================================================================
    # These are per-token values that need to be sliced per chunk
    # Compute them ONCE here and store on storage GPU
    # IMPORTANT: Need to call BOTH audio and video ada value functions!
    
    if run_a2v:
        if _config.verbose >= 2:
            logger.info(f"[CHUNK-AV] Pre-computing audio-to-video cross-attention ada values")
        
        # Audio table → audio-sized ada values (for scaling ax in cross-attn)
        (
            a_scale_ca_a2v,  # For a2v: scale audio context
            a_shift_ca_a2v,  # For a2v: shift audio context
            a_scale_ca_v2a,  # For v2a: scale audio query (not used here)
            a_shift_ca_v2a,  # For v2a: shift audio query (not used here)
            gate_v2a,        # For v2a: gate (not used here)
        ) = self.get_av_ca_ada_values(
            self.scale_shift_table_a2v_ca_audio,
            batch_size,
            a_cross_scale_shift_timestep,
            a_cross_gate_timestep,
        )
        
        # Video table → video-sized ada values (for scaling vx and gate in cross-attn)
        (
            v_scale_ca_a2v_full,  # For a2v: scale video query - NEEDS SLICING
            v_shift_ca_a2v_full,  # For a2v: shift video query - NEEDS SLICING
            v_scale_ca_v2a,       # For v2a: scale video context (not used here)
            v_shift_ca_v2a,       # For v2a: shift video context (not used here)
            gate_a2v_full,        # For a2v: gate - NEEDS SLICING
        ) = self.get_av_ca_ada_values(
            self.scale_shift_table_a2v_ca_video,
            batch_size,
            v_cross_scale_shift_timestep,
            v_cross_gate_timestep,
        )
        
        # Store VIDEO ada values on storage GPU for per-chunk slicing (they're video-sequence-length)
        v_shift_ca_storage = v_shift_ca_a2v_full.to(storage_device, non_blocking=True)
        v_scale_ca_storage = v_scale_ca_a2v_full.to(storage_device, non_blocking=True)
        gate_a2v_storage = gate_a2v_full.to(storage_device, non_blocking=True)
        torch.cuda.synchronize(storage_device)
        del v_shift_ca_a2v_full, v_scale_ca_a2v_full, gate_a2v_full
        
        # Clean up unused v2a variables (they'll be recomputed in Phase 3 if needed)
        del a_scale_ca_v2a, a_shift_ca_v2a, gate_v2a
        del v_scale_ca_v2a, v_shift_ca_v2a
        
        # Audio ada values are small (audio sequence length), keep on compute device
        # (a_scale_ca_a2v, a_shift_ca_a2v are already on compute device)
    
    # ================================================================
    # PHASE 2: Process video chunks with global K, V
    # ================================================================
    
    video_output_chunks = []
    
    if run_vx:
        if _config.verbose >= 1:
            logger.info(f"[CHUNK-AV] Phase 2: Processing {num_chunks} video chunks with global attention")
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, v_seq_len)
            
            if _config.verbose >= 2:
                logger.info(f"[CHUNK-AV] Chunk {chunk_idx+1}/{num_chunks}: tokens {start_idx}-{end_idx}")
            
            # Get video chunk from storage
            vx_chunk = vx_storage[:, start_idx:end_idx, :].to(compute_device, non_blocking=True)
            
            # Get ada values for this chunk
            if vgate_msa_storage.shape[1] > 1:
                vgate_chunk = vgate_msa_storage[:, start_idx:end_idx, :].to(compute_device, non_blocking=True)
                vscale_mlp_chunk = vscale_mlp_storage[:, start_idx:end_idx, :].to(compute_device, non_blocking=True)
                vshift_mlp_chunk = vshift_mlp_storage[:, start_idx:end_idx, :].to(compute_device, non_blocking=True)
                vgate_mlp_chunk = vgate_mlp_storage[:, start_idx:end_idx, :].to(compute_device, non_blocking=True)
                vscale_msa_chunk = vscale_msa_storage[:, start_idx:end_idx, :].to(compute_device, non_blocking=True)
                vshift_msa_chunk = vshift_msa_storage[:, start_idx:end_idx, :].to(compute_device, non_blocking=True)
            else:
                vgate_chunk = vgate_msa_storage.to(compute_device, non_blocking=True)
                vscale_mlp_chunk = vscale_mlp_storage.to(compute_device, non_blocking=True)
                vshift_mlp_chunk = vshift_mlp_storage.to(compute_device, non_blocking=True)
                vgate_mlp_chunk = vgate_mlp_storage.to(compute_device, non_blocking=True)
                vscale_msa_chunk = vscale_msa_storage.to(compute_device, non_blocking=True)
                vshift_msa_chunk = vshift_msa_storage.to(compute_device, non_blocking=True)
            torch.cuda.synchronize(compute_device)
            
            # ============================================================
            # VIDEO SELF-ATTENTION (attn1) with global K, V
            # ============================================================
            vx_norm_chunk = comfy.ldm.common_dit.rms_norm(vx_chunk) * (1 + vscale_msa_chunk) + vshift_msa_chunk
            vq_chunk = v_attn1.to_q(vx_norm_chunk)
            vq_chunk = v_attn1.q_norm(vq_chunk)
            del vx_norm_chunk, vscale_msa_chunk, vshift_msa_chunk
            
            # Apply RoPE to Q
            if v_pe is not None:
                cos_freqs, sin_freqs = v_pe[0], v_pe[1]
                split_pe = v_pe[2] if len(v_pe) > 2 else False
                if cos_freqs.dim() == 4:
                    cos_chunk = cos_freqs[:, :, start_idx:end_idx, :]
                    sin_chunk = sin_freqs[:, :, start_idx:end_idx, :]
                elif cos_freqs.dim() == 3:
                    cos_chunk = cos_freqs[:, start_idx:end_idx, :]
                    sin_chunk = sin_freqs[:, start_idx:end_idx, :]
                else:
                    cos_chunk, sin_chunk = cos_freqs, sin_freqs
                pe_chunk = (cos_chunk, sin_chunk, split_pe) if len(v_pe) > 2 else (cos_chunk, sin_chunk)
                vq_chunk = apply_rotary_emb(vq_chunk, pe_chunk)
            
            # Move Q to storage GPU, run attention there
            vq_storage = vq_chunk.to(storage_device, non_blocking=True)
            torch.cuda.synchronize(storage_device)
            del vq_chunk
            
            with torch.cuda.device(storage_device):
                v_attn1_out_storage = attn_module.optimized_attention(
                    vq_storage, vk_storage, vv_storage,
                    v_attn1.heads,
                    attn_precision=v_attn1.attn_precision,
                    transformer_options=transformer_options
                )
            del vq_storage
            
            # Move back to compute GPU
            v_attn1_out = v_attn1_out_storage.to(compute_device, non_blocking=True)
            torch.cuda.synchronize(compute_device)
            del v_attn1_out_storage
            torch.cuda.empty_cache()
            
            # Apply output projection and residual
            v_attn1_out = v_attn1.to_out(v_attn1_out)
            vx_chunk = vx_chunk + v_attn1_out * vgate_chunk
            del v_attn1_out, vgate_chunk
            
            # ============================================================
            # VIDEO CROSS-ATTENTION (attn2) with text
            # ============================================================
            vx_chunk = vx_chunk + self.attn2(
                comfy.ldm.common_dit.rms_norm(vx_chunk),
                context=v_context,
                mask=attention_mask,
                transformer_options=transformer_options,
            )
            
            # ============================================================
            # AUDIO-TO-VIDEO CROSS-ATTENTION (if enabled)
            # ============================================================
            if run_a2v:
                # Get SLICED ada values for this chunk
                if v_scale_ca_storage.shape[1] > 1:
                    v_scale_ca_chunk = v_scale_ca_storage[:, start_idx:end_idx, :].to(compute_device, non_blocking=True)
                    v_shift_ca_chunk = v_shift_ca_storage[:, start_idx:end_idx, :].to(compute_device, non_blocking=True)
                    gate_a2v_chunk = gate_a2v_storage[:, start_idx:end_idx, :].to(compute_device, non_blocking=True)
                else:
                    v_scale_ca_chunk = v_scale_ca_storage.to(compute_device, non_blocking=True)
                    v_shift_ca_chunk = v_shift_ca_storage.to(compute_device, non_blocking=True)
                    gate_a2v_chunk = gate_a2v_storage.to(compute_device, non_blocking=True)
                torch.cuda.synchronize(compute_device)
                
                vx_norm3 = comfy.ldm.common_dit.rms_norm(vx_chunk)
                ax_norm3 = comfy.ldm.common_dit.rms_norm(ax)
                
                # Apply SLICED ada values to video chunk
                vx_norm3_mod = vx_norm3 * (1 + v_scale_ca_chunk) + v_shift_ca_chunk
                # Audio ada values are NOT sliced (audio is full sequence)
                ax_norm3_mod = ax_norm3 * (1 + a_scale_ca_a2v) + a_shift_ca_a2v
                
                # Slice v_cross_pe for this chunk (PE for video query positions)
                if v_cross_pe is not None:
                    v_cross_cos, v_cross_sin = v_cross_pe[0], v_cross_pe[1]
                    v_cross_split_pe = v_cross_pe[2] if len(v_cross_pe) > 2 else False
                    if v_cross_cos.dim() == 4:
                        v_cross_cos_chunk = v_cross_cos[:, :, start_idx:end_idx, :]
                        v_cross_sin_chunk = v_cross_sin[:, :, start_idx:end_idx, :]
                    elif v_cross_cos.dim() == 3:
                        v_cross_cos_chunk = v_cross_cos[:, start_idx:end_idx, :]
                        v_cross_sin_chunk = v_cross_sin[:, start_idx:end_idx, :]
                    else:
                        v_cross_cos_chunk, v_cross_sin_chunk = v_cross_cos, v_cross_sin
                    v_cross_pe_chunk = (v_cross_cos_chunk, v_cross_sin_chunk, v_cross_split_pe) if len(v_cross_pe) > 2 else (v_cross_cos_chunk, v_cross_sin_chunk)
                else:
                    v_cross_pe_chunk = None
                
                a2v_out = self.audio_to_video_attn(
                    vx_norm3_mod,
                    context=ax_norm3_mod,
                    pe=v_cross_pe_chunk,  # SLICED PE for video queries
                    k_pe=a_cross_pe,      # Full PE for audio keys (not chunked)
                    transformer_options=transformer_options,
                )
                vx_chunk = vx_chunk + a2v_out * gate_a2v_chunk
                del vx_norm3, ax_norm3, vx_norm3_mod, ax_norm3_mod, a2v_out
                del v_scale_ca_chunk, v_shift_ca_chunk, gate_a2v_chunk
            
            # ============================================================
            # VIDEO FFN
            # ============================================================
            vy = comfy.ldm.common_dit.rms_norm(vx_chunk)
            vy = vy * (1 + vscale_mlp_chunk) + vshift_mlp_chunk
            vx_chunk = vx_chunk + self.ff(vy) * vgate_mlp_chunk
            del vy, vscale_mlp_chunk, vshift_mlp_chunk, vgate_mlp_chunk
            
            # Store chunk result on storage GPU
            chunk_result = vx_chunk.to(storage_device, non_blocking=True)
            torch.cuda.synchronize(storage_device)
            video_output_chunks.append(chunk_result)
            del vx_chunk
            torch.cuda.empty_cache()
    
    # ================================================================
    # PHASE 3: Handle video-to-audio cross attention and audio FFN
    # ================================================================
    # For v2a: Q=audio (small), K/V=video (huge)
    # Keep video on storage GPU, move audio there, run attention, move result back
    
    if run_v2a:
        if _config.verbose >= 1:
            logger.info(f"[CHUNK-AV] Phase 3: Video-to-audio cross attention (on storage GPU)")
        
        # Need BOTH ada value functions for v2a:
        # - Audio table → audio-sized values for scaling ax (query)
        # - Video table → video-sized values for scaling vx (context/K,V)
        
        # Audio table → audio-sized ada values
        (
            _,  # a_scale_ca_a2v (not needed)
            _,  # a_shift_ca_a2v (not needed)
            a_scale_ca_v2a,  # For v2a: scale audio query
            a_shift_ca_v2a,  # For v2a: shift audio query
            gate_v2a,        # For v2a: gate
        ) = self.get_av_ca_ada_values(
            self.scale_shift_table_a2v_ca_audio,
            batch_size,
            a_cross_scale_shift_timestep,
            a_cross_gate_timestep,
        )
        
        # Video table → video-sized ada values
        (
            _,  # v_scale_ca_a2v (not needed)
            _,  # v_shift_ca_a2v (not needed)
            v_scale_ca_v2a,  # For v2a: scale video context
            v_shift_ca_v2a,  # For v2a: shift video context
            _,               # gate_a2v (not needed)
        ) = self.get_av_ca_ada_values(
            self.scale_shift_table_a2v_ca_video,
            batch_size,
            v_cross_scale_shift_timestep,
            v_cross_gate_timestep,
        )
        
        # Concatenate video chunks ON STORAGE GPU (don't move to compute GPU!)
        vx_full_storage = torch.cat(video_output_chunks, dim=1)
        
        # Move ada values to storage GPU for video modulation
        v_scale_ca_v2a_storage = v_scale_ca_v2a.to(storage_device, non_blocking=True)
        v_shift_ca_v2a_storage = v_shift_ca_v2a.to(storage_device, non_blocking=True)
        torch.cuda.synchronize(storage_device)
        del v_scale_ca_v2a, v_shift_ca_v2a
        
        # Compute normalized and modulated video ON STORAGE GPU
        with torch.cuda.device(storage_device):
            vx_norm3_storage = comfy.ldm.common_dit.rms_norm(vx_full_storage)
            vx_norm3_mod_storage = vx_norm3_storage * (1 + v_scale_ca_v2a_storage) + v_shift_ca_v2a_storage
            del vx_norm3_storage, v_scale_ca_v2a_storage, v_shift_ca_v2a_storage
        
        # Audio is small - normalize and modulate on compute GPU, then move to storage
        ax_norm3 = comfy.ldm.common_dit.rms_norm(ax)
        ax_norm3_mod = ax_norm3 * (1 + a_scale_ca_v2a) + a_shift_ca_v2a
        del ax_norm3, a_scale_ca_v2a, a_shift_ca_v2a
        
        ax_norm3_mod_storage = ax_norm3_mod.to(storage_device, non_blocking=True)
        torch.cuda.synchronize(storage_device)
        del ax_norm3_mod
        
        # Move positional embeddings to storage GPU
        # a_cross_pe is for audio queries (Q), v_cross_pe is for video keys (K)
        def move_pe_to_device(pe, device):
            if pe is None:
                return None
            cos, sin = pe[0], pe[1]
            cos_moved = cos.to(device, non_blocking=True)
            sin_moved = sin.to(device, non_blocking=True)
            if len(pe) > 2:
                return (cos_moved, sin_moved, pe[2])  # pe[2] is split_pe flag (bool)
            return (cos_moved, sin_moved)
        
        a_cross_pe_storage = move_pe_to_device(a_cross_pe, storage_device)
        v_cross_pe_storage = move_pe_to_device(v_cross_pe, storage_device)
        torch.cuda.synchronize(storage_device)
        
        # Run v2a cross-attention ON STORAGE GPU
        # Q=audio (small), K/V=video (huge, already on storage GPU)
        with torch.cuda.device(storage_device):
            v2a_out_storage = self.video_to_audio_attn(
                ax_norm3_mod_storage,
                context=vx_norm3_mod_storage,
                pe=a_cross_pe_storage,      # PE for audio queries - now on storage GPU
                k_pe=v_cross_pe_storage,    # PE for video keys - now on storage GPU
                transformer_options=transformer_options,
            )
        del ax_norm3_mod_storage, vx_norm3_mod_storage
        del a_cross_pe_storage, v_cross_pe_storage
        
        # Move result back to compute GPU and apply residual
        v2a_out = v2a_out_storage.to(compute_device, non_blocking=True)
        torch.cuda.synchronize(compute_device)
        del v2a_out_storage
        
        ax = ax + v2a_out * gate_v2a
        del v2a_out, gate_v2a
        torch.cuda.empty_cache()
    
    # Audio FFN
    if run_ax:
        ay = comfy.ldm.common_dit.rms_norm(ax)
        ay = ay * (1 + ascale_mlp) + ashift_mlp
        ax = ax + self.audio_ff(ay) * agate_mlp
        del ay
    
    # ================================================================
    # PHASE 4: Concatenate video and return
    # ================================================================
    
    if _config.verbose >= 1:
        logger.info(f"[CHUNK-AV] Phase 4: Concatenating {len(video_output_chunks)} video chunks")
    
    # Clean up storage
    if run_vx:
        del vk_storage, vv_storage, vx_storage
        del vshift_msa_storage, vscale_msa_storage, vgate_msa_storage
        del vshift_mlp_storage, vscale_mlp_storage, vgate_mlp_storage
    
    if run_a2v:
        del v_shift_ca_storage, v_scale_ca_storage, gate_a2v_storage
        del a_scale_ca_a2v, a_shift_ca_a2v
    
    if run_v2a:
        vx_out = vx_full_storage.to(compute_device, non_blocking=True)
        del vx_full_storage
    else:
        vx_out_storage = torch.cat(video_output_chunks, dim=1)
        vx_out = vx_out_storage.to(compute_device, non_blocking=True)
        del vx_out_storage
    
    del video_output_chunks
    torch.cuda.synchronize(compute_device)
    torch.cuda.empty_cache()
    
    # Update stats
    _config.blocks_chunked += 1
    _config.total_chunks_processed += num_chunks
    _config.last_seq_len = v_seq_len
    _config.last_num_chunks = num_chunks
    
    if _config.verbose >= 1:
        out_size_mb = vx_out.numel() * vx_out.element_size() / 1024**2
        logger.info(f"[CHUNK-AV] Block complete: video {tuple(vx_out.shape)} ({out_size_mb:.1f}MB)")
    
    return vx_out, ax


# ============================================================================
# Hook Installation
# ============================================================================

def install_sequence_chunk_hook() -> bool:
    """Install hooks on both BasicTransformerBlock and BasicAVTransformerBlock."""
    global _original_block_forward, _hook_installed
    global _original_av_block_forward, _av_hook_installed
    
    success = True
    
    # Hook BasicTransformerBlock (for LTXVModel - video only)
    if not _hook_installed:
        try:
            from comfy.ldm.lightricks.model import BasicTransformerBlock
            _original_block_forward = BasicTransformerBlock.forward
            BasicTransformerBlock.forward = chunked_block_forward
            _hook_installed = True
            logger.info("[CHUNK] Hook installed on BasicTransformerBlock.forward()")
        except Exception as e:
            logger.warning(f"[CHUNK] Could not hook BasicTransformerBlock: {e}")
            success = False
    
    # Hook BasicAVTransformerBlock (for LTXAVModel - audio+video)
    if not _av_hook_installed:
        try:
            from comfy.ldm.lightricks.av_model import BasicAVTransformerBlock
            _original_av_block_forward = BasicAVTransformerBlock.forward
            BasicAVTransformerBlock.forward = chunked_av_block_forward
            _av_hook_installed = True
            logger.info("[CHUNK] Hook installed on BasicAVTransformerBlock.forward()")
        except Exception as e:
            logger.warning(f"[CHUNK] Could not hook BasicAVTransformerBlock: {e}")
            # This is OK - model might not have AV support
    
    return success or _av_hook_installed


def uninstall_sequence_chunk_hook() -> bool:
    """Remove hooks and restore original forward methods."""
    global _original_block_forward, _hook_installed
    global _original_av_block_forward, _av_hook_installed
    
    # Restore BasicTransformerBlock
    if _hook_installed:
        try:
            from comfy.ldm.lightricks.model import BasicTransformerBlock
            if _original_block_forward is not None:
                BasicTransformerBlock.forward = _original_block_forward
                _original_block_forward = None
                _hook_installed = False
                logger.info("[CHUNK] BasicTransformerBlock hook removed")
        except Exception as e:
            logger.error(f"[CHUNK] Failed to remove BasicTransformerBlock hook: {e}")
    
    # Restore BasicAVTransformerBlock
    if _av_hook_installed:
        try:
            from comfy.ldm.lightricks.av_model import BasicAVTransformerBlock
            if _original_av_block_forward is not None:
                BasicAVTransformerBlock.forward = _original_av_block_forward
                _original_av_block_forward = None
                _av_hook_installed = False
                logger.info("[CHUNK] BasicAVTransformerBlock hook removed")
        except Exception as e:
            logger.error(f"[CHUNK] Failed to remove BasicAVTransformerBlock hook: {e}")
    
    return True


# ============================================================================
# ComfyUI Nodes
# ============================================================================

class SequenceChunkedBlockNode:
    """
    Sequence Chunked Block for Multi-GPU Memory Distribution
    
    Processes transformer blocks in sequence chunks, storing intermediate
    results on a secondary GPU. This dramatically reduces peak memory
    on the compute GPU.
    
    For 1920x1080 video with 120K tokens (Stage 2):
    - Without chunking: ~4.7GB peak (latent + FFN intermediate)
    - With chunking (10K): ~400MB peak per chunk
    
    Place this node BEFORE your sampler.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "storage_gpu": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 7,
                    "tooltip": "GPU to store chunks during processing (typically cuda:1)"
                }),
                "chunk_size": ("INT", {
                    "default": 10000,
                    "min": 1000,
                    "max": 50000,
                    "step": 1000,
                    "tooltip": "Tokens per chunk. Smaller = less memory, more overhead. 10K-15K recommended."
                }),
                "min_seq_length": ("INT", {
                    "default": 30000,
                    "min": 1000,
                    "max": 200000,
                    "step": 1000,
                    "tooltip": "Minimum sequence length to trigger chunking. Below this, runs normally."
                }),
                "verbose": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 2,
                    "tooltip": "0=silent, 1=summary, 2=detailed"
                }),
            },
        }
    
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "setup"
    CATEGORY = "multigpu"
    
    def setup(
        self, 
        model, 
        storage_gpu: int, 
        chunk_size: int, 
        min_seq_length: int, 
        verbose: int
    ):
        global _config
        
        lines = ["Sequence Chunked Block"]
        lines.append("=" * 50)
        
        try:
            # Check GPUs
            num_gpus = torch.cuda.device_count()
            lines.append(f"Available GPUs: {num_gpus}")
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                mem_gb = props.total_memory / 1024**3
                lines.append(f"  cuda:{i} = {props.name} ({mem_gb:.1f} GB)")
            
            if storage_gpu >= num_gpus:
                lines.append(f"")
                lines.append(f"ERROR: storage_gpu={storage_gpu} but only {num_gpus} GPUs available!")
                lines.append(f"Use storage_gpu=1 if you have 2 GPUs.")
                return (model, "\n".join(lines))
            
            if storage_gpu == 0:
                lines.append("")
                lines.append("WARNING: storage_gpu=0 is the compute GPU!")
                lines.append("This won't save memory. Use storage_gpu=1 or higher.")
                lines.append("")
            
            # Create config
            _config = SequenceChunkConfig(
                storage_gpu=storage_gpu,
                compute_gpu=0,
                chunk_size=chunk_size,
                min_seq_length=min_seq_length,
                enabled=True,
                verbose=verbose,
                offload_intermediates=True,
            )
            
            # Install hook
            if install_sequence_chunk_hook():
                lines.append("")
                lines.append("✓ Hook installed successfully")
            else:
                lines.append("")
                lines.append("✗ Hook installation failed - check terminal for errors")
                return (model, "\n".join(lines))
            
            lines.append("")
            lines.append("Configuration:")
            lines.append(f"  Compute GPU: cuda:0 (model weights stream here)")
            lines.append(f"  Storage GPU: cuda:{storage_gpu} (chunks stored here)")
            lines.append(f"  Chunk size: {chunk_size} tokens")
            lines.append(f"  Min seq length: {min_seq_length} tokens")
            
            lines.append("")
            lines.append("Memory estimates:")
            lines.append(f"  Per chunk (10K tokens): ~80MB latent + ~320MB FFN = ~400MB peak")
            lines.append(f"  Full 120K tokens: ~945MB latent + ~3.7GB FFN = ~4.7GB peak")
            lines.append(f"  Savings: ~4.3GB on compute GPU!")
            
            lines.append("")
            lines.append("How it works:")
            lines.append("  1. Full latent moved to storage GPU (frees compute GPU)")
            lines.append("  2. Each chunk (~10K tokens) is processed through full block:")
            lines.append("     - Self-attention (attn1)")
            lines.append("     - Cross-attention (attn2)")
            lines.append("     - FFN (4x expansion)")
            lines.append("  3. Chunk result stored on storage GPU")
            lines.append("  4. Repeat for all chunks")
            lines.append("  5. Concatenate and return to compute GPU")
            
        except Exception as e:
            lines.append(f"ERROR: {e}")
            import traceback
            lines.append(traceback.format_exc())
        
        return (model, "\n".join(lines))


class SequenceChunkedBlockReportNode:
    """Report statistics from sequence chunked block processing."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "report"
    CATEGORY = "multigpu"
    
    def report(self, images):
        global _config
        
        lines = ["Sequence Chunked Block Report"]
        lines.append("=" * 50)
        
        if _config is None:
            lines.append("No chunking active")
        else:
            lines.append(f"Blocks processed with chunking: {_config.blocks_chunked}")
            lines.append(f"Total chunks processed: {_config.total_chunks_processed}")
            lines.append(f"Last sequence length: {_config.last_seq_len}")
            lines.append(f"Last number of chunks: {_config.last_num_chunks}")
            
            if _config.last_seq_len > 0 and _config.last_num_chunks > 0:
                chunk_size = _config.last_seq_len / _config.last_num_chunks
                lines.append(f"Effective chunk size: {chunk_size:.0f} tokens")
        
        return (images, "\n".join(lines))


class SequenceChunkedBlockDisableNode:
    """Disable sequence chunking and restore original behavior."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any_input": ("*",),
            },
        }
    
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "disable"
    CATEGORY = "multigpu"
    
    def disable(self, any_input):
        global _config
        
        uninstall_sequence_chunk_hook()
        _config = None
        logger.info("[CHUNK] Sequence chunking disabled")
        
        return (any_input,)


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "SequenceChunkedBlockNode": SequenceChunkedBlockNode,
    "SequenceChunkedBlockReportNode": SequenceChunkedBlockReportNode,
    "SequenceChunkedBlockDisableNode": SequenceChunkedBlockDisableNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SequenceChunkedBlockNode": "Sequence Chunked Block (Multi-GPU)",
    "SequenceChunkedBlockReportNode": "Sequence Chunk Report",
    "SequenceChunkedBlockDisableNode": "Sequence Chunk Disable",
}
