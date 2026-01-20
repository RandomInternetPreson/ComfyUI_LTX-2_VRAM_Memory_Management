"""
LTX-2 Multi-GPU Chunked Processing v6
=====================================

KEY FIX: Process each preparation step ONCE, then immediately chunk & offload!

v5's problem: Calling _prepare_timestep per-chunk created FULL 7.73GB timestep
tensors each time (22 chunks × 7.73GB = 170GB total!).

v6 Strategy:
1. Call _process_input ONCE → immediately chunk vx and offload
2. Call _prepare_timestep ONCE → immediately chunk timestep and offload
3. Call _prepare_PE ONCE → immediately chunk PE and offload  
4. Process transformer blocks with pre-chunked data

Each massive tensor is created only ONCE, then immediately distributed
before the next massive tensor is created.

Architecture:
- GPU 0: Compute GPU (only holds one thing at a time)
- GPUs 1-N: Storage GPUs (hold chunks round-robin)

More GPUs = More frames!
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple, Any, Callable
import logging
import math
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LTXMultiGPU")


# ============================================================================
# Configuration
# ============================================================================

class MultiGPUConfig:
    def __init__(
        self,
        storage_gpus: List[int] = [1],
        compute_gpu: int = 0,
        transformer_chunk_size: int = 8000,
        upsampler_temporal_chunk: int = 8,
        upsampler_overlap: int = 2,
        min_seq_length: int = 60000,
        enabled: bool = True,
        verbose: int = 1,
    ):
        self.storage_gpus = storage_gpus
        self.compute_gpu = compute_gpu
        self.transformer_chunk_size = transformer_chunk_size
        self.upsampler_temporal_chunk = upsampler_temporal_chunk
        self.upsampler_overlap = upsampler_overlap
        self.min_seq_length = min_seq_length
        self.enabled = enabled
        self.verbose = verbose
        
    @property
    def compute_device(self) -> str:
        return f'cuda:{self.compute_gpu}'
    
    @property
    def num_storage_gpus(self) -> int:
        return len(self.storage_gpus)
    
    def get_storage_device(self, idx: int) -> str:
        gpu_idx = self.storage_gpus[idx % self.num_storage_gpus]
        return f'cuda:{gpu_idx}'
    
    def get_storage_gpu_id(self, idx: int) -> int:
        return self.storage_gpus[idx % self.num_storage_gpus]


# Global state
_config: Optional[MultiGPUConfig] = None
_original_forward: Optional[Callable] = None
_forward_hook_installed: bool = False
_original_upsampler_forward: Optional[Callable] = None
_upsampler_hook_installed: bool = False


# ============================================================================
# Utility Functions
# ============================================================================

def estimate_seq_length(x):
    """Estimate sequence length from input shape."""
    if isinstance(x, (list, tuple)):
        vx = x[0]
        if vx.dim() == 5:
            return vx.shape[2] * vx.shape[3] * vx.shape[4]
        elif vx.dim() == 3:
            return vx.shape[1]
    elif x.dim() == 5:
        return x.shape[2] * x.shape[3] * x.shape[4]
    elif x.dim() == 3:
        return x.shape[1]
    return 0


def chunk_tensor(tensor, chunk_size, dim=1):
    """Chunk a tensor along specified dimension and return list of chunks."""
    if tensor is None:
        return None
    seq_len = tensor.shape[dim]
    num_chunks = math.ceil(seq_len / chunk_size)
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, seq_len)
        if dim == 1:
            chunk = tensor[:, start:end]
        elif dim == 2:
            chunk = tensor[:, :, start:end]
        else:
            # Generic slicing
            slices = [slice(None)] * tensor.dim()
            slices[dim] = slice(start, end)
            chunk = tensor[tuple(slices)]
        chunks.append((chunk, start, end))
    return chunks


def chunk_and_offload(tensor, name, chunk_size, config, dim=1):
    """Chunk a tensor and distribute to storage GPUs. Returns list of (chunk, start, end, device, gpu_id)."""
    if tensor is None:
        return None, 0
    
    seq_len = tensor.shape[dim]
    num_chunks = math.ceil(seq_len / chunk_size)
    
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, seq_len)
        
        storage_device = config.get_storage_device(i)
        storage_gpu_id = config.get_storage_gpu_id(i)
        
        # Slice the tensor
        if dim == 1:
            chunk = tensor[:, start:end].to(storage_device, non_blocking=True)
        elif dim == 2:
            chunk = tensor[:, :, start:end].to(storage_device, non_blocking=True)
        else:
            slices = [slice(None)] * tensor.dim()
            slices[dim] = slice(start, end)
            chunk = tensor[tuple(slices)].to(storage_device, non_blocking=True)
        
        torch.cuda.synchronize(storage_gpu_id)
        chunks.append((chunk, start, end, storage_device, storage_gpu_id))
    
    if config.verbose >= 2:
        mb = tensor.numel() * tensor.element_size() / 1024**2
        logger.info(f"[OFFLOAD] {name}: {mb:.1f}MB -> {num_chunks} chunks")
    
    return chunks, num_chunks


def chunk_and_offload_nested(item, name, chunk_size, seq_len, config, dim=1):
    """Recursively chunk and offload nested structures (lists, tuples, tensors)."""
    if item is None:
        return None, 0
    
    if isinstance(item, torch.Tensor):
        # Check if this tensor has the sequence dimension
        if item.dim() > dim and item.shape[dim] == seq_len:
            return chunk_and_offload(item, name, chunk_size, config, dim)
        else:
            # Small tensor or broadcasted - just move to first storage GPU
            storage_device = config.get_storage_device(0)
            storage_gpu_id = config.get_storage_gpu_id(0)
            stored = item.to(storage_device, non_blocking=True)
            torch.cuda.synchronize(storage_gpu_id)
            return stored, 0  # 0 means not chunked
    
    elif isinstance(item, (list, tuple)):
        results = []
        num_chunks = 0
        for i, sub_item in enumerate(item):
            sub_result, sub_chunks = chunk_and_offload_nested(
                sub_item, f"{name}[{i}]", chunk_size, seq_len, config, dim
            )
            results.append(sub_result)
            if sub_chunks > 0:
                num_chunks = sub_chunks
        return type(item)(results), num_chunks
    
    else:
        # Not a tensor, return as-is
        return item, 0


def is_chunked_tensor_list(storage):
    """Check if storage is a list of (chunk, start, end, device, gpu_id) tuples."""
    if not isinstance(storage, list) or len(storage) == 0:
        return False
    first = storage[0]
    return isinstance(first, tuple) and len(first) == 5 and isinstance(first[0], torch.Tensor)


def get_chunk_from_nested(storage, chunk_idx, is_chunked, compute_device, compute_gpu):
    """Retrieve a chunk from nested storage structure."""
    if storage is None:
        return None
    
    # Check if this is a chunked tensor list: [(chunk, start, end, device, gpu_id), ...]
    if is_chunked_tensor_list(storage):
        chunk, start, end, dev, gid = storage[chunk_idx]
        result = chunk.to(compute_device, non_blocking=True)
        torch.cuda.synchronize(compute_gpu)
        return result
    
    elif isinstance(storage, (list, tuple)):
        # Nested structure - recurse
        results = []
        for sub_storage in storage:
            sub_result = get_chunk_from_nested(sub_storage, chunk_idx, is_chunked, compute_device, compute_gpu)
            results.append(sub_result)
        return type(storage)(results)
    
    elif isinstance(storage, torch.Tensor):
        # Not chunked, just move to compute
        result = storage.to(compute_device, non_blocking=True)
        torch.cuda.synchronize(compute_gpu)
        return result
    
    else:
        return storage


def chunk_pe_structure(pe_full, chunk_size, seq_len, config):
    """
    Chunk PE structure specifically for LTX-2 AV model.
    
    PE structure is: [(v_pe, v_cross_pe), (a_pe, a_cross_pe)]
    where each *_pe is a tuple (cos_freqs, sin_freqs, split_pe_bool)
    
    We need to chunk cos_freqs and sin_freqs along the sequence dimension.
    """
    if pe_full is None:
        return None, 0
    
    num_chunks = math.ceil(seq_len / chunk_size)
    
    def chunk_pe_tuple(pe_tuple, name):
        """Chunk a single PE tuple (cos, sin, split)."""
        if pe_tuple is None:
            return None
        
        if not isinstance(pe_tuple, tuple) or len(pe_tuple) < 2:
            # Not a standard PE tuple, just store on first GPU
            if isinstance(pe_tuple, torch.Tensor):
                dev = config.get_storage_device(0)
                gid = config.get_storage_gpu_id(0)
                stored = pe_tuple.to(dev, non_blocking=True)
                torch.cuda.synchronize(gid)
                return stored
            return pe_tuple
        
        cos_freqs, sin_freqs = pe_tuple[0], pe_tuple[1]
        split_pe = pe_tuple[2] if len(pe_tuple) > 2 else False
        
        # Determine sequence dimension
        if cos_freqs.dim() == 4:
            seq_dim = 2  # (batch, heads, seq, dim)
        elif cos_freqs.dim() == 3:
            seq_dim = 1  # (batch, seq, dim)
        else:
            # Can't chunk, just store
            dev = config.get_storage_device(0)
            gid = config.get_storage_gpu_id(0)
            cos_s = cos_freqs.to(dev, non_blocking=True)
            sin_s = sin_freqs.to(dev, non_blocking=True)
            torch.cuda.synchronize(gid)
            return (cos_s, sin_s, split_pe) if len(pe_tuple) > 2 else (cos_s, sin_s)
        
        pe_seq_len = cos_freqs.shape[seq_dim]
        
        # Check if this PE matches our sequence length
        if pe_seq_len != seq_len:
            # Different length (maybe audio), just store on first GPU
            dev = config.get_storage_device(0)
            gid = config.get_storage_gpu_id(0)
            cos_s = cos_freqs.to(dev, non_blocking=True)
            sin_s = sin_freqs.to(dev, non_blocking=True)
            torch.cuda.synchronize(gid)
            return (cos_s, sin_s, split_pe) if len(pe_tuple) > 2 else (cos_s, sin_s)
        
        # Chunk the cos and sin tensors
        cos_chunks = []
        sin_chunks = []
        
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, seq_len)
            
            dev = config.get_storage_device(i)
            gid = config.get_storage_gpu_id(i)
            
            if seq_dim == 2:
                cos_c = cos_freqs[:, :, start:end, :].to(dev, non_blocking=True)
                sin_c = sin_freqs[:, :, start:end, :].to(dev, non_blocking=True)
            else:
                cos_c = cos_freqs[:, start:end, :].to(dev, non_blocking=True)
                sin_c = sin_freqs[:, start:end, :].to(dev, non_blocking=True)
            
            torch.cuda.synchronize(gid)
            cos_chunks.append((cos_c, start, end, dev, gid))
            sin_chunks.append((sin_c, start, end, dev, gid))
        
        # Return structure: (chunked_cos_list, chunked_sin_list, split_pe)
        return (cos_chunks, sin_chunks, split_pe)
    
    # PE structure for AV model: [(v_pe, v_cross_pe), (a_pe, a_cross_pe)]
    if isinstance(pe_full, (list, tuple)) and len(pe_full) == 2:
        v_pe_pair = pe_full[0]  # (v_pe, v_cross_pe)
        a_pe_pair = pe_full[1]  # (a_pe, a_cross_pe)
        
        if isinstance(v_pe_pair, (list, tuple)) and len(v_pe_pair) == 2:
            # AV model structure
            v_pe, v_cross_pe = v_pe_pair
            a_pe, a_cross_pe = a_pe_pair
            
            v_pe_chunked = chunk_pe_tuple(v_pe, "v_pe")
            v_cross_pe_chunked = chunk_pe_tuple(v_cross_pe, "v_cross_pe")
            a_pe_chunked = chunk_pe_tuple(a_pe, "a_pe")
            a_cross_pe_chunked = chunk_pe_tuple(a_cross_pe, "a_cross_pe")
            
            pe_storage = [
                (v_pe_chunked, v_cross_pe_chunked),
                (a_pe_chunked, a_cross_pe_chunked)
            ]
            return pe_storage, num_chunks
    
    # Fallback: use generic chunking
    return chunk_and_offload_nested(pe_full, "pe", chunk_size, seq_len, config, dim=1)


def get_pe_chunk(pe_storage, chunk_idx, compute_device, compute_gpu):
    """
    Retrieve PE chunk for a specific chunk index.
    
    Returns the PE in the format expected by apply_rotary_emb: (cos, sin, split)
    """
    if pe_storage is None:
        return None
    
    def get_pe_tuple_chunk(pe_tuple_storage):
        """Get a single PE tuple chunk."""
        if pe_tuple_storage is None:
            return None
        
        if not isinstance(pe_tuple_storage, tuple):
            # Not a tuple, might be a tensor
            if isinstance(pe_tuple_storage, torch.Tensor):
                return pe_tuple_storage.to(compute_device, non_blocking=True)
            return pe_tuple_storage
        
        # Check if first element is a chunked list
        if len(pe_tuple_storage) >= 2:
            cos_storage = pe_tuple_storage[0]
            sin_storage = pe_tuple_storage[1]
            split_pe = pe_tuple_storage[2] if len(pe_tuple_storage) > 2 else False
            
            # Check if cos_storage is chunked
            if is_chunked_tensor_list(cos_storage):
                cos_chunk, _, _, _, _ = cos_storage[chunk_idx]
                sin_chunk, _, _, _, _ = sin_storage[chunk_idx]
                
                cos_c = cos_chunk.to(compute_device, non_blocking=True)
                sin_c = sin_chunk.to(compute_device, non_blocking=True)
                torch.cuda.synchronize(compute_gpu)
                
                return (cos_c, sin_c, split_pe) if len(pe_tuple_storage) > 2 else (cos_c, sin_c)
            else:
                # Not chunked, just tensors
                if isinstance(cos_storage, torch.Tensor):
                    cos_c = cos_storage.to(compute_device, non_blocking=True)
                    sin_c = sin_storage.to(compute_device, non_blocking=True)
                    torch.cuda.synchronize(compute_gpu)
                    return (cos_c, sin_c, split_pe) if len(pe_tuple_storage) > 2 else (cos_c, sin_c)
                return pe_tuple_storage
        
        return pe_tuple_storage
    
    # PE structure: [(v_pe_chunked, v_cross_pe_chunked), (a_pe_chunked, a_cross_pe_chunked)]
    if isinstance(pe_storage, (list, tuple)) and len(pe_storage) == 2:
        v_pe_pair = pe_storage[0]
        a_pe_pair = pe_storage[1]
        
        if isinstance(v_pe_pair, (list, tuple)) and len(v_pe_pair) == 2:
            v_pe_chunked, v_cross_pe_chunked = v_pe_pair
            a_pe_chunked, a_cross_pe_chunked = a_pe_pair
            
            v_pe_chunk = get_pe_tuple_chunk(v_pe_chunked)
            v_cross_pe_chunk = get_pe_tuple_chunk(v_cross_pe_chunked)
            a_pe_chunk = get_pe_tuple_chunk(a_pe_chunked)
            a_cross_pe_chunk = get_pe_tuple_chunk(a_cross_pe_chunked)
            
            return [
                (v_pe_chunk, v_cross_pe_chunk),
                (a_pe_chunk, a_cross_pe_chunk)
            ]
    
    # Fallback: use generic retrieval
    return get_chunk_from_nested(pe_storage, chunk_idx, True, compute_device, compute_gpu)


# ============================================================================
# Chunked Forward Hook
# ============================================================================

def chunked_forward(
    self, x, timestep, context, attention_mask, frame_rate=25, 
    transformer_options={}, keyframe_idxs=None, denoise_mask=None, **kwargs
):
    """Chunked forward pass with immediate offloading after each preparation step."""
    global _config, _original_forward
    
    if _config is None or not _config.enabled:
        return _original_forward(
            self, x, timestep, context, attention_mask, frame_rate,
            transformer_options, keyframe_idxs, denoise_mask=denoise_mask, **kwargs
        )
    
    est_seq_len = estimate_seq_length(x)
    
    if est_seq_len < _config.min_seq_length:
        if _config.verbose >= 1:
            logger.info(f"[FORWARD] Small sequence ({est_seq_len} tokens), using original")
        return _original_forward(
            self, x, timestep, context, attention_mask, frame_rate,
            transformer_options, keyframe_idxs, denoise_mask=denoise_mask, **kwargs
        )
    
    if _config.verbose >= 1:
        mem_before = torch.cuda.memory_allocated(_config.compute_gpu) / 1024**2
        logger.info(f"[FORWARD] GPU{_config.compute_gpu} before: {mem_before:.1f}MB")
        logger.info(f"[FORWARD] Large sequence ({est_seq_len} tokens), using CHUNKED pipeline")
    
    try:
        return _chunked_forward_v6(
            self, x, timestep, context, attention_mask, frame_rate,
            transformer_options, keyframe_idxs, denoise_mask, kwargs
        )
    except Exception as e:
        logger.error(f"[FORWARD] Error: {e}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
        logger.info("[FORWARD] Attempting fallback...")
        return _original_forward(
            self, x, timestep, context, attention_mask, frame_rate,
            transformer_options, keyframe_idxs, denoise_mask=denoise_mask, **kwargs
        )


def _chunked_forward_v6(
    self, x, timestep, context, attention_mask, frame_rate,
    transformer_options, keyframe_idxs, denoise_mask, kwargs
):
    """
    v6 Implementation: Process each step ONCE, immediately chunk & offload.
    """
    global _config
    
    import comfy.ldm.common_dit
    import comfy.ldm.modules.attention as attn_module
    from comfy.ldm.lightricks.model import apply_rotary_emb
    
    compute_device = _config.compute_device
    compute_gpu = _config.compute_gpu
    chunk_size = _config.transformer_chunk_size
    first_storage = _config.get_storage_device(0)
    first_storage_id = _config.get_storage_gpu_id(0)
    
    # Get input info
    if isinstance(x, list):
        input_dtype = x[0].dtype
        batch_size = x[0].shape[0]
    else:
        input_dtype = x.dtype
        batch_size = x.shape[0]
    
    merged_args = {**transformer_options, **kwargs}
    
    # ================================================================
    # STEP 1: Process input (creates patchified x on GPU0)
    # Then IMMEDIATELY chunk and offload
    # ================================================================
    
    if _config.verbose >= 1:
        logger.info(f"[FORWARD] Step 1: Processing input...")
        mem = torch.cuda.memory_allocated(compute_gpu) / 1024**2
        logger.info(f"[FORWARD] GPU{compute_gpu} before _process_input: {mem:.1f}MB")
    
    x_full, pixel_coords, additional_args = self._process_input(
        x, keyframe_idxs, denoise_mask, **merged_args
    )
    merged_args.update(additional_args)
    
    # Get sequence length from patchified output
    if isinstance(x_full, list):
        vx_full = x_full[0]
        ax_full = x_full[1] if len(x_full) > 1 else None
        is_av_model = True
    else:
        vx_full = x_full
        ax_full = None
        is_av_model = False
    
    v_seq_len = vx_full.shape[1]
    num_chunks = math.ceil(v_seq_len / chunk_size)
    
    if _config.verbose >= 1:
        mem = torch.cuda.memory_allocated(compute_gpu) / 1024**2
        logger.info(f"[FORWARD] GPU{compute_gpu} after _process_input: {mem:.1f}MB")
        logger.info(f"[FORWARD] Video tokens: {v_seq_len} -> {num_chunks} chunks of {chunk_size}")
    
    # IMMEDIATELY chunk and offload vx
    vx_chunks, _ = chunk_and_offload(vx_full, "vx", chunk_size, _config, dim=1)
    del vx_full
    
    # Audio is small, keep on first storage GPU
    if ax_full is not None and ax_full.numel() > 0:
        ax_storage = ax_full.to(first_storage, non_blocking=True)
        torch.cuda.synchronize(first_storage_id)
        del ax_full
    else:
        ax_storage = ax_full
    
    del x_full
    torch.cuda.empty_cache()
    
    if _config.verbose >= 1:
        mem = torch.cuda.memory_allocated(compute_gpu) / 1024**2
        logger.info(f"[FORWARD] GPU{compute_gpu} after vx offload: {mem:.1f}MB")
    
    # ================================================================
    # STEP 2: Prepare timestep (creates huge timestep tensors on GPU0)
    # Then IMMEDIATELY chunk and offload
    # ================================================================
    
    if _config.verbose >= 1:
        logger.info(f"[FORWARD] Step 2: Preparing timestep...")
        mem = torch.cuda.memory_allocated(compute_gpu) / 1024**2
        logger.info(f"[FORWARD] GPU{compute_gpu} before _prepare_timestep: {mem:.1f}MB")
    
    timestep_full, embedded_timestep_full = self._prepare_timestep(
        timestep, batch_size, input_dtype, **merged_args
    )
    
    if _config.verbose >= 1:
        mem = torch.cuda.memory_allocated(compute_gpu) / 1024**2
        logger.info(f"[FORWARD] GPU{compute_gpu} after _prepare_timestep: {mem:.1f}MB")
    
    # IMMEDIATELY chunk and offload timestep
    timestep_storage, ts_is_chunked = chunk_and_offload_nested(
        timestep_full, "timestep", chunk_size, v_seq_len, _config, dim=1
    )
    del timestep_full
    torch.cuda.empty_cache()
    
    # IMMEDIATELY chunk and offload embedded_timestep
    emb_ts_storage, emb_is_chunked = chunk_and_offload_nested(
        embedded_timestep_full, "embedded_timestep", chunk_size, v_seq_len, _config, dim=1
    )
    del embedded_timestep_full
    torch.cuda.empty_cache()
    
    if _config.verbose >= 1:
        mem = torch.cuda.memory_allocated(compute_gpu) / 1024**2
        logger.info(f"[FORWARD] GPU{compute_gpu} after timestep offload: {mem:.1f}MB")
    
    # ================================================================
    # STEP 3: Prepare context (small, keep on compute)
    # ================================================================
    
    # We need a reference x for context preparation - use first chunk temporarily
    vx_ref = vx_chunks[0][0].to(compute_device, non_blocking=True)
    torch.cuda.synchronize(compute_gpu)
    
    if is_av_model:
        x_ref = [vx_ref, ax_storage.to(compute_device) if ax_storage is not None else torch.empty(0, device=compute_device)]
    else:
        x_ref = vx_ref
    
    context_prepared, attention_mask_prepared = self._prepare_context(
        context, batch_size, x_ref, attention_mask
    )
    attention_mask_prepared = self._prepare_attention_mask(attention_mask_prepared, input_dtype)
    
    del vx_ref, x_ref
    torch.cuda.empty_cache()
    
    # Context is small, keep on compute device or move to first storage
    # Actually, we need it for every attention, so keep on compute
    
    # ================================================================
    # STEP 4: Prepare positional embeddings (creates huge PE on GPU0)
    # Then IMMEDIATELY chunk and offload
    # ================================================================
    
    if _config.verbose >= 1:
        logger.info(f"[FORWARD] Step 4: Preparing PE...")
        mem = torch.cuda.memory_allocated(compute_gpu) / 1024**2
        logger.info(f"[FORWARD] GPU{compute_gpu} before _prepare_PE: {mem:.1f}MB")
    
    pe_full = self._prepare_positional_embeddings(pixel_coords, frame_rate, input_dtype)
    
    if _config.verbose >= 1:
        mem = torch.cuda.memory_allocated(compute_gpu) / 1024**2
        logger.info(f"[FORWARD] GPU{compute_gpu} after _prepare_PE: {mem:.1f}MB")
    
    # IMMEDIATELY chunk and offload PE using dedicated PE function
    pe_storage, pe_num_chunks = chunk_pe_structure(pe_full, chunk_size, v_seq_len, _config)
    del pe_full
    torch.cuda.empty_cache()
    
    if _config.verbose >= 1:
        mem = torch.cuda.memory_allocated(compute_gpu) / 1024**2
        logger.info(f"[FORWARD] GPU{compute_gpu} after PE offload: {mem:.1f}MB")
        logger.info(f"[FORWARD] All preparation complete. Starting transformer blocks...")
    
    # ================================================================
    # STEP 5: Process transformer blocks with chunked attention
    # ================================================================
    
    # For AV model, we need to handle both video and audio attention
    # Audio is small (~2K tokens), so doesn't need chunking
    
    # Extract context for video/audio if AV model
    if is_av_model and isinstance(context_prepared, (list, tuple)):
        v_context = context_prepared[0]
        a_context = context_prepared[1] if len(context_prepared) > 1 else None
    else:
        v_context = context_prepared
        a_context = None
    
    # Get audio tensors (small, keep on compute device)
    # ax_storage is on first_storage, we'll move it to compute when needed
    ax_compute = None
    if is_av_model and ax_storage is not None and ax_storage.numel() > 0:
        ax_compute = ax_storage.to(compute_device, non_blocking=True)
        torch.cuda.synchronize(compute_gpu)
    
    # Extract audio timestep and PE from storage (audio parts are NOT chunked)
    # timestep structure: [v_timestep, a_timestep, (av_ca_audio_ss, av_ca_video_ss, av_ca_a2v_gate, av_ca_v2a_gate)]
    # pe structure: [(v_pe, v_cross_pe), (a_pe, a_cross_pe)]
    
    a_timestep = None
    av_ca_audio_scale_shift = None
    av_ca_video_scale_shift = None
    av_ca_a2v_gate = None
    av_ca_v2a_gate = None
    a_pe = None
    a_cross_pe = None
    v_cross_pe_full = None  # We'll need this for cross-modal attention
    
    if is_av_model:
        # Get audio timestep - it's small, stored on first storage GPU (not chunked)
        if isinstance(timestep_storage, (list, tuple)) and len(timestep_storage) > 1:
            a_ts_storage = timestep_storage[1]
            if isinstance(a_ts_storage, torch.Tensor):
                a_timestep = a_ts_storage.to(compute_device, non_blocking=True)
            elif is_chunked_tensor_list(a_ts_storage):
                # It was chunked, gather it (shouldn't happen for audio but just in case)
                a_timestep = torch.cat([c.to(compute_device) for c, _, _, _, _ in a_ts_storage], dim=1)
            torch.cuda.synchronize(compute_gpu)
        
        # Get cross-attention timesteps (index 2 in timestep structure)
        if isinstance(timestep_storage, (list, tuple)) and len(timestep_storage) > 2:
            ca_timesteps = timestep_storage[2]
            if isinstance(ca_timesteps, (list, tuple)) and len(ca_timesteps) >= 4:
                # These may or may not be chunked depending on their dimensions
                def get_full_tensor(item):
                    if item is None:
                        return None
                    if isinstance(item, torch.Tensor):
                        return item.to(compute_device, non_blocking=True)
                    if is_chunked_tensor_list(item):
                        return torch.cat([c.to(compute_device) for c, _, _, _, _ in item], dim=1)
                    return item
                
                av_ca_audio_scale_shift = get_full_tensor(ca_timesteps[0])
                av_ca_video_scale_shift = get_full_tensor(ca_timesteps[1])
                av_ca_a2v_gate = get_full_tensor(ca_timesteps[2])
                av_ca_v2a_gate = get_full_tensor(ca_timesteps[3])
                torch.cuda.synchronize(compute_gpu)
        
        # Get audio PE from pe_storage
        # pe_storage structure: [(v_pe_chunked, v_cross_pe_chunked), (a_pe, a_cross_pe)]
        if isinstance(pe_storage, (list, tuple)) and len(pe_storage) > 1:
            a_pe_pair = pe_storage[1]  # (a_pe, a_cross_pe)
            if isinstance(a_pe_pair, (list, tuple)) and len(a_pe_pair) >= 2:
                a_pe_storage = a_pe_pair[0]
                a_cross_pe_storage = a_pe_pair[1]
                
                def get_pe_tuple(pe_item):
                    """Convert stored PE to (cos, sin, split) tuple on compute device."""
                    if pe_item is None:
                        return None
                    if isinstance(pe_item, tuple) and len(pe_item) >= 2:
                        # Check if already just tensors
                        if isinstance(pe_item[0], torch.Tensor):
                            cos = pe_item[0].to(compute_device, non_blocking=True)
                            sin = pe_item[1].to(compute_device, non_blocking=True)
                            split = pe_item[2] if len(pe_item) > 2 else False
                            return (cos, sin, split)
                        # Check if chunked
                        if is_chunked_tensor_list(pe_item[0]):
                            cos = torch.cat([c.to(compute_device) for c, _, _, _, _ in pe_item[0]], dim=1)
                            sin = torch.cat([c.to(compute_device) for c, _, _, _, _ in pe_item[1]], dim=1)
                            split = pe_item[2] if len(pe_item) > 2 else False
                            return (cos, sin, split)
                    return pe_item
                
                a_pe = get_pe_tuple(a_pe_storage)
                a_cross_pe = get_pe_tuple(a_cross_pe_storage)
                torch.cuda.synchronize(compute_gpu)
    
    for block_idx, block in enumerate(self.transformer_blocks):
        if _config.verbose >= 1 and (block_idx % 8 == 0 or block_idx == len(self.transformer_blocks) - 1):
            mem = torch.cuda.memory_allocated(compute_gpu) / 1024**2
            logger.info(f"[FORWARD] Block {block_idx+1}/{len(self.transformer_blocks)}, GPU{compute_gpu}: {mem:.1f}MB")
        
        v_attn1 = block.attn1
        
        # ============================================================
        # PART A: Compute K, V for all video chunks
        # ============================================================
        k_chunks = []
        v_chunks_kv = []
        ada_storage = []
        
        for chunk_idx in range(num_chunks):
            # Get vx chunk
            vx_data, start, end, dev, gid = vx_chunks[chunk_idx]
            vx_compute = vx_data.to(compute_device, non_blocking=True)
            torch.cuda.synchronize(compute_gpu)
            
            # Get timestep chunk
            ts_chunk = get_chunk_from_nested(timestep_storage, chunk_idx, ts_is_chunked, compute_device, compute_gpu)
            
            # Extract video timestep from nested structure
            if isinstance(ts_chunk, (list, tuple)):
                v_ts = ts_chunk[0]
            else:
                v_ts = ts_chunk
            
            # Compute ada values
            shift_msa, scale_msa, gate_msa = block.get_ada_values(
                block.scale_shift_table, batch_size, v_ts, slice(0, 3)
            )
            shift_mlp, scale_mlp, gate_mlp = block.get_ada_values(
                block.scale_shift_table, batch_size, v_ts, slice(3, 6)
            )
            
            storage_device = _config.get_storage_device(chunk_idx)
            ada_storage.append((
                shift_msa.to(storage_device), scale_msa.to(storage_device), gate_msa.to(storage_device),
                shift_mlp.to(storage_device), scale_mlp.to(storage_device), gate_mlp.to(storage_device),
            ))
            
            # Compute K, V
            vx_norm = comfy.ldm.common_dit.rms_norm(vx_compute) * (1 + scale_msa) + shift_msa
            k = v_attn1.to_k(vx_norm)
            v_val = v_attn1.to_v(vx_norm)
            if hasattr(v_attn1, 'k_norm') and v_attn1.k_norm is not None:
                k = v_attn1.k_norm(k)
            
            # Get PE chunk and apply RoPE to K
            pe_chunk = get_pe_chunk(pe_storage, chunk_idx, compute_device, compute_gpu)
            if pe_chunk is not None:
                # pe_chunk is [(v_pe, v_cross_pe), (a_pe, a_cross_pe)]
                # v_pe is (cos, sin, split)
                if isinstance(pe_chunk, (list, tuple)) and len(pe_chunk) > 0:
                    v_pe_pair = pe_chunk[0]  # (v_pe, v_cross_pe)
                    if isinstance(v_pe_pair, (list, tuple)) and len(v_pe_pair) > 0:
                        v_pe = v_pe_pair[0]  # (cos, sin, split)
                        if v_pe is not None and isinstance(v_pe, tuple) and len(v_pe) >= 2:
                            k = apply_rotary_emb(k, v_pe)
            
            # Store K, V on storage GPU
            k_chunks.append(k.to(storage_device, non_blocking=True))
            v_chunks_kv.append(v_val.to(storage_device, non_blocking=True))
            
            del vx_compute, vx_norm, k, v_val, ts_chunk, pe_chunk
            del shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
            torch.cuda.empty_cache()
        
        # Sync all storage GPUs
        for gpu_id in _config.storage_gpus:
            torch.cuda.synchronize(gpu_id)
        
        # Concatenate K, V on first storage GPU
        k_full = torch.cat([k.to(first_storage) for k in k_chunks], dim=1)
        v_full = torch.cat([v.to(first_storage) for v in v_chunks_kv], dim=1)
        del k_chunks, v_chunks_kv
        
        # ============================================================
        # PART B: Video self-attention + text cross-attention for each chunk
        # (Store intermediate results - don't do FFN yet!)
        # ============================================================
        intermediate_vx_chunks = []
        
        for chunk_idx in range(num_chunks):
            vx_data, start, end, storage_device, storage_gpu_id = vx_chunks[chunk_idx]
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_storage[chunk_idx]
            
            # Move to compute
            vx_compute = vx_data.to(compute_device, non_blocking=True)
            scale_msa_c = scale_msa.to(compute_device, non_blocking=True)
            shift_msa_c = shift_msa.to(compute_device, non_blocking=True)
            gate_msa_c = gate_msa.to(compute_device, non_blocking=True)
            torch.cuda.synchronize(compute_gpu)
            
            # Compute Q
            vx_norm = comfy.ldm.common_dit.rms_norm(vx_compute) * (1 + scale_msa_c) + shift_msa_c
            q = v_attn1.to_q(vx_norm)
            if hasattr(v_attn1, 'q_norm') and v_attn1.q_norm is not None:
                q = v_attn1.q_norm(q)
            del vx_norm
            
            # Apply RoPE to Q
            pe_chunk = get_pe_chunk(pe_storage, chunk_idx, compute_device, compute_gpu)
            if pe_chunk is not None:
                if isinstance(pe_chunk, (list, tuple)) and len(pe_chunk) > 0:
                    v_pe_pair = pe_chunk[0]
                    if isinstance(v_pe_pair, (list, tuple)) and len(v_pe_pair) > 0:
                        v_pe = v_pe_pair[0]
                        if v_pe is not None and isinstance(v_pe, tuple) and len(v_pe) >= 2:
                            q = apply_rotary_emb(q, v_pe)
            del pe_chunk
            
            # Attention on first storage GPU
            q_storage = q.to(first_storage, non_blocking=True)
            torch.cuda.synchronize(first_storage_id)
            del q
            
            with torch.cuda.device(first_storage):
                attn_out = attn_module.optimized_attention(
                    q_storage, k_full, v_full,
                    v_attn1.heads,
                    attn_precision=v_attn1.attn_precision,
                    transformer_options=transformer_options
                )
            del q_storage
            
            # Back to compute
            attn_out = attn_out.to(compute_device, non_blocking=True)
            torch.cuda.synchronize(compute_gpu)
            
            attn_out = v_attn1.to_out(attn_out)
            vx_compute = vx_compute + attn_out * gate_msa_c
            del attn_out, scale_msa_c, shift_msa_c, gate_msa_c
            
            # Video text cross-attention
            vx_compute = vx_compute + block.attn2(
                comfy.ldm.common_dit.rms_norm(vx_compute),
                context=v_context,
                mask=attention_mask_prepared,
                transformer_options=transformer_options,
            )
            
            # Store intermediate (before FFN and cross-modal attention)
            intermediate_vx_chunks.append((vx_compute.to(storage_device, non_blocking=True), start, end, storage_device, storage_gpu_id))
            del vx_compute
            torch.cuda.empty_cache()
        
        del k_full, v_full
        
        # ============================================================
        # PART C: Audio self-attention + text cross-attention
        # ============================================================
        if is_av_model and ax_compute is not None and ax_compute.numel() > 0:
            # Audio ada values
            ashift_msa, ascale_msa, agate_msa = block.get_ada_values(
                block.audio_scale_shift_table, batch_size, a_timestep, slice(0, 3)
            )
            
            # Audio self-attention
            norm_ax = comfy.ldm.common_dit.rms_norm(ax_compute) * (1 + ascale_msa) + ashift_msa
            ax_compute = ax_compute + block.audio_attn1(
                norm_ax, pe=a_pe, transformer_options=transformer_options
            ) * agate_msa
            del norm_ax, ashift_msa, ascale_msa, agate_msa
            
            # Audio text cross-attention
            if a_context is not None:
                ax_compute = ax_compute + block.audio_attn2(
                    comfy.ldm.common_dit.rms_norm(ax_compute),
                    context=a_context,
                    mask=attention_mask_prepared,
                    transformer_options=transformer_options,
                )
        
        # ============================================================
        # PART D: Cross-modal attention (a2v and v2a)
        # ============================================================
        if is_av_model and ax_compute is not None and ax_compute.numel() > 0:
            run_a2v = transformer_options.get("a2v_cross_attn", True)
            run_v2a = transformer_options.get("v2a_cross_attn", True)
            
            if run_a2v or run_v2a:
                ax_norm3 = comfy.ldm.common_dit.rms_norm(ax_compute)
                
                # Get cross-attention ada values for audio
                if av_ca_audio_scale_shift is not None and av_ca_v2a_gate is not None:
                    (
                        scale_ca_audio_a2v, shift_ca_audio_a2v,
                        scale_ca_audio_v2a, shift_ca_audio_v2a,
                        gate_out_v2a,
                    ) = block.get_av_ca_ada_values(
                        block.scale_shift_table_a2v_ca_audio,
                        ax_compute.shape[0],
                        av_ca_audio_scale_shift,
                        av_ca_v2a_gate,
                    )
                else:
                    run_a2v = False
                    run_v2a = False
                
                # a2v: For each video chunk, add audio context
                if run_a2v:
                    a2v_output_chunks = []
                    
                    for chunk_idx in range(num_chunks):
                        vx_data, start, end, storage_device, storage_gpu_id = intermediate_vx_chunks[chunk_idx]
                        vx_compute = vx_data.to(compute_device, non_blocking=True)
                        torch.cuda.synchronize(compute_gpu)
                        
                        vx_norm3 = comfy.ldm.common_dit.rms_norm(vx_compute)
                        
                        # Get chunk-specific cross PE
                        pe_chunk = get_pe_chunk(pe_storage, chunk_idx, compute_device, compute_gpu)
                        v_cross_pe_chunk = None
                        if pe_chunk is not None and isinstance(pe_chunk, (list, tuple)) and len(pe_chunk) > 0:
                            v_pe_pair = pe_chunk[0]
                            if isinstance(v_pe_pair, (list, tuple)) and len(v_pe_pair) > 1:
                                v_cross_pe_chunk = v_pe_pair[1]
                        
                        # Get video cross-attention ada values (per chunk if needed)
                        if av_ca_video_scale_shift is not None and av_ca_a2v_gate is not None:
                            # Check if these are per-token (have seq dimension matching full video)
                            # If so, we need to slice them for this chunk
                            v_ca_ss = av_ca_video_scale_shift
                            v_ca_gate = av_ca_a2v_gate
                            
                            if v_ca_ss.dim() > 1 and v_ca_ss.shape[1] == v_seq_len:
                                v_ca_ss = v_ca_ss[:, start:end]
                            if v_ca_gate.dim() > 1 and v_ca_gate.shape[1] == v_seq_len:
                                v_ca_gate = v_ca_gate[:, start:end]
                            
                            (
                                scale_ca_video_a2v, shift_ca_video_a2v,
                                scale_ca_video_v2a_chunk, shift_ca_video_v2a_chunk,
                                gate_out_a2v,
                            ) = block.get_av_ca_ada_values(
                                block.scale_shift_table_a2v_ca_video,
                                vx_compute.shape[0],
                                v_ca_ss,
                                v_ca_gate,
                            )
                            
                            vx_scaled = vx_norm3 * (1 + scale_ca_video_a2v) + shift_ca_video_a2v
                            ax_scaled = ax_norm3 * (1 + scale_ca_audio_a2v) + shift_ca_audio_a2v
                            
                            a2v_out = block.audio_to_video_attn(
                                vx_scaled,
                                context=ax_scaled,
                                pe=v_cross_pe_chunk,
                                k_pe=a_cross_pe,
                                transformer_options=transformer_options,
                            ) * gate_out_a2v
                            
                            vx_compute = vx_compute + a2v_out
                            del a2v_out, vx_scaled, ax_scaled, vx_norm3
                            del scale_ca_video_a2v, shift_ca_video_a2v, scale_ca_video_v2a_chunk, shift_ca_video_v2a_chunk, gate_out_a2v
                        
                        a2v_output_chunks.append((vx_compute.to(storage_device, non_blocking=True), start, end, storage_device, storage_gpu_id))
                        del vx_compute, pe_chunk
                        torch.cuda.empty_cache()
                    
                    # Replace intermediate chunks with a2v output
                    intermediate_vx_chunks = a2v_output_chunks
                
                # v2a: Gather full video, add to audio
                if run_v2a:
                    # Gather full video temporarily
                    vx_full_for_v2a = torch.cat([c.to(compute_device) for c, _, _, _, _ in intermediate_vx_chunks], dim=1)
                    torch.cuda.synchronize(compute_gpu)
                    
                    vx_norm3_full = comfy.ldm.common_dit.rms_norm(vx_full_for_v2a)
                    
                    # Compute video ada values for FULL video (for v2a)
                    (
                        scale_ca_video_a2v_full, shift_ca_video_a2v_full,
                        scale_ca_video_v2a, shift_ca_video_v2a,
                        gate_out_a2v_full,
                    ) = block.get_av_ca_ada_values(
                        block.scale_shift_table_a2v_ca_video,
                        vx_full_for_v2a.shape[0],
                        av_ca_video_scale_shift,
                        av_ca_a2v_gate,
                    )
                    
                    # Get full video cross PE
                    v_cross_pe_chunks = []
                    for chunk_idx in range(num_chunks):
                        pe_chunk = get_pe_chunk(pe_storage, chunk_idx, compute_device, compute_gpu)
                        if pe_chunk is not None and isinstance(pe_chunk, (list, tuple)) and len(pe_chunk) > 0:
                            v_pe_pair = pe_chunk[0]
                            if isinstance(v_pe_pair, (list, tuple)) and len(v_pe_pair) > 1:
                                v_cross_pe_chunk = v_pe_pair[1]
                                if v_cross_pe_chunk is not None and isinstance(v_cross_pe_chunk, tuple):
                                    v_cross_pe_chunks.append(v_cross_pe_chunk)
                    
                    # Concatenate video cross PE if we have chunks
                    v_cross_pe_full = None
                    if len(v_cross_pe_chunks) > 0:
                        # Concatenate cos and sin separately
                        cos_list = [pe[0] for pe in v_cross_pe_chunks if pe[0] is not None]
                        sin_list = [pe[1] for pe in v_cross_pe_chunks if pe[1] is not None]
                        if cos_list and sin_list:
                            cos_full = torch.cat(cos_list, dim=2 if cos_list[0].dim() == 4 else 1)
                            sin_full = torch.cat(sin_list, dim=2 if sin_list[0].dim() == 4 else 1)
                            split = v_cross_pe_chunks[0][2] if len(v_cross_pe_chunks[0]) > 2 else False
                            v_cross_pe_full = (cos_full, sin_full, split)
                    
                    vx_scaled_v2a = vx_norm3_full * (1 + scale_ca_video_v2a) + shift_ca_video_v2a
                    ax_scaled_v2a = ax_norm3 * (1 + scale_ca_audio_v2a) + shift_ca_audio_v2a
                    
                    v2a_out = block.video_to_audio_attn(
                        ax_scaled_v2a,
                        context=vx_scaled_v2a,
                        pe=a_cross_pe,
                        k_pe=v_cross_pe_full,
                        transformer_options=transformer_options,
                    ) * gate_out_v2a
                    
                    ax_compute = ax_compute + v2a_out
                    
                    del vx_full_for_v2a, vx_norm3_full, vx_scaled_v2a, ax_scaled_v2a, v2a_out
                    del scale_ca_video_a2v_full, shift_ca_video_a2v_full, scale_ca_video_v2a, shift_ca_video_v2a, gate_out_a2v_full
                    del v_cross_pe_full, v_cross_pe_chunks
                    torch.cuda.empty_cache()
                
                del ax_norm3
                del scale_ca_audio_a2v, shift_ca_audio_a2v, scale_ca_audio_v2a, shift_ca_audio_v2a, gate_out_v2a
        
        # ============================================================
        # PART E: Video FFN for each chunk
        # ============================================================
        output_chunks = []
        
        for chunk_idx in range(num_chunks):
            vx_data, start, end, storage_device, storage_gpu_id = intermediate_vx_chunks[chunk_idx]
            _, _, _, shift_mlp, scale_mlp, gate_mlp = ada_storage[chunk_idx]
            
            vx_compute = vx_data.to(compute_device, non_blocking=True)
            scale_mlp_c = scale_mlp.to(compute_device, non_blocking=True)
            shift_mlp_c = shift_mlp.to(compute_device, non_blocking=True)
            gate_mlp_c = gate_mlp.to(compute_device, non_blocking=True)
            torch.cuda.synchronize(compute_gpu)
            
            y = comfy.ldm.common_dit.rms_norm(vx_compute) * (1 + scale_mlp_c) + shift_mlp_c
            vx_compute = vx_compute + block.ff(y) * gate_mlp_c
            del y, scale_mlp_c, shift_mlp_c, gate_mlp_c
            
            output_chunks.append((vx_compute.to(storage_device, non_blocking=True), start, end, storage_device, storage_gpu_id))
            del vx_compute
            torch.cuda.empty_cache()
        
        del ada_storage, intermediate_vx_chunks
        
        # ============================================================
        # PART F: Audio FFN
        # ============================================================
        if is_av_model and ax_compute is not None and ax_compute.numel() > 0:
            ashift_mlp, ascale_mlp, agate_mlp = block.get_ada_values(
                block.audio_scale_shift_table, batch_size, a_timestep, slice(3, None)
            )
            
            ax_scaled = comfy.ldm.common_dit.rms_norm(ax_compute) * (1 + ascale_mlp) + ashift_mlp
            ax_compute = ax_compute + block.audio_ff(ax_scaled) * agate_mlp
            del ax_scaled, ashift_mlp, ascale_mlp, agate_mlp
        
        # Sync and update vx_chunks
        for gpu_id in _config.storage_gpus:
            torch.cuda.synchronize(gpu_id)
        vx_chunks = output_chunks
    
    # ================================================================
    # STEP 6: Process output
    # ================================================================
    
    if _config.verbose >= 1:
        logger.info(f"[FORWARD] Processing output...")
    
    # Gather vx from all chunks
    vx_final = torch.cat([c.to(compute_device) for c, _, _, _, _ in vx_chunks], dim=1)
    torch.cuda.synchronize(compute_gpu)
    
    # Reconstruct x for _process_output
    # NOTE: ax_compute was updated throughout the block loop, so use it directly!
    if is_av_model:
        # ax_compute already has the processed audio from all transformer blocks
        # Don't reload from ax_storage - that would lose all the audio processing!
        x_final = [vx_final, ax_compute if ax_compute is not None else torch.empty(0, device=compute_device)]
    else:
        x_final = vx_final
    
    # Gather FULL embedded_timestep (not just chunk 0!)
    def gather_nested(storage, num_chunks):
        """Gather all chunks from nested storage structure."""
        if storage is None:
            return None
        
        # Check if this is a chunked tensor list
        if is_chunked_tensor_list(storage):
            # Gather all chunks
            gathered = torch.cat([c.to(compute_device) for c, _, _, _, _ in storage], dim=1)
            torch.cuda.synchronize(compute_gpu)
            return gathered
        
        elif isinstance(storage, (list, tuple)):
            # Recurse into nested structure
            results = [gather_nested(sub, num_chunks) for sub in storage]
            return type(storage)(results)
        
        elif isinstance(storage, torch.Tensor):
            # Not chunked, just move
            result = storage.to(compute_device, non_blocking=True)
            torch.cuda.synchronize(compute_gpu)
            return result
        
        else:
            return storage
    
    emb_ts_compute = gather_nested(emb_ts_storage, num_chunks)
    
    # Process output
    output = self._process_output(x_final, emb_ts_compute, keyframe_idxs, **merged_args)
    
    if _config.verbose >= 1:
        mem = torch.cuda.memory_allocated(compute_gpu) / 1024**2
        logger.info(f"[FORWARD] Complete. GPU{compute_gpu}: {mem:.1f}MB")
    
    return output


# ============================================================================
# Hook Installation
# ============================================================================

def install_forward_hook():
    global _original_forward, _forward_hook_installed
    
    if _forward_hook_installed:
        return True
    
    try:
        from comfy.ldm.lightricks.model import LTXBaseModel
        _original_forward = LTXBaseModel._forward
        LTXBaseModel._forward = chunked_forward
        _forward_hook_installed = True
        logger.info("[FORWARD] Hook installed on LTXBaseModel._forward")
        return True
    except Exception as e:
        logger.error(f"[FORWARD] Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def uninstall_forward_hook():
    global _original_forward, _forward_hook_installed
    
    if not _forward_hook_installed:
        return
    
    try:
        from comfy.ldm.lightricks.model import LTXBaseModel
        LTXBaseModel._forward = _original_forward
        _original_forward = None
        _forward_hook_installed = False
    except:
        pass


# ============================================================================
# Upsampler (unchanged)
# ============================================================================

def chunked_upsampler_forward(self, latent: torch.Tensor) -> torch.Tensor:
    global _config, _original_upsampler_forward
    
    if _config is None or not _config.enabled:
        return _original_upsampler_forward(self, latent)
    
    b, c, f, h, w = latent.shape
    chunk_size = _config.upsampler_temporal_chunk
    overlap = _config.upsampler_overlap
    
    if f <= chunk_size + overlap:
        return _original_upsampler_forward(self, latent)
    
    if _config.verbose >= 1:
        logger.info(f"[UPSAMPLER] Chunking {f} frames")
    
    try:
        return _chunked_upsampler_impl(self, latent, b, c, f, h, w, chunk_size, overlap)
    except Exception as e:
        logger.error(f"[UPSAMPLER] Error: {e}")
        torch.cuda.empty_cache()
        return _original_upsampler_forward(self, latent)


def _chunked_upsampler_impl(self, latent, b, c, f, h, w, chunk_size, overlap):
    global _config
    
    compute_device = _config.compute_device
    scale = getattr(self, 'spatial_scale', 2.0)
    out_h, out_w = int(h * scale), int(w * scale)
    
    step = chunk_size - overlap
    num_chunks = math.ceil((f - overlap) / step) if step > 0 else 1
    
    input_chunks = []
    for i in range(num_chunks):
        start = i * step
        end = min(start + chunk_size, f)
        if i == num_chunks - 1:
            end = f
            start = max(0, end - chunk_size)
        
        dev = _config.get_storage_device(i)
        gid = _config.get_storage_gpu_id(i)
        chunk = latent[:, :, start:end, :, :].to(dev, non_blocking=True)
        torch.cuda.synchronize(gid)
        input_chunks.append((chunk, start, end, dev, gid))
    
    del latent
    torch.cuda.empty_cache()
    
    output_chunks = []
    for i, (chunk, start, end, dev, gid) in enumerate(input_chunks):
        chunk_c = chunk.to(compute_device, non_blocking=True)
        torch.cuda.synchronize(_config.compute_gpu)
        del chunk
        
        out = _original_upsampler_forward(self, chunk_c)
        del chunk_c
        torch.cuda.empty_cache()
        
        out_s = out.to(dev, non_blocking=True)
        torch.cuda.synchronize(gid)
        del out
        torch.cuda.empty_cache()
        
        output_chunks.append((out_s, start, end, dev, gid))
    
    del input_chunks
    
    first_storage = _config.get_storage_device(0)
    first_gid = _config.get_storage_gpu_id(0)
    output = torch.empty((b, c, f, out_h, out_w), dtype=output_chunks[0][0].dtype, device=first_storage)
    
    for i, (chunk, start, end, dev, gid) in enumerate(output_chunks):
        if dev != first_storage:
            chunk = chunk.to(first_storage, non_blocking=True)
            torch.cuda.synchronize(first_gid)
        
        if i == 0:
            output[:, :, start:end, :, :] = chunk
        else:
            blend_start, blend_end = start, start + overlap
            if blend_end > blend_start and overlap > 0:
                w = torch.linspace(0, 1, overlap, device=first_storage).view(1, 1, -1, 1, 1)
                output[:, :, blend_start:blend_end, :, :] = (
                    output[:, :, blend_start:blend_end, :, :] * (1 - w) +
                    chunk[:, :, :overlap, :, :] * w
                )
            
            if end > blend_end:
                output[:, :, blend_end:end, :, :] = chunk[:, :, overlap:overlap+(end-blend_end), :, :]
        del chunk
    
    del output_chunks
    torch.cuda.empty_cache()
    
    result = output.to(compute_device, non_blocking=True)
    torch.cuda.synchronize(_config.compute_gpu)
    del output
    torch.cuda.empty_cache()
    
    return result


def install_upsampler_hook():
    global _original_upsampler_forward, _upsampler_hook_installed
    
    if _upsampler_hook_installed:
        return True
    
    try:
        from comfy.ldm.lightricks.latent_upsampler import LatentUpsampler
        _original_upsampler_forward = LatentUpsampler.forward
        LatentUpsampler.forward = chunked_upsampler_forward
        _upsampler_hook_installed = True
        logger.info("[UPSAMPLER] Hook installed")
        return True
    except:
        return False


def uninstall_upsampler_hook():
    global _original_upsampler_forward, _upsampler_hook_installed
    if not _upsampler_hook_installed:
        return
    try:
        from comfy.ldm.lightricks.latent_upsampler import LatentUpsampler
        LatentUpsampler.forward = _original_upsampler_forward
        _original_upsampler_forward = None
        _upsampler_hook_installed = False
    except:
        pass


# ============================================================================
# ComfyUI Node
# ============================================================================

class LTXMultiGPUChunkedNode:
    """LTX-2 Multi-GPU Chunked v6 - Process once, chunk immediately!"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "storage_gpus": ("STRING", {"default": "1,2,3"}),
                "transformer_chunk_size": ("INT", {"default": 8000, "min": 2000, "max": 30000, "step": 1000}),
                "upsampler_temporal_chunk": ("INT", {"default": 8, "min": 2, "max": 32}),
                "upsampler_overlap": ("INT", {"default": 2, "min": 0, "max": 8}),
                "min_seq_length": ("INT", {"default": 60000, "min": 1000, "max": 500000, "step": 1000}),
                "verbose": ("INT", {"default": 1, "min": 0, "max": 2}),
            },
        }
    
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "setup"
    CATEGORY = "multigpu"
    
    def setup(self, model, storage_gpus, transformer_chunk_size, upsampler_temporal_chunk, upsampler_overlap, min_seq_length, verbose):
        global _config
        
        lines = ["LTX-2 Multi-GPU v6: Process once, chunk immediately!"]
        lines.append("=" * 50)
        
        try:
            gpu_list = [int(g.strip()) for g in storage_gpus.split(',')]
            
            num_gpus = torch.cuda.device_count()
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                lines.append(f"  cuda:{i} = {props.name} ({props.total_memory/1024**3:.1f} GB)")
            
            _config = MultiGPUConfig(
                storage_gpus=gpu_list,
                compute_gpu=0,
                transformer_chunk_size=transformer_chunk_size,
                upsampler_temporal_chunk=upsampler_temporal_chunk,
                upsampler_overlap=upsampler_overlap,
                min_seq_length=min_seq_length,
                enabled=True,
                verbose=verbose,
            )
            
            hooks = []
            if install_forward_hook():
                hooks.append("Forward")
            if install_upsampler_hook():
                hooks.append("Upsampler")
            
            lines.append(f"\n✓ Hooks: {', '.join(hooks)}")
            lines.append(f"Storage GPUs: {gpu_list}")
            lines.append(f"\nv6 Strategy:")
            lines.append(f"  1. _process_input -> immediately chunk & offload")
            lines.append(f"  2. _prepare_timestep -> immediately chunk & offload")
            lines.append(f"  3. _prepare_PE -> immediately chunk & offload")
            lines.append(f"  4. Process blocks with pre-chunked data")
            
        except Exception as e:
            lines.append(f"ERROR: {e}")
        
        return (model, "\n".join(lines))


class LTXMultiGPUChunkedDisableNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"any_input": ("*",)}}
    
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "disable"
    CATEGORY = "multigpu"
    
    def disable(self, any_input):
        global _config
        uninstall_forward_hook()
        uninstall_upsampler_hook()
        _config = None
        return (any_input,)


NODE_CLASS_MAPPINGS = {
    "LTXMultiGPUChunkedNode": LTXMultiGPUChunkedNode,
    "LTXMultiGPUChunkedDisableNode": LTXMultiGPUChunkedDisableNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXMultiGPUChunkedNode": "LTX Multi-GPU Chunked v6",
    "LTXMultiGPUChunkedDisableNode": "LTX Multi-GPU Disable",
}
