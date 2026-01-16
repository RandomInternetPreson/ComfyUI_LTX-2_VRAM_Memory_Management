"""
ComfyUI LTX-2 Sequence Parallel Node (V5)
==========================================

Implements sequence parallelism with ring attention for very long sequences.

This approach splits the sequence (tokens) across GPUs rather than splitting
attention heads. Each GPU holds 1/N of the sequence and uses ring attention
to compute full self-attention.

Best for: I2V workflows with 600K+ tokens after spatial upscaling.
"""

from .tensor_parallel_v5 import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
