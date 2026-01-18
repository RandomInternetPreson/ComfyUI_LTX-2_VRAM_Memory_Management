"""
Sequence Chunked Block - ComfyUI Custom Node
=============================================

Distributes transformer block processing across multiple GPUs by chunking
the sequence dimension. This allows generating much longer videos without
running out of memory on the compute GPU.

Installation:
    Place this folder in ComfyUI/custom_nodes/

Usage:
    Add the "Sequence Chunked Block (Multi-GPU)" node before your sampler.
    Set storage_gpu to a GPU other than 0 (e.g., 1).
"""

from .sequence_chunked_block import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

WEB_DIRECTORY = None
