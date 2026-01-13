"""
Tensor Parallel V2 + Chunked FFN for ComfyUI
============================================

True parallel attention + chunked FFN for maximum memory efficiency.

Strategies:
1. Attention: Split by heads across GPUs (true parallelism)
2. FFN: Chunked execution (reduces peak intermediate memory)

Why chunked FFN instead of parallel?
- ComfyUI offloads weights to CPU in lowvram mode
- Can't deepcopy offloaded tensors to multiple GPUs
- Chunking reduces peak memory without needing weight copies

Memory savings example (700 frames, 6 GPUs):
  Attention: 120GB → ~20GB per GPU (parallel)
  FFN peak: 3.3GB → ~0.55GB (chunked)

Credits:
- Tensor parallelism: Megatron-LM (NVIDIA)
- Implementation: Claude (Anthropic) with Aaron
"""

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TensorParallelV2")

__version__ = "0.3.1"  # Chunked FFN (works with offloading!)

def _init():
    import torch
    logger.info("=" * 60)
    logger.info("Tensor Parallel V2 + Chunked FFN v%s", __version__)
    logger.info("=" * 60)
    
    num_gpus = torch.cuda.device_count()
    logger.info(f"GPUs available: {num_gpus}")
    
    if num_gpus >= 2:
        logger.info("Ready!")
        logger.info("")
        logger.info("This version:")
        logger.info("  - Attention: parallel across GPUs (by heads)")
        logger.info("  - FFN: chunked (reduces peak memory)")
        logger.info("")
        logger.info("Works with ComfyUI's lowvram/offloading!")
    else:
        logger.warning("Need 2+ GPUs for attention parallelism")
    
    logger.info("=" * 60)

_init()

from .tensor_parallel_v2 import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
