"""
Tensor Parallel V3 - Safe Chunking Mode
=======================================

This version ONLY does FFN chunking - no multi-GPU operations.
Guaranteed compatible with ComfyUI's offloading mode AND LoRA patches.

For multi-GPU attention parallelism, use V2 with offloading DISABLED.

V3.1 Changes:
- Fixed LoRA compatibility by using forward method patching instead of module replacement
- Module hierarchy is preserved so LoRA weight patches can still find their targets
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Callable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TensorParallelV3")


def create_chunked_forward(original_forward: Callable, num_chunks: int) -> Callable:
    """
    Create a chunked forward function that wraps the original.
    This preserves the original module's interface while adding chunking.
    """
    def chunked_forward(x):
        # Handle different input shapes
        if x.dim() == 3:
            batch_size, seq_len, hidden = x.shape
        elif x.dim() == 2:
            # Some FFNs might get 2D input
            return original_forward(x)
        else:
            return original_forward(x)
        
        # Don't chunk if sequence is short
        if seq_len < num_chunks * 100:
            return original_forward(x)
        
        chunk_size = max(1, (seq_len + num_chunks - 1) // num_chunks)
        
        outputs = []
        for i in range(0, seq_len, chunk_size):
            end_idx = min(i + chunk_size, seq_len)
            chunk = x[:, i:end_idx, :]
            out = original_forward(chunk)
            outputs.append(out)
        
        return torch.cat(outputs, dim=1)
    
    return chunked_forward


class TensorParallelV3Pipeline:
    """
    Safe FFN chunking pipeline that preserves module hierarchy for LoRA compatibility.
    
    Instead of replacing modules (which breaks LoRA weight paths), this version
    patches the forward method directly on the existing modules.
    """
    
    _instances: Dict[int, 'TensorParallelV3Pipeline'] = {}
    
    def __init__(self, model: nn.Module, ffn_chunks: int = 8, verbose: int = 1):
        self.model = model
        self.model_id = id(model)
        self.ffn_chunks = ffn_chunks
        self.verbose = verbose
        
        # Store original forward methods for cleanup
        self.original_forwards: Dict[str, Callable] = {}
        self.patched_modules: Dict[str, nn.Module] = {}
        
        TensorParallelV3Pipeline._instances[self.model_id] = self
    
    @classmethod
    def get_instance(cls, model: nn.Module) -> Optional['TensorParallelV3Pipeline']:
        return cls._instances.get(id(model))
    
    def setup(self) -> Dict[str, Any]:
        info = {"ffn_wrapped": 0, "ffn_found": 0}
        
        if self.verbose >= 1:
            logger.info(f"TensorParallelV3: Setting up FFN chunking...")
            logger.info(f"  Chunks: {self.ffn_chunks}")
        
        # Find FFN modules
        ffn_modules = {}
        for name, module in self.model.named_modules():
            if hasattr(module, 'net') and isinstance(module.net, nn.Sequential):
                if name.endswith('.ff') or 'ff' in name.split('.')[-1]:
                    ffn_modules[name] = module
        
        info["ffn_found"] = len(ffn_modules)
        
        if self.verbose >= 1:
            logger.info(f"  Found {len(ffn_modules)} FFN modules")
        
        for name, module in ffn_modules.items():
            try:
                # Store the original forward method
                # Use __func__ to get the unbound method if it's a bound method
                if hasattr(module.forward, '__func__'):
                    self.original_forwards[name] = module.forward.__func__
                else:
                    self.original_forwards[name] = module.forward
                
                # Create the chunked forward and bind it to the module
                chunked_fn = create_chunked_forward(module.forward, self.ffn_chunks)
                
                # Patch the forward method directly on the module instance
                module.forward = chunked_fn
                
                self.patched_modules[name] = module
                info["ffn_wrapped"] += 1
                
            except Exception as e:
                if self.verbose >= 1:
                    logger.warning(f"  Could not patch {name}: {e}")
        
        if self.verbose >= 1:
            logger.info(f"  Wrapped {info['ffn_wrapped']} FFN modules")
        
        return info
    
    def cleanup(self):
        """Restore original forward methods."""
        for name, module in self.patched_modules.items():
            if name in self.original_forwards:
                original = self.original_forwards[name]
                # Rebind the original forward method
                if callable(original):
                    # If it was an unbound method, bind it back
                    import types
                    if isinstance(original, types.FunctionType):
                        module.forward = types.MethodType(original, module)
                    else:
                        module.forward = original
        
        self.original_forwards.clear()
        self.patched_modules.clear()
        TensorParallelV3Pipeline._instances.pop(self.model_id, None)


class TensorParallelV3Node:
    """
    Safe FFN chunking - works with ALL modes including offloading AND LoRA.
    
    This node reduces peak memory by processing FFN layers in chunks.
    Place this node AFTER any LoRA nodes for best compatibility.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "ffn_chunks": ("INT", {"default": 8, "min": 2, "max": 32}),
                "verbose": ("INT", {"default": 1, "min": 0, "max": 2}),
            },
        }
    
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "setup"
    CATEGORY = "multigpu/experimental"
    
    def setup(self, model, ffn_chunks, verbose):
        lines = ["Tensor Parallel V3 - Safe FFN Chunking"]
        lines.append("=" * 50)
        
        try:
            target = model
            if hasattr(model, 'model'):
                target = model.model
                if hasattr(target, 'diffusion_model'):
                    target = target.diffusion_model
            
            existing = TensorParallelV3Pipeline.get_instance(target)
            if existing:
                lines.append("Already active - cleaning up old instance first")
                existing.cleanup()
            
            pipeline = TensorParallelV3Pipeline(
                model=target,
                ffn_chunks=ffn_chunks,
                verbose=verbose,
            )
            
            info = pipeline.setup()
            
            lines.append(f"FFN modules found: {info['ffn_found']}")
            lines.append(f"FFN modules wrapped: {info['ffn_wrapped']}")
            lines.append(f"Chunks: {ffn_chunks}")
            lines.append("")
            lines.append("This reduces FFN peak memory.")
            lines.append("Safe with offloading mode AND LoRA.")
            lines.append("")
            lines.append("Note: Place AFTER LoRA nodes for best results.")
            
        except Exception as e:
            lines.append(f"ERROR: {e}")
            import traceback
            lines.append(traceback.format_exc())
        
        return (model, "\n".join(lines))


NODE_CLASS_MAPPINGS = {
    "TensorParallelV3Node": TensorParallelV3Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorParallelV3Node": "Tensor Parallel V3 (Safe FFN Chunking)",
}
