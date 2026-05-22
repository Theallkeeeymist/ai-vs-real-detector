"""
GPU memory management and utilities.
Handles CUDA memory optimization and CPU fallback.
"""

import torch
import os
from ai_detector.logging.logger import logger


def setup_gpu_memory():
    """
    Configure GPU memory for optimal usage.
    Must be called BEFORE creating models.
    """
    try:
        # Enable memory-efficient algorithms
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        # Set PyTorch memory allocation config
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Clear any existing cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            logger.info("✓ GPU memory cleared and optimized")
        
    except Exception as e:
        logger.warning(f"Could not optimize GPU memory: {e}")


def get_device():
    """
    Get the best available device (GPU with fallback to CPU).
    """
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"✓ Using GPU: {gpu_name} ({gpu_memory_gb:.2f} GB)")
    else:
        device = "cpu"
        logger.warning("⚠️ CUDA not available, falling back to CPU (training will be slow)")
    
    return device


def get_gpu_memory_info():
    """
    Get current GPU memory usage.
    """
    if not torch.cuda.is_available():
        return None
    
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    free = total - allocated
    
    return {
        "total_gb": total,
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "free_gb": free,
        "percent_used": (allocated / total) * 100
    }


def log_gpu_memory_status():
    """
    Log current GPU memory usage.
    """
    info = get_gpu_memory_info()
    if info is None:
        logger.info("GPU not available")
        return
    
    logger.info(f"GPU Memory - Total: {info['total_gb']:.2f}GB | "
                f"Allocated: {info['allocated_gb']:.2f}GB | "
                f"Free: {info['free_gb']:.2f}GB | "
                f"Used: {info['percent_used']:.1f}%")


def clear_gpu_cache():
    """
    Clear GPU cache to free up memory.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        logger.info("✓ GPU cache cleared")


class GPUMemoryMonitor:
    """
    Context manager to monitor GPU memory during training.
    """
    
    def __init__(self, step_name: str = "Step"):
        self.step_name = step_name
        self.start_allocated = 0
        self.start_reserved = 0
    
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.start_allocated = torch.cuda.memory_allocated() / 1e9
            self.start_reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f"[{self.step_name}] Starting GPU memory: "
                       f"{self.start_allocated:.2f}GB allocated")
        return self
    
    def __exit__(self, *args):
        if torch.cuda.is_available():
            end_allocated = torch.cuda.memory_allocated() / 1e9
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            
            logger.info(f"[{self.step_name}] Peak GPU memory: {peak_memory:.2f}GB")
            logger.info(f"[{self.step_name}] End GPU memory: {end_allocated:.2f}GB")