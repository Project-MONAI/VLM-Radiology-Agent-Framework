# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Device utilities for GPU-agnostic code supporting both NVIDIA CUDA and AMD ROCm.

This module provides helper functions to detect and manage GPU devices
in a way that works seamlessly with both NVIDIA and AMD hardware.
"""

import os
import logging
from typing import Optional, Union, List

import torch

logger = logging.getLogger(__name__)


def is_rocm_available() -> bool:
    """Check if ROCm (AMD GPU) is available."""
    try:
        # Check if PyTorch was built with ROCm
        return hasattr(torch.version, 'hip') and torch.version.hip is not None
    except Exception:
        return False


def is_cuda_available() -> bool:
    """Check if CUDA (NVIDIA GPU) is available."""
    return torch.cuda.is_available() and not is_rocm_available()


def get_gpu_backend() -> str:
    """
    Get the GPU backend being used.
    
    Returns:
        str: 'rocm' for AMD, 'cuda' for NVIDIA, 'cpu' if no GPU
    """
    if not torch.cuda.is_available():
        return 'cpu'
    if is_rocm_available():
        return 'rocm'
    return 'cuda'


def get_device(device_id: Optional[int] = None) -> torch.device:
    """
    Get the appropriate torch device.
    
    Args:
        device_id: Optional GPU device ID. If None, uses default GPU or CPU.
    
    Returns:
        torch.device: The device to use for tensors and models.
    """
    if not torch.cuda.is_available():
        logger.info("No GPU available, using CPU")
        return torch.device('cpu')
    
    if device_id is not None:
        device = torch.device(f'cuda:{device_id}')
    else:
        device = torch.device('cuda')
    
    backend = get_gpu_backend()
    logger.info(f"Using {backend.upper()} device: {device}")
    return device


def get_device_name(device_id: int = 0) -> str:
    """Get the name of the GPU device."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(device_id)
    return "CPU"


def get_device_count() -> int:
    """Get the number of available GPU devices."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def get_visible_devices_env_var() -> str:
    """
    Get the environment variable name for visible devices.
    
    Returns:
        str: 'HIP_VISIBLE_DEVICES' for ROCm, 'CUDA_VISIBLE_DEVICES' for CUDA
    """
    if is_rocm_available():
        return 'HIP_VISIBLE_DEVICES'
    return 'CUDA_VISIBLE_DEVICES'


def set_visible_devices(device_ids: Union[int, List[int], str]) -> None:
    """
    Set visible GPU devices.
    
    Args:
        device_ids: Device ID(s) to make visible. Can be int, list of ints, or comma-separated string.
    """
    if isinstance(device_ids, int):
        device_str = str(device_ids)
    elif isinstance(device_ids, list):
        device_str = ','.join(map(str, device_ids))
    else:
        device_str = str(device_ids)
    
    env_var = get_visible_devices_env_var()
    os.environ[env_var] = device_str
    logger.info(f"Set {env_var}={device_str}")


def to_device(tensor_or_model, device: Optional[torch.device] = None):
    """
    Move a tensor or model to the specified device.
    
    Args:
        tensor_or_model: PyTorch tensor or model to move
        device: Target device. If None, uses default GPU or CPU.
    
    Returns:
        The tensor or model on the target device
    """
    if device is None:
        device = get_device()
    return tensor_or_model.to(device)


def get_gpu_memory_info(device_id: int = 0) -> dict:
    """
    Get GPU memory information.
    
    Args:
        device_id: GPU device ID
    
    Returns:
        dict: Memory info with 'total', 'allocated', 'cached', 'free' keys (in bytes)
    """
    if not torch.cuda.is_available():
        return {'total': 0, 'allocated': 0, 'cached': 0, 'free': 0}
    
    torch.cuda.set_device(device_id)
    total = torch.cuda.get_device_properties(device_id).total_memory
    allocated = torch.cuda.memory_allocated(device_id)
    cached = torch.cuda.memory_reserved(device_id)
    free = total - allocated
    
    return {
        'total': total,
        'allocated': allocated,
        'cached': cached,
        'free': free
    }


def empty_cache() -> None:
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("GPU cache cleared")


def synchronize(device_id: Optional[int] = None) -> None:
    """Synchronize GPU operations."""
    if torch.cuda.is_available():
        if device_id is not None:
            torch.cuda.synchronize(device_id)
        else:
            torch.cuda.synchronize()


def print_gpu_info() -> None:
    """Print GPU information for debugging."""
    backend = get_gpu_backend()
    print(f"GPU Backend: {backend.upper()}")
    print(f"PyTorch version: {torch.__version__}")
    
    if is_rocm_available():
        print(f"ROCm/HIP version: {torch.version.hip}")
    elif torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
    
    device_count = get_device_count()
    print(f"Number of GPUs: {device_count}")
    
    for i in range(device_count):
        name = get_device_name(i)
        mem_info = get_gpu_memory_info(i)
        total_gb = mem_info['total'] / (1024**3)
        print(f"  GPU {i}: {name} ({total_gb:.1f} GB)")


if __name__ == "__main__":
    print_gpu_info()
