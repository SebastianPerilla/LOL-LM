"""Device detection utilities for optimizing model inference."""

import platform
from typing import Literal

DeviceType = Literal["cuda", "mps", "cpu"]


def is_cuda_available() -> bool:
    """Check if CUDA (NVIDIA GPU) is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def is_mps_available() -> bool:
    """Check if MPS (Apple Silicon) is available."""
    try:
        import torch
        # MPS is available on macOS 12.3+ with Apple Silicon
        if platform.system() == "Darwin" and platform.processor() == "arm":
            return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        return False
    except ImportError:
        return False


def get_device(device_preference: DeviceType | None = None) -> DeviceType:
    """
    Automatically detect and return the best available device for model inference.
    
    Priority order:
    1. CUDA (if available and requested)
    2. MPS (Apple Silicon - if available)
    3. CPU (fallback)
    
    Args:
        device_preference: Optional device preference. If None, auto-detects best device.
                          If specified, checks if available and falls back if not.
    
    Returns:
        Device string: "cuda", "mps", or "cpu"
    """
    # If preference is specified, try to use it
    if device_preference:
        if device_preference == "cuda" and is_cuda_available():
            return "cuda"
        elif device_preference == "mps" and is_mps_available():
            return "mps"
        elif device_preference == "cpu":
            return "cpu"
        # If preference not available, fall through to auto-detection
    
    # Auto-detect best available device
    if is_cuda_available():
        return "cuda"
    elif is_mps_available():
        return "mps"
    else:
        return "cpu"


def get_device_info() -> dict[str, str | bool]:
    """
    Get information about available devices.
    
    Returns:
        Dictionary with device information
    """
    return {
        "system": platform.system(),
        "processor": platform.processor(),
        "cuda_available": is_cuda_available(),
        "mps_available": is_mps_available(),
        "recommended_device": get_device(),
    }

