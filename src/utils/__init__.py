"""Utility functions for the application."""

# Load .env file first before other imports
from utils import env  # noqa: F401

from utils.device import get_device, is_mps_available, is_cuda_available

__all__ = ["get_device", "is_mps_available", "is_cuda_available"]

