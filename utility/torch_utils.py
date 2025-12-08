from __future__ import annotations

import os
from typing import Optional

import torch


def get_default_device(env_var: str = "TORCH_DEVICE") -> torch.device:
    """
    Pick the best available torch device with a macOS Metal-first policy.

    Order of preference (unless overridden via env var):
    1. MPS if built and available (Metal on Apple Silicon).
    2. CUDA if available.
    3. CPU fallback.
    """

    override: Optional[str] = os.getenv(env_var)
    if override:
        try:
            return torch.device(override)
        except (TypeError, ValueError):
            # Invalid override; fall back to automatic selection.
            pass

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
