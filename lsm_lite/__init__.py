"""
LSM Lite - A lightweight implementation of Liquid State Machine for conversational AI.

This package provides a simplified, maintainable implementation of LSM with:
- Custom tokenizers with multiple backends
- Sparse sine-activated reservoir computing
- 2D/3D CNN support for spatial-temporal processing
- HuggingFace dataset integration
- Model persistence and text generation
"""

from .api import LSMLite
from .utils.config import LSMConfig

__version__ = "0.1.0"
__all__ = ["LSMLite", "LSMConfig"]
