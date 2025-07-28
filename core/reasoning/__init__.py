"""
Reasoning Module - DSPy-based Hybrid Reasoning System
Advanced query understanding and response optimization for LLMind
"""

from .dspy_wrapper import DSPyWrapper
from .hybrid_manager import HybridStackManager
from .strategies import *

__all__ = [
    'DSPyWrapper',
    'HybridStackManager',
] 