"""
Neurobiomorphic AI: Production-Grade Biologically-Inspired Artificial Intelligence

A comprehensive framework for building neurobiomorphic AI systems with:
- Advanced neural plasticity mechanisms
- Causal reasoning engines  
- Uncertainty quantification
- Language reasoning capabilities
- Hierarchical reinforcement learning
- Multi-modal learning systems
"""

__version__ = "0.1.0"
__author__ = "Neurobiomorphic AI Team"
__email__ = "team@neurobiomorphic-ai.com"

# Core module imports for easy access
from . import config
from . import neural_plasticity
from . import reasoning
from . import language_reasoning
from . import hierarchical_agent
from . import reinforcement_learning
from . import hybrid_learning
from . import monitoring
from . import utils

__all__ = [
    "config",
    "neural_plasticity", 
    "reasoning",
    "language_reasoning",
    "hierarchical_agent",
    "reinforcement_learning", 
    "hybrid_learning",
    "monitoring",
    "utils",
]