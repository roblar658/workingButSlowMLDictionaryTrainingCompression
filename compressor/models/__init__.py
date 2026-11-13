"""
Models Subpackage
=================
Exports statistical, machine-learning, and context routing components:
- PAQBitFSM: Bit-level finite state machine
- ContextRouter: Causal context routing and feature hashing
- NeuralLogisticTrainer: Adam optimization, int8 quantization, and range encoding
"""

from compressor.models.paq_fsm import PAQBitFSM
from compressor.models.context_router import ContextRouter
from compressor.models.neural_trainer import NeuralLogisticTrainer
from compressor.models.zero_paq import ZeroPAQModel

__all__ = ["PAQBitFSM", "ContextRouter", "NeuralLogisticTrainer", "ZeroPAQModel"]
