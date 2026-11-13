"""
Compressor Package
==================
Modular architecture for lossless dictionary compression down to < 88 KiB.

Subpackages:
- morphology: Morphological transform operators, Exact Cover DP solver, candidate dataset generator.
- models: 3-order PAQ bit-FSM, causal context router, multi-context neural logistic trainer.
- encoding: Canonical Huffman tree, static rANS, DP-optimal LZSS, safe non-bidi UTF-8 embedding.
- codegen: Standalone C++20 (9ms) and Python (0.35s) solver code generators.
"""

from compressor.morphology.operators import MorphologicalRules
from compressor.morphology.cover_dp import ExactCoverDP
from compressor.morphology.dataset import DatasetBuilder
from compressor.models.paq_fsm import PAQBitFSM
from compressor.models.context_router import ContextRouter
from compressor.models.neural_trainer import NeuralLogisticTrainer
from compressor.models.zero_paq import ZeroPAQModel
from compressor.encoding.huffman import CanonicalHuffman
from compressor.encoding.rans import RangeANS
from compressor.encoding.lzss import DPOptimalLZSS
from compressor.encoding.utf8_rans import UTF8Encoder
from compressor.codegen.cpp_generator import CppSolverGenerator
from compressor.codegen.py_generator import PySolverGenerator
from compressor.pipeline import DictionaryCompressionPipeline

__version__ = "2.0.0"
__all__ = [
    "MorphologicalRules",
    "ExactCoverDP",
    "DatasetBuilder",
    "PAQBitFSM",
    "ContextRouter",
    "NeuralLogisticTrainer",
    "ZeroPAQModel",
    "CanonicalHuffman",
    "RangeANS",
    "DPOptimalLZSS",
    "UTF8Encoder",
    "CppSolverGenerator",
    "PySolverGenerator",
    "DictionaryCompressionPipeline",
]
