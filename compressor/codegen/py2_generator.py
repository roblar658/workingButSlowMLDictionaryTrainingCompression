"""
Python 2 Solver Generator Module
================================
Emits a Python 2.7 / 3.x dual-compatible standalone C-in-Python solver
(`solve_py2.py` < 88 KiB) backed by the shared NN+PAQ C decoder compiled via gcc,
with a Python 2/3 pure-Python fallback when gcc is unavailable.
"""

from typing import Any, Dict, Tuple

from compressor.codegen.py_generator import PySolverGenerator


class Py2SolverGenerator:
    """Emits standalone Python 2.7-compatible dictionary decompressor."""

    @classmethod
    def generate(cls, a: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Generate standalone solve_py2.py (< 88 KiB) with C decoder + Py2/Py3 fallback."""
        return PySolverGenerator._generate_solver(a, py2=True)
