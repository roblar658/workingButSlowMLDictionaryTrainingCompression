#!/usr/bin/env python3
"""
ML Dictionary Training Compressor — Standalone Python 2.7/3.x Solver Builder
=============================================================================
Builds the compressed neural-PAQ standalone script `solve_py2.py` (< 88 KiB)
compatible with Python 2.7 and Python 3.x:
1. Shared ML + PAQ-FSM + Huffman + rANS compression pipeline
2. 122-char non-bidi & no-Ctrl-Z UTF-8 base-2305806 rANS encoding
3. Headerless C99 NN+PAQ shared-library decoder + Py2/Py3 pure-Python fallback

Usage:
    python mlDictionaryTrainingCompressionPy2.py [dict.txt] [solve_py2.py] [ml_dataset.pkl]
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time

from compressor.codegen.py2_generator import Py2SolverGenerator
from compressor.pipeline import DictionaryCompressionPipeline


def build_py2_solver(
    dict_path: str = "dict.txt",
    out_py: str = "solve_py2.py",
    dataset_path: str = "ml_dataset.pkl",
    epochs: int = 500,
):
    """Run the compression pipeline and emit + verify `solve_py2.py`."""
    t0 = time.time()
    pkl = dataset_path if (dataset_path and os.path.exists(dataset_path)) else None
    art = DictionaryCompressionPipeline.build(
        dict_path=dict_path,
        dataset_path=pkl,
        epochs=epochs,
    )

    py_src, info = Py2SolverGenerator.generate(art)
    py_bytes = py_src.encode("utf-8")
    with open(out_py, "wb") as f:
        f.write(py_bytes)

    total_sz = len(py_bytes)
    assert total_sz < 90112, f"Size limit exceeded: {total_sz} >= 90112 bytes"
    assert 26 not in py_bytes, "Unexpected 0x1A byte in solve_py2.py"

    print(
        f"Stats: blob={info['blob_len']} c_src={info['c_src_len']} "
        f"c_lz={info['c_lz_len']} py_fb={info['py_lz_len']} u8={info['u8_len']}"
    )
    print(f"Total {out_py}: {total_sz} bytes ({total_sz / 1024:.2f} KiB, headroom={90112 - total_sz} B)")

    if os.path.exists(dict_path):
        expected = open(dict_path, "rb").read()
        with tempfile.TemporaryDirectory() as td:
            if shutil.which("gcc"):
                dll_p = os.path.join(td, "check_py2." + ("dll" if os.name == "nt" else "so"))
                cmd = ["gcc", "-O0", "-Wall", "-Wextra", "-Wbidi-chars=any", "-shared"]
                if os.name != "nt":
                    cmd.append("-fPIC")
                cmd.extend(["-x", "c", "-", "-o", dll_p])
                c_res = subprocess.run(cmd, input=info["c_src"], capture_output=True, text=True)
                assert c_res.returncode == 0, c_res.stderr
                assert not c_res.stderr.strip(), f"gcc warnings: {c_res.stderr}"

                r_gcc = subprocess.run([sys.executable, out_py], capture_output=True)
                assert r_gcc.returncode == 0 and r_gcc.stdout == expected, "Python (gcc) mismatch"

            env_fb = os.environ.copy()
            env_fb["PATH"] = ""
            r_fb = subprocess.run([sys.executable, out_py], capture_output=True, env=env_fb)
            assert r_fb.returncode == 0 and r_fb.stdout == expected, "Pure-Python fallback mismatch"

            py2_bin = shutil.which("python2")
            if py2_bin:
                r_py2 = subprocess.run([py2_bin, out_py], capture_output=True)
                assert r_py2.returncode == 0 and r_py2.stdout == expected, "python2 mismatch"

        print(f"Verified exact match ({len(expected)} bytes) in {time.time() - t0:.2f}s")

    return py_src, info, total_sz


if __name__ == "__main__":
    d_path = sys.argv[1] if len(sys.argv) > 1 else "dict.txt"
    o_path = sys.argv[2] if len(sys.argv) > 2 else "solve_py2.py"
    p_path = sys.argv[3] if len(sys.argv) > 3 else "ml_dataset.pkl"
    build_py2_solver(d_path, o_path, p_path)
