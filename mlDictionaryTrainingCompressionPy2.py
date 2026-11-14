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

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from compressor.codegen.py2_generator import Py2SolverGenerator
from compressor.pipeline import DictionaryCompressionPipeline

DEFAULT_DICT = os.path.join(HERE, "dict.txt")
DEFAULT_OUT = os.path.join(HERE, "solve_py2.py")
DEFAULT_DATASET = os.path.join(HERE, "ml_dataset.pkl")


def _run_and_verify(cmd, norm_ref: bytes, label: str, env=None) -> None:
    """Execute a generated solver command and verify canonical dictionary output."""
    proc = subprocess.run(cmd, capture_output=True, env=env)
    stderr_txt = proc.stderr.decode("utf-8", "replace").strip()
    assert proc.returncode == 0, f"{label} failed (rc={proc.returncode}): {stderr_txt}"
    out = proc.stdout.replace(b"\r\n", b"\n")
    assert out == norm_ref, (
        f"{label} mismatch (out={len(out)} B vs expected={len(norm_ref)} B; "
        "if you edited dict.txt, re-run save_ml_dataset.py)"
    )


def build_py2_solver(
    dict_path: str = DEFAULT_DICT,
    out_py: str = DEFAULT_OUT,
    dataset_path: str = DEFAULT_DATASET,
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
    assert 26 not in py_bytes, "Unexpected 0x1A byte in solve_py2.py"

    print(
        f"Stats: blob={info['blob_len']} c_src={info['c_src_len']} "
        f"c_lz={info['c_lz_len']} py_fb={info['py_lz_len']} u8={info['u8_len']}"
    )
    print(f"Total {out_py}: {total_sz} bytes ({total_sz / 1024:.2f} KiB, headroom={90112 - total_sz:+d} B)")
    if total_sz >= 90112:
        print(f"WARNING: Size {total_sz} >= 90112 B (88 KiB limit exceeded for this dictionary)")

    if os.path.exists(dict_path):
        with open(dict_path, "rb") as f:
            raw_bytes = f.read()
        raw_ref = raw_bytes.replace(b"\r\n", b"\n")
        norm_words = sorted(
            dict.fromkeys(w for w in raw_bytes.decode("utf-8-sig").splitlines() if w)
        )
        norm_ref = ("\n".join(norm_words) + "\n").encode("utf-8") if norm_words else b""

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

                _run_and_verify([sys.executable, out_py], norm_ref, "Python (gcc)")
            else:
                print("WARNING: gcc not found on PATH; skipping C-decoder compile check")

            env_fb = os.environ.copy()
            env_fb["PATH"] = ""
            _run_and_verify([sys.executable, out_py], norm_ref, "Pure-Python fallback", env=env_fb)

            py2_bin = shutil.which("python2")
            if py2_bin:
                _run_and_verify([py2_bin, out_py], norm_ref, "python2")

        raw_match = norm_ref == raw_ref
        print(
            f"Verified exact canonical match ({len(norm_ref)} bytes, "
            f"{len(norm_words)} words, raw_identical={raw_match}) in {time.time() - t0:.2f}s"
        )

    return py_src, info, total_sz


if __name__ == "__main__":
    d_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DICT
    o_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    p_path = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_DATASET
    build_py2_solver(d_path, o_path, p_path)
