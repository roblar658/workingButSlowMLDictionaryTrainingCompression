"""
Fast C-in-Python Neural PAQ Dictionary Compression Builder
==========================================================
Trains the 3-order PAQ bit-FSM logistic model and emits standalone solve.py
(< 88 KiB) with an embedded headerless C decoder compiled via gcc plus a
pure-Python fallback.
"""

import os
import subprocess
import sys
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from compressor.codegen import PySolverGenerator
from compressor.pipeline import DictionaryCompressionPipeline

DICT_PATH = os.path.join(HERE, "dict.txt")
DATASET_PATH = os.path.join(HERE, "ml_dataset.pkl")
OUT_PY = os.path.join(HERE, "solve.py")


def build_python_solver(dict_path: str = DICT_PATH, dataset_path: str = DATASET_PATH, epochs: int = 500):
    """Build standalone solve.py using the unified NN+PAQ compression pipeline."""
    art = DictionaryCompressionPipeline.build(dict_path, dataset_path, epochs=epochs)
    py_src, info = PySolverGenerator.generate(art)
    print(
        f"  py sizes: c_src={info['c_src_len']:,} c_lz={info['c_lz_len']:,} "
        f"py_lz={info['py_lz_len']:,} blob={info['blob_len']:,}"
    )
    return py_src, art, info


def main():
    t0 = time.perf_counter()
    py_src, art, info = build_python_solver()
    with open(OUT_PY, "wb") as f:
        f.write(py_src.encode("utf-8"))

    sz = os.path.getsize(OUT_PY)
    u8_sz = info["u8_len"]
    dt_build = time.perf_counter() - t0
    print(f"\nWrote {OUT_PY}: {sz:,} B ({sz / 1024:.2f} KiB) [Limit < 90,112 B, headroom = {90112 - sz:+d} B]")
    print(f"Build time: {dt_build:.1f} s | UTF-8 payloads: {u8_sz:,} B | Wrapper: {sz - u8_sz:,} B")
    if sz >= 90112:
        print(f"WARNING: Size {sz} >= 90112 B (88 KiB limit exceeded for this dictionary)")

    print("\nVerifying embedded C decoder compiles cleanly (-Wall -Wextra -Wbidi-chars=any) ...")
    tmp_dll = os.path.join(HERE, "_verify_py.dll" if os.name == "nt" else "_verify_py.so")
    try:
        gcc_cmd = ["gcc", "-O0", "-Wall", "-Wextra", "-Wbidi-chars=any", "-shared"]
        if os.name != "nt":
            gcc_cmd.append("-fPIC")
        gcc_cmd.extend(["-x", "c", "-", "-o", tmp_dll])
        res_c = subprocess.run(gcc_cmd, input=info["c_src"], capture_output=True, text=True)
        assert res_c.returncode == 0, f"gcc failed:\n{res_c.stderr}"
        assert not res_c.stderr.strip(), f"gcc warnings:\n{res_c.stderr}"
        print("gcc -O0: 0 warnings, OK!")
    finally:
        if os.path.exists(tmp_dll):
            os.remove(tmp_dll)

    print(f"\nVerifying exact output of {OUT_PY} against {DICT_PATH} ...")
    raw_ref = open(DICT_PATH, "rb").read().replace(b"\r\n", b"\n")
    norm_words = sorted(dict.fromkeys(w for w in open(DICT_PATH, "r", encoding="utf-8-sig").read().splitlines() if w))
    norm_ref = ("\n".join(norm_words) + "\n").encode("utf-8") if norm_words else b""

    t_run = time.perf_counter()
    proc = subprocess.run([sys.executable, OUT_PY], capture_output=True)
    dt_run = time.perf_counter() - t_run
    assert proc.returncode == 0, f"solve.py failed (rc={proc.returncode}):\n{proc.stderr.decode('utf-8', 'replace')}"
    out_dict = proc.stdout.replace(b"\r\n", b"\n")
    assert out_dict == norm_ref, f"Mismatch: out={len(out_dict)} vs ref={len(norm_ref)}"
    raw_match = (out_dict == raw_ref)
    print(f"Exact output verified in {dt_run*1000:.1f} ms! ({len(out_dict):,} bytes, {art['total_words']:,} words, byte_identical={raw_match})")


if __name__ == "__main__":
    main()
