#!/usr/bin/env python3
"""
C++ Solver Trainer & Compiler Pipeline
======================================
Trains the 3-order PAQ bit-FSM logistic model and generates standalone solve.cpp (< 88 KiB, ~9ms).
Delegates core algorithms to the modular `compressor` package.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import struct
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from compressor.encoding import UTF8Encoder
from compressor.codegen import CppSolverGenerator
from compressor.pipeline import DictionaryCompressionPipeline

DICT_PATH = os.path.join(HERE, "dict.txt")
DATASET_PATH = os.path.join(HERE, "ml_dataset.pkl")
OUT_CPP = os.path.join(HERE, "solve.cpp")
OUT_EXE = os.path.join(tempfile.gettempdir(), "solve_cpp_verify" + (".exe" if os.name == "nt" else ""))


def build_single_core_solver(
    dict_path: str = DICT_PATH,
    dataset_path: str = DATASET_PATH,
    m3b: int = 20,
    fsm_st: int = 80,
    f1_k: int = 4,
    qs: int = 6,
    epochs: int = 500,
):
    """Build standalone C++20 solver using the unified NN+PAQ compression pipeline."""
    art = DictionaryCompressionPipeline.build(
        dict_path, dataset_path, m3b=m3b, fsm_st=fsm_st, f1_k=f1_k, qs=qs, epochs=epochs
    )
    meta_lz = art["meta_lz"]
    pci_b = art["pci_b"]
    ml_bin = art["ml_bin"]
    blob = struct.pack("<HH", len(meta_lz), len(pci_b)) + meta_lz + pci_b + ml_bin + b"\x00\x00\x00\x00"
    u_str, x_end = UTF8Encoder.encode_cpp_nobidi(blob)

    cpp_src = CppSolverGenerator.generate(
        u_str=u_str,
        x_end=x_end,
        w_bytes_len=art["w_bytes_len"],
        n_fsm=art["n_fsm"],
        f1_k=art["f1_k"],
        m3b=art["m3b"],
        n_tot_ctx=art["n_tot_ctx"],
        sig_pad=art["sig_pad"],
        sig_cut=art["sig_cut"],
        denom=art["denom"],
        n_ch=art["n_ch"],
        n_masks=art["n_masks"],
        n_up_inf=art["n_up_inf"],
        n_base=art["n_base"],
        total_words=art["total_words"],
        total_out_bytes=art["total_out_bytes"],
        max_bchars=art["max_bchars"],
        max_wbytes=art["max_wbytes"],
        blob_len=len(blob),
        meta_raw_len=art["meta_raw_len"],
        use_h2_hash=art["use_h2_hash"],
        cb_shift=art["cb_shift"],
        h_stride=art["h_stride"],
        ac0=art["ac0"],
        aci=art["aci"],
        ex_cap=art["ex_cap"],
    )
    return cpp_src, u_str, art["total_words"]


def main():
    t0 = time.perf_counter()
    cpp_src, u_str, total_words = build_single_core_solver()
    with open(OUT_CPP, "wb") as f:
        f.write(cpp_src.encode("utf-8"))

    sz = os.path.getsize(OUT_CPP)
    u8_sz = len(u_str.encode("utf-8"))
    dt_build = time.perf_counter() - t0
    print(f"\nWrote {OUT_CPP}: {sz:,} B ({sz / 1024:.2f} KiB) [Limit < 90,112 B, headroom = {90112 - sz:+d} B]")
    print(f"Build time: {dt_build:.1f} s | UTF-8 payload: {u8_sz:,} B | C++ code: {sz - u8_sz:,} B")
    if sz >= 90112:
        print(f"WARNING: Size {sz} >= 90112 B (88 KiB limit exceeded for this dictionary)")

    print("\nVerifying compilation with g++ -O2 and -O3 with -Wall -Wextra -Wbidi-chars=any ...")
    res_o2 = subprocess.run(["g++", "-O2", "-std=c++20", "-Wall", "-Wextra", "-Wbidi-chars=any", OUT_CPP, "-o", OUT_EXE], capture_output=True, text=True)
    assert res_o2.returncode == 0, f"g++ -O2 failed:\n{res_o2.stderr}"
    assert not res_o2.stderr.strip(), f"g++ -O2 warnings:\n{res_o2.stderr}"
    print("g++ -O2: 0 warnings, OK!")

    res_o3 = subprocess.run(["g++", "-O3", "-std=c++20", "-Wall", "-Wextra", "-Wbidi-chars=any", OUT_CPP, "-o", OUT_EXE], capture_output=True, text=True)
    assert res_o3.returncode == 0, f"g++ -O3 failed:\n{res_o3.stderr}"
    assert not res_o3.stderr.strip(), f"g++ -O3 warnings:\n{res_o3.stderr}"
    print("g++ -O3: 0 warnings, OK!")

    print(f"\nVerifying exact output against {DICT_PATH} ...")
    raw_ref = open(DICT_PATH, "rb").read().replace(b"\r\n", b"\n")
    norm_words = sorted(dict.fromkeys(w for w in open(DICT_PATH, "r", encoding="utf-8-sig").read().splitlines() if w))
    norm_ref = ("\n".join(norm_words) + "\n").encode("utf-8") if norm_words else b""
    proc = subprocess.run([OUT_EXE], capture_output=True)
    out_dict = proc.stdout.replace(b"\r\n", b"\n")
    assert out_dict == norm_ref, f"Mismatch: out={len(out_dict)} vs ref={len(norm_ref)}"
    raw_match = (out_dict == raw_ref)
    print(f"Exact output verified! ({len(out_dict):,} bytes, {total_words:,} words, byte_identical={raw_match})")

    if os.path.exists(OUT_EXE):
        os.remove(OUT_EXE)


if __name__ == "__main__":
    main()
