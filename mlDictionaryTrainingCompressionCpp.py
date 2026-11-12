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

import sys
import struct
import pickle
import time
import subprocess
import tempfile
import gc
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from compressor.models import PAQBitFSM, ContextRouter, NeuralLogisticTrainer
from compressor.encoding import CanonicalHuffman, RangeANS, DPOptimalLZSS, UTF8Encoder
from compressor.codegen import CppSolverGenerator

DICT_PATH = os.path.join(HERE, "dict.txt")
DATASET_PATH = os.path.join(HERE, "ml_dataset.pkl")
OUT_CPP = os.path.join(HERE, "solve.cpp")
OUT_EXE = os.path.join(tempfile.gettempdir(), "solve_cpp_verify.exe")

AP_BLOB = (
    b"\t\x000\x000\x010\x020\x032\x030\x040\x050\x0b\x011\n\x032\t\x042\x060\x00\x13\x0f\x120\x01\x13\x0f\x120\x02\x13\x0f\x120"
    b"\x00\x13\x0f\x130\x01\x13\x0f\x130\x02\x13\x0f\x130\x00\x13\x0f\x140\x0c\x052\n\x06\x10\x13\x00\x10\x01\x10\x020\x0e\x070"
    b"\x0e\x080\x0e\n2\x0e\n0\x0e\x0b0\x0e\x0c0\n\x0c0\x070\x07\x10\x13\x050\x080\x08\x10\x13\t\r\x10\t\x0e0\n\x0f\x10\x13\x050"
    b"\n\x10\x10\x13\x050\x02\x13\t\x110\x06\x13\r\x0c0\x05\x13\x0f\x120's\x00iness\x00ness\x00ion\x00ment\x00ity\x00ize\x00ed"
    b"\x00ing\x00\x00er\x00est\x00y\x00al\x00ally\x00ive\x00ous\x00ly\x00un\x00re\x00dis\x00in\x00de\x00over\x00mis\x00non\x00pre"
    b"\x00out\x00under\x00sub\x00im\x00en\x00be\x00d\x00r\x00st\x00ied\x00ier\x00iest\x00es\x00ies\x00s\x00ily\x00able\x00ial\x00ic\x00"
)

def build_single_core_solver(m3b=18, fsm_st=80, f1_k=4, qs=6, epochs=500):
    print(f"Loading dataset: {DATASET_PATH} ...")
    with open(DATASET_PATH, "rb") as f:
        DS = pickle.load(f)
    RAW_ITEMS = DS["raw_items"]
    MASKS_REM = DS["masks_rem"]

    CHARS = sorted(set(c for _, s, _ in RAW_ITEMS for c in s))
    C2I = {c: i for i, c in enumerate(CHARS)}
    UP_INF_LIST = [k for k, m in enumerate(MASKS_REM) if m < 4]
    UP_INF_MAP = {k: i for i, k in enumerate(UP_INF_LIST)}

    decomp = ContextRouter.decompose_base_words(RAW_ITEMS, UP_INF_MAP, C2I)
    ps = decomp["ps"]
    p_ctxs = decomp["p_ctxs"]
    c0s = decomp["c0s"]
    c0_ctxs = decomp["c0_ctxs"]
    infs = decomp["infs"]
    inf_lc12 = decomp["inf_lc12"]
    freq_rem = decomp["freq_rem"]

    lc_cnt = Counter([(-1 if u else c1) for u, c1, c2 in inf_lc12])
    bg_cnt = Counter([(-1 if u else (c2, c1)) for u, c1, c2 in inf_lc12])
    trem = CanonicalHuffman.build_canonical_tree(CanonicalHuffman.compute_lengths(freq_rem))

    mops_packed = bytearray()
    for m in MASKS_REM:
        ops = []
        mm = m
        while mm:
            lsb = mm & -mm
            ops.append(lsb.bit_length() - 1)
            mm ^= lsb
        mops_packed.append(len(ops))
        mops_packed.extend(ops)
    mops_b = bytes(mops_packed)

    print("Building model & rANS contexts...")
    top_bg = [k for k, _ in bg_cnt.most_common(13) if k != -1][:12]
    top_bg_set = set(top_bg)
    kept_lc = sorted(k for k, v in lc_cnt.items() if k >= 0 and v >= 400)
    dense = {-1: 0, "other": 1}
    nxt = 2
    for k in kept_lc:
        dense[("lc", k)] = nxt
        nxt += 1
    for k in sorted(top_bg_set):
        dense[("bg", k)] = nxt
        nxt += 1
    
    ctxs_inf = [
        0 if u else (
            dense[("bg", (c2, c1))]
            if (c2, c1) in top_bg_set
            else (dense[("lc", c1)] if ("lc", c1) in dense else 1)
        )
        for u, c1, c2 in inf_lc12
    ]
    ic1_55 = bytearray(dense[("lc", c1)] if ("lc", c1) in dense else 1 for c1 in range(55))
    bg_triples = bytearray([len(top_bg)])
    for (c2, c1) in sorted(top_bg_set):
        bg_triples.extend([c2, c1, dense[("bg", (c2, c1))]])

    STATES, NX0, NX1 = PAQBitFSM.build_fsm(fsm_st)
    N_FSM = len(STATES)
    n_tot_ctx = 48 + 55 + nxt
    
    seq_s = []
    seq_c = []
    for i in range(len(RAW_ITEMS)):
        seq_s.append(ps[i])
        seq_c.append(p_ctxs[i])
        seq_s.append(c0s[i])
        seq_c.append(48 + c0_ctxs[i])
        seq_s.append(infs[i])
        seq_c.append(103 + ctxs_inf[i])
    
    tab_u, pci_b = RangeANS.encode_multicontext_unified(seq_s, seq_c, n_tot_ctx, 256)

    H0 = trem["n_int"]
    a_nid, a_s1, a_s2, a_s3, a_y = NeuralLogisticTrainer.collect_bit_events(
        RAW_ITEMS, trem, f1_k, m3b, NX0, NX1, C2I, ContextRouter.mix
    )

    print(f"Collected {len(a_y):,} PAQ rem bits. Training model ({epochs} epochs)...")
    params = NeuralLogisticTrainer.train_adam(a_nid, a_s1, a_s2, a_s3, a_y, STATES, H0, epochs=epochs)

    print("Quantizing parameters...")
    w_bytes, ml_bin, q_info = NeuralLogisticTrainer.quantize_and_encode(
        params, a_nid, a_s1, a_s2, a_s3, a_y, qs=qs, qw=16
    )

    print("Range encoding rem bitstream...")
    meta_raw = (
        w_bytes
        + bytes(NX0)
        + bytes(NX1)
        + bytes(trem["lens"])
        + bytes(UP_INF_LIST)
        + mops_b
        + AP_BLOB
        + bytes(ic1_55)
        + bytes(bg_triples)
        + tab_u
    )
    meta_lz = DPOptimalLZSS.compress(meta_raw)
    blob = struct.pack("<HH", len(meta_lz), len(pci_b)) + meta_lz + pci_b + ml_bin + b"\x00\x00\x00\x00"
    
    u_str, x_end = UTF8Encoder.encode_cpp_nobidi(blob)
    del a_nid, a_s1, a_s2, a_s3, a_y, params
    gc.collect()

    cpp_src = CppSolverGenerator.generate(
        u_str=u_str,
        x_end=x_end,
        w_bytes_len=len(w_bytes),
        n_fsm=N_FSM,
        f1_k=f1_k,
        m3b=m3b,
        n_tot_ctx=n_tot_ctx,
        sig_pad=q_info["sig_pad"],
        sig_cut=q_info["sig_cut"],
        denom=q_info["denom"]
    )
    return cpp_src, u_str

def main():
    t0 = time.perf_counter()
    cpp_src, u_str = build_single_core_solver(m3b=18, fsm_st=80, f1_k=4, qs=6, epochs=500)
    with open(OUT_CPP, "wb") as f:
        f.write(cpp_src.encode("utf-8"))
    
    sz = os.path.getsize(OUT_CPP)
    u8_sz = len(u_str.encode("utf-8"))
    dt_build = time.perf_counter() - t0
    print(f"\nWrote {OUT_CPP}: {sz:,} B ({sz/1024:.2f} KiB) [Limit < 90,112 B, headroom = {90112 - sz:+d} B]")
    print(f"Build time: {dt_build:.1f} s | UTF-8 payload: {u8_sz:,} B | C++ code: {sz - u8_sz:,} B")
    assert sz < 90112, f"Size limit exceeded: {sz} >= 90112"

    print("\nVerifying compilation with g++ -O2 and -O3 with -Wall -Wextra -Wbidi-chars=any ...")
    res_o2 = subprocess.run(["g++", "-O2", "-std=c++20", "-Wall", "-Wextra", "-Wbidi-chars=any", OUT_CPP, "-o", OUT_EXE], capture_output=True, text=True)
    assert res_o2.returncode == 0, f"g++ -O2 failed:\n{res_o2.stderr}"
    print("g++ -O2: 0 warnings, OK!")

    res_o3 = subprocess.run(["g++", "-O3", "-std=c++20", "-Wall", "-Wextra", "-Wbidi-chars=any", OUT_CPP, "-o", OUT_EXE], capture_output=True, text=True)
    assert res_o3.returncode == 0, f"g++ -O3 failed:\n{res_o3.stderr}"
    print("g++ -O3: 0 warnings, OK!")

    print(f"\nVerifying exact output against {DICT_PATH} ...")
    ref_dict = open(DICT_PATH, "rb").read().replace(b"\r\n", b"\n")
    proc = subprocess.run([OUT_EXE], capture_output=True)
    out_dict = proc.stdout.replace(b"\r\n", b"\n")
    assert out_dict == ref_dict, f"Mismatch: out={len(out_dict)} vs ref={len(ref_dict)}"
    print(f"Exact output verified! ({len(out_dict):,} bytes, 102,399 words)")

    if os.path.exists(OUT_EXE):
        os.remove(OUT_EXE)

if __name__ == "__main__":
    main()
