#!/usr/bin/env python3
"""
Rust Solver Trainer & Compiler Pipeline
=======================================
Trains the 3-order PAQ bit-FSM logistic model and generates standalone solve.rs (< 88 KiB).
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

from compressor.codegen import RsSolverGenerator
from compressor.encoding import UTF8Encoder
from compressor.pipeline import DictionaryCompressionPipeline

DICT_PATH = os.path.join(HERE, "dict.txt")
DATASET_PATH = os.path.join(HERE, "ml_dataset.pkl")
OUT_RS = os.path.join(HERE, "solve.rs")
OUT_EXE = os.path.join(tempfile.gettempdir(), "solve_rs_verify" + (".exe" if os.name == "nt" else ""))


def build_rust_solver(
    dict_path: str = DICT_PATH,
    dataset_path: str = DATASET_PATH,
    m3b: int = 20,
    fsm_st: int = 80,
    f1_k: int = 4,
    qs: int = 6,
    epochs: int = 500,
):
    """Build standalone pure-Rust solver using the unified NN+PAQ compression pipeline."""
    art = DictionaryCompressionPipeline.build(
        dict_path, dataset_path, m3b=m3b, fsm_st=fsm_st, f1_k=f1_k, qs=qs, epochs=epochs
    )
    meta_lz = art["meta_lz"]
    pci_b = art["pci_b"]
    ml_bin = art["ml_bin"]
    blob = struct.pack("<HH", len(meta_lz), len(pci_b)) + meta_lz + pci_b + ml_bin + b"\x00\x00\x00\x00"
    u_str, x_end = UTF8Encoder.encode_rs_nobidi(blob)

    rs_src = RsSolverGenerator.generate(
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
    return rs_src, u_str, art["total_words"]


def main():
    t0 = time.perf_counter()
    rs_src, u_str, total_words = build_rust_solver()
    with open(OUT_RS, "wb") as f:
        f.write(rs_src.encode("utf-8"))

    sz = os.path.getsize(OUT_RS)
    u8_sz = len(u_str.encode("utf-8"))
    dt_build = time.perf_counter() - t0
    print(f"\nWrote {OUT_RS}: {sz:,} B ({sz / 1024:.2f} KiB) [Limit < 90,112 B, headroom = {90112 - sz:+d} B]")
    print(f"Build time: {dt_build:.1f} s | UTF-8 payload: {u8_sz:,} B | Rust code: {sz - u8_sz:,} B")
    if sz >= 90112:
        print(f"WARNING: Size {sz} >= 90112 B (88 KiB limit exceeded for this dictionary)")

    print("\nVerifying compilation with rustc -O and -C opt-level=3 (default & 2021 editions) with -D warnings ...")
    for label, extra in [("rustc -O", ["-O"]), ("rustc -C opt-level=3 --edition 2021", ["-C", "opt-level=3", "--edition", "2021"])]:
        res = subprocess.run(["rustc", *extra, "-D", "warnings", OUT_RS, "-o", OUT_EXE], capture_output=True, text=True)
        assert res.returncode == 0, f"{label} failed:\n{res.stderr}"
        assert not res.stderr.strip(), f"{label} warnings:\n{res.stderr}"
        print(f"{label}: 0 warnings, OK!")

    print(f"\nVerifying exact output against {DICT_PATH} ...")
    with open(DICT_PATH, "rb") as f:
        raw_ref = f.read().replace(b"\r\n", b"\n")
    with open(DICT_PATH, "r", encoding="utf-8-sig") as f:
        norm_words = sorted(dict.fromkeys(w for w in f.read().splitlines() if w))
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
