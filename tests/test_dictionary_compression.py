#!/usr/bin/env python3
"""Automated test suite for arbitrary-UTF-8 ML dictionary compression and C++/Python/Rust solvers."""

import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compressor.codegen import CppSolverGenerator, PySolverGenerator, RsSolverGenerator
from compressor.encoding import DPOptimalLZSS, UTF8Encoder
from compressor.models import ContextRouter
from compressor.morphology import DatasetBuilder
from compressor.pipeline import DictionaryCompressionPipeline
from mlDictionaryTrainingCompressionCpp import build_single_core_solver
from mlDictionaryTrainingCompressionRs import build_rust_solver


SIZE_LIMIT_BYTES = 90112  # 88 KiB
BIDI_CODEPOINTS = set(range(0x202A, 0x202F)) | set(range(0x2066, 0x206A)) | {0x061C, 0x200E, 0x200F}


def _decode_py_u8(s: str, x: int) -> bytes:
    """Reference decoder matching UTF8Encoder.encode_py_nobidi."""
    u0 = (1 << 40) // 2322190
    out = bytearray()
    for ch in reversed(s):
        k = ord(ch)
        sh = 22 if k < 128 else 15 if k < 2048 else 8
        while x >= (u0 << sh):
            out.append(x & 255)
            x >>= 8
        if k < 128:
            x = (x >> 14) * 2322190 + ((k - 1 - (k > 10) - (k > 13) - (k > 34) - (k > 92)) << 14) + (x & 16383)
        elif k < 2048:
            x = (x >> 7) * 2322190 + 2015232 + ((k - 128 - (k > 1564)) << 7) + (x & 127)
        else:
            off = 2048 if k < 8304 else 2160 if k < 57344 else 4208
            x = x * 2322190 + 2260864 + k - off
    return (x - (1 << 40)).to_bytes(5, "big") + bytes(out[::-1])


def _decode_cpp_u8(s: str, x: int) -> bytes:
    """Reference decoder matching UTF8Encoder.encode_cpp_nobidi."""
    safe1 = [i for i in range(1, 128) if i not in (10, 13)]
    c_tbl = {ch: idx << 14 for idx, ch in enumerate(safe1)}
    r_val = 2354958
    o2 = 2048000
    o3 = 2293632
    p4 = 1 << 40
    u0 = p4 // r_val
    out = bytearray()
    for ch in reversed(s):
        k = ord(ch)
        sh = 22 if k < 128 else 15 if k < 2048 else 8
        while x >= (u0 << sh):
            out.append(x & 255)
            x >>= 8
        if k < 128:
            x = (x >> 14) * r_val + c_tbl[k] + (x & 16383)
        elif k < 2048:
            x = (x >> 7) * r_val + o2 + ((k - 128 - (k > 1564)) << 7) + (x & 127)
        else:
            off = 2048 if k < 8304 else 2160 if k < 57344 else 4208
            x = x * r_val + o3 + k - off
    return (x - p4).to_bytes(5, "big") + bytes(out[::-1])


def _lzss_decompress(blob: bytes) -> bytes:
    """Reference decoder matching DPOptimalLZSS.compress."""
    def rd(b: bytes, mb: int):
        nu, nl = struct.unpack_from("<HH", b)
        p = 4
        s = 0
        tbl = []
        for _ in range(nu):
            d = b[p]; p += 1
            if d == 255:
                d += b[p]; p += 1
            s += d
            f = b[p]; p += 1
            if f >= 240:
                f = ((f - 240) << 8) | b[p]; p += 1
            tbl.extend((s, f, j) for j in range(f))
        x = int.from_bytes(b[p:p + 4], "big")
        p += 4
        out = []
        mask = (1 << mb) - 1
        for _ in range(nl):
            sym, freq, j = tbl[x & mask]
            x = freq * (x >> mb) + j
            while x < 65536:
                x = (x << 8) | b[p]
                p += 1
            out.append(sym)
        return out

    l1, l2, l3 = struct.unpack_from("<HHH", blob)
    w_syms = rd(blob[6:], 11)
    d_syms = rd(blob[6 + l1:], 8)
    lp = 6 + l1 + l2
    extra = blob[lp + l3:]
    out = bytearray()
    di = li = bp = 0
    for s in w_syms:
        if s < 256:
            out.append(s)
        else:
            length = 34 + blob[lp + li] if s == 287 else 3 + s - 256
            if s == 287:
                li += 1
            d_code = d_syms[di]
            di += 1
            if d_code < 2:
                dist = d_code + 1
            else:
                g = d_code - 1
                dist = 1 + (1 << g)
                for k in range(g):
                    dist += ((extra[bp >> 3] >> (bp & 7)) & 1) << k
                    bp += 1
            for _ in range(length):
                out.append(out[-dist])
    return bytes(out)


def _norm_dict_bytes(raw_bytes: bytes) -> bytes:
    """Compute canonical UTF-8 LF-delimited sorted unique dictionary bytes."""
    words = sorted(dict.fromkeys(w for w in raw_bytes.decode("utf-8-sig").splitlines() if w))
    return ("\n".join(words) + "\n").encode("utf-8") if words else b""


class TestEncodingPrimitives(unittest.TestCase):
    """Unit tests for UTF-8 non-bidi coding, LZSS compression, and KL context routing."""

    def test_utf8_encoder_roundtrip_and_nobidi(self):
        rng_payload = bytes((i * 73 + (i >> 3) * 19) & 255 for i in range(4096))
        samples = [b"", b"\x00", b"\xff" * 7, b")qQ\"\\\"\r\n's", rng_payload]
        for raw in samples:
            py_s, py_x = UTF8Encoder.encode_py_nobidi(raw)
            self.assertFalse(any(c in py_s for c in ('"', "\\", "\n", "\r")))
            self.assertFalse(any(ord(c) in BIDI_CODEPOINTS for c in py_s))
            dec_py = _decode_py_u8(py_s, py_x)
            expected = raw if len(raw) >= 5 else raw + b"\x00" * (5 - len(raw))
            self.assertEqual(dec_py, expected)

            cpp_s, cpp_x = UTF8Encoder.encode_cpp_nobidi(raw)
            self.assertNotIn(')qQ"', cpp_s)
            self.assertFalse(any(c in cpp_s for c in ("\n", "\r", "\0")))
            self.assertFalse(any(ord(c) in BIDI_CODEPOINTS for c in cpp_s))
            dec_cpp = _decode_cpp_u8(cpp_s, cpp_x)
            self.assertEqual(dec_cpp, expected)

            rs_s, rs_x = UTF8Encoder.encode_rs_nobidi(raw)
            hashes = UTF8Encoder.rs_raw_hashes(rs_s)
            self.assertNotIn('"' + hashes, rs_s)
            self.assertFalse(any(c in rs_s for c in ("\n", "\r", "\0")))
            self.assertFalse(any(ord(c) in BIDI_CODEPOINTS for c in rs_s))
            dec_rs = _decode_cpp_u8(rs_s, rs_x)
            self.assertEqual(dec_rs, expected)

    def test_dp_optimal_lzss_roundtrip(self):
        samples = [
            b"",
            b"a",
            ("ÆØÅæøå's\n" * 60 + "abcdef" * 40).encode("utf-8"),
            bytes((i * 131 ^ (i >> 2)) & 255 for i in range(1500)),
        ]
        for raw in samples:
            comp = DPOptimalLZSS.compress(raw)
            self.assertEqual(_lzss_decompress(comp), raw)

    def test_context_router_kl_bounds(self):
        raw_items = [(0, "æøå", 0), (1, "b", 1), (0, "Øst", 0)]
        c2i = {ch: i for i, ch in enumerate(sorted({ch for _, s, _ in raw_items for ch in s}))}
        decomp = ContextRouter.decompose_base_words(raw_items, {0: 0, 1: 1}, c2i)
        kl = ContextRouter.build_kl_c0_inf_maps(decomp, 2, c0_clusters=48, inf_clusters=32)
        self.assertGreaterEqual(kl["ac0"], 1)
        self.assertLessEqual(kl["ac0"], 48)
        self.assertGreaterEqual(kl["aci"], 1)
        self.assertLessEqual(kl["aci"], 32)
        self.assertGreaterEqual(kl["ex_cap"], 16)


class TestArbitraryDictionaries(unittest.TestCase):
    """End-to-end tests on empty, tiny, Norwegian UTF-8, large-alphabet, and long-word dicts."""

    def _assert_roundtrip(self, raw_bytes: bytes, use_pickle: bool = False) -> None:
        exp = _norm_dict_bytes(raw_bytes)
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            dp = td_p / "dict.txt"
            dp.write_bytes(raw_bytes)
            pkl_path = str(td_p / "ml_dataset.pkl") if use_pickle else None
            if use_pickle:
                DatasetBuilder(str(dp)).save_pickle(pkl_path)

            art = DictionaryCompressionPipeline.build(str(dp), dataset_path=pkl_path, epochs=80)

            # 1) Python generator: strict C compile check + gcc path + forced pure-Python fallback
            py_src, info = PySolverGenerator.generate(art)
            py_p = td_p / "solve.py"
            py_p.write_text(py_src, encoding="utf-8")

            if shutil.which("gcc"):
                dll_p = td_p / ("check.dll" if os.name == "nt" else "check.so")
                gcc_cmd = ["gcc", "-O0", "-Wall", "-Wextra", "-Wbidi-chars=any", "-shared"]
                if os.name != "nt":
                    gcc_cmd.append("-fPIC")
                gcc_cmd.extend(["-x", "c", "-", "-o", str(dll_p)])
                res_c = subprocess.run(gcc_cmd, input=info["c_src"], capture_output=True, text=True)
                self.assertEqual(res_c.returncode, 0, res_c.stderr)
                self.assertEqual(res_c.stderr.strip(), "", res_c.stderr)

                res_py_gcc = subprocess.run([sys.executable, str(py_p)], capture_output=True)
                self.assertEqual(res_py_gcc.returncode, 0, res_py_gcc.stderr.decode("utf-8", "replace"))
                self.assertEqual(res_py_gcc.stdout.replace(b"\r\n", b"\n"), exp)

            env_no_gcc = os.environ.copy()
            env_no_gcc["PATH"] = ""
            res_py_fb = subprocess.run([sys.executable, str(py_p)], capture_output=True, env=env_no_gcc)
            self.assertEqual(res_py_fb.returncode, 0, res_py_fb.stderr.decode("utf-8", "replace"))
            self.assertEqual(res_py_fb.stdout.replace(b"\r\n", b"\n"), exp)

            # 2) C++ generator: strict g++ compile check + execution
            cpp_src, _, _ = build_single_core_solver(str(dp), dataset_path=pkl_path, epochs=80)
            cpp_p = td_p / "solve.cpp"
            exe_p = td_p / ("solve.exe" if os.name == "nt" else "solve")
            cpp_p.write_text(cpp_src, encoding="utf-8")

            if shutil.which("g++"):
                res_cxx = subprocess.run(
                    ["g++", "-O2", "-std=c++20", "-Wall", "-Wextra", "-Wbidi-chars=any", str(cpp_p), "-o", str(exe_p)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(res_cxx.returncode, 0, res_cxx.stderr)
                self.assertEqual(res_cxx.stderr.strip(), "", res_cxx.stderr)
                res_run = subprocess.run([str(exe_p)], capture_output=True)
                self.assertEqual(res_run.returncode, 0, res_run.stderr.decode("utf-8", "replace"))
                self.assertEqual(res_run.stdout.replace(b"\r\n", b"\n"), exp)

            # 3) Rust generator: strict rustc compile check + execution
            rs_src, _, _ = build_rust_solver(str(dp), dataset_path=pkl_path, epochs=80)
            rs_p = td_p / "solve.rs"
            rs_exe_p = td_p / ("solve_rs.exe" if os.name == "nt" else "solve_rs")
            rs_p.write_text(rs_src, encoding="utf-8")

            if shutil.which("rustc"):
                for flags in (["-O"], ["-C", "opt-level=3", "--edition", "2021"]):
                    res_rs = subprocess.run(
                        ["rustc", *flags, "-D", "warnings", str(rs_p), "-o", str(rs_exe_p)],
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(res_rs.returncode, 0, res_rs.stderr)
                    self.assertEqual(res_rs.stderr.strip(), "", res_rs.stderr)
                res_run_rs = subprocess.run([str(rs_exe_p)], capture_output=True)
                self.assertEqual(res_run_rs.returncode, 0, res_run_rs.stderr.decode("utf-8", "replace"))
                self.assertEqual(res_run_rs.stdout.replace(b"\r\n", b"\n"), exp)

    def test_empty_and_single_word_dicts(self):
        self._assert_roundtrip(b"")
        self._assert_roundtrip(b"a\n")
        self._assert_roundtrip("å\n".encode("utf-8"))

    def test_norwegian_utf8_apostrophe_bom_and_morph_edge_cases(self):
        words = [
            "Ære", "Æres", "Ærlig", "Øst", "Øst's", "Østs", "Øy", "Øys", "Ås", "Åss",
            "blå", "blåe", "blåne", "blå's", "bær", "bære", "bærer", "bæres",
            "føy", "føyes", "føyde", "gøy", "gøye", "løy", "løye", "øle", "øler",
            "søvn", "være", "åre", "ærlig", "ønske", "ønsker", "ønsket", "ønskes",
            "don't", "it's", "rock'n'roll",
            "abut", "abutment", "abbreviation", "abbreviation's", "abbreviations",
            "allergy", "allergic", "box", "boxes", "try", "tries", "tried", "trying",
        ]
        # Reverse with UTF-8 BOM, duplicates, blank lines, and CRLF line endings
        dirty = "\ufeff" + "\r\n".join(reversed(words + ["blå", "", "Øst", "Ås"])) + "\r\n\r\n"
        self._assert_roundtrip(dirty.encode("utf-8"), use_pickle=True)

    def test_all_uppercase_dict(self):
        self._assert_roundtrip("ABC\nFOO\nFOO's\nFOOs\nÆRLIG\nØST\nÅS\nÅSs\n".encode("utf-8"))

    def test_large_utf8_alphabet(self):
        words = [chr(0x00C0 + i) + chr(0x03B1 + (i % 24)) + chr(0x4E00 + (i % 15)) for i in range(90)]
        words += ["cat", "cats", "dog", "dogs", "ÆØÅæøå"]
        self._assert_roundtrip(("\n".join(words) + "\n").encode("utf-8"))

    def test_long_words_and_long_shared_prefix(self):
        pref = "æøå" * 100
        words = [pref + "alpha", pref + "beta", pref + "betas", "x" * 72 + "y", "x" * 72 + "ies"]
        self._assert_roundtrip(("\n".join(words) + "\n").encode("utf-8"))


class TestRepositorySolvers(unittest.TestCase):
    """Validates generated solve.cpp, solve.py, and solve.rs on the main REPO dict.txt."""

    def test_main_dict_solvers(self):
        dict_p = REPO_ROOT / "dict.txt"
        cpp_p = REPO_ROOT / "solve.cpp"
        py_p = REPO_ROOT / "solve.py"
        rs_p = REPO_ROOT / "solve.rs"
        if not (dict_p.exists() and cpp_p.exists() and py_p.exists() and rs_p.exists()):
            self.skipTest("dict.txt / solve.cpp / solve.py / solve.rs not present")

        raw_ref = dict_p.read_bytes().replace(b"\r\n", b"\n")
        exp = _norm_dict_bytes(raw_ref)

        cpp_sz = cpp_p.stat().st_size
        py_sz = py_p.stat().st_size
        rs_sz = rs_p.stat().st_size
        self.assertLess(cpp_sz, SIZE_LIMIT_BYTES, f"solve.cpp {cpp_sz} >= {SIZE_LIMIT_BYTES}")
        self.assertLess(py_sz, SIZE_LIMIT_BYTES, f"solve.py {py_sz} >= {SIZE_LIMIT_BYTES}")
        self.assertLess(rs_sz, SIZE_LIMIT_BYTES, f"solve.rs {rs_sz} >= {SIZE_LIMIT_BYTES}")

        with tempfile.TemporaryDirectory() as td:
            if shutil.which("g++"):
                exe_p = Path(td) / ("solve_main.exe" if os.name == "nt" else "solve_main")
                res_o2 = subprocess.run(
                    ["g++", "-O2", "-std=c++20", "-Wall", "-Wextra", "-Wbidi-chars=any", str(cpp_p), "-o", str(exe_p)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(res_o2.returncode, 0, res_o2.stderr)
                self.assertEqual(res_o2.stderr.strip(), "", res_o2.stderr)
                res_run = subprocess.run([str(exe_p)], capture_output=True)
                self.assertEqual(res_run.returncode, 0)
                self.assertEqual(res_run.stdout.replace(b"\r\n", b"\n"), exp)

            if shutil.which("rustc"):
                rs_exe_p = Path(td) / ("solve_main_rs.exe" if os.name == "nt" else "solve_main_rs")
                for flags in (["-O"], ["-C", "opt-level=3", "--edition", "2021"]):
                    res_rs = subprocess.run(
                        ["rustc", *flags, "-D", "warnings", str(rs_p), "-o", str(rs_exe_p)],
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(res_rs.returncode, 0, res_rs.stderr)
                    self.assertEqual(res_rs.stderr.strip(), "", res_rs.stderr)
                res_run_rs = subprocess.run([str(rs_exe_p)], capture_output=True)
                self.assertEqual(res_run_rs.returncode, 0)
                self.assertEqual(res_run_rs.stdout.replace(b"\r\n", b"\n"), exp)

            if shutil.which("gcc"):
                res_py_gcc = subprocess.run([sys.executable, str(py_p)], capture_output=True)
                self.assertEqual(res_py_gcc.returncode, 0, res_py_gcc.stderr.decode("utf-8", "replace"))
                self.assertEqual(res_py_gcc.stdout.replace(b"\r\n", b"\n"), exp)

            env_no_gcc = os.environ.copy()
            env_no_gcc["PATH"] = ""
            res_py_fb = subprocess.run([sys.executable, str(py_p)], capture_output=True, env=env_no_gcc)
            self.assertEqual(res_py_fb.returncode, 0, res_py_fb.stderr.decode("utf-8", "replace"))
            self.assertEqual(res_py_fb.stdout.replace(b"\r\n", b"\n"), exp)


if __name__ == "__main__":
    unittest.main(verbosity=2)

