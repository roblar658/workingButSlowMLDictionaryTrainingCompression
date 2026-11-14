"""
UTF-8 Stream Encoder Module
===========================
Encodes arbitrary binary streams into valid, safe, non-bidi UTF-8 strings.
Provides variants for C++ raw string literals, Rust raw string literals, and Python strings.
"""

from typing import List, Tuple

_SAFE1_CPP_RS = [i for i in range(1, 128) if i not in (10, 13)]
_SAFE1_PY = [i for i in range(1, 128) if i not in (10, 13, 34, 92)]


def _encode_nobidi(raw: bytes, safe1: List[int]) -> Tuple[str, int]:
    """Encode raw bytes into non-bidi 1-to-3-byte UTF-8 codepoints via rANS."""
    w1 = len(safe1)
    w2 = 1919
    w3 = 6144 + 46992 + 8190
    r = w1 * 16384 + w2 * 128 + w3
    off2 = w1 * 16384
    off3 = off2 + w2 * 128

    l_bound = 1 << 40
    l_eff = (l_bound // r) * r
    if len(raw) < 5:
        raw = raw + b"\x00" * (5 - len(raw))
    x = l_bound + int.from_bytes(raw[:5], "big")
    cps = []
    di = 5
    nd = len(raw)

    while True:
        q, d = divmod(x, r)
        if d < off2:
            idx, rem = divmod(d, 16384)
            x = q * 16384 + rem
            s = safe1[idx]
        elif d < off3:
            idx, rem = divmod(d - off2, 128)
            x = q * 128 + rem
            s = 0x0080 + idx + (1 if (0x0080 + idx) >= 0x061C else 0)
        else:
            idx = d - off3
            x = q
            s = (
                (0x0800 + idx)
                if idx < 6144
                else (
                    (0x2070 + idx - 6144)
                    if idx < 53136
                    else (0xE000 + idx - 53136)
                )
            )
        cps.append(chr(s))
        done = False
        while x < l_eff:
            if di >= nd:
                done = True
                break
            x = (x << 8) | raw[di]
            di += 1
        if done:
            break

    return "".join(cps), x


class UTF8Encoder:
    """Encodes compressed byte payloads into high-density non-bidi UTF-8 codepoints."""

    @staticmethod
    def encode_cpp_nobidi(raw: bytes) -> Tuple[str, int]:
        """Encode binary stream into UTF-8 codepoints safe for C++ R\"qQ(...)qQ\" literals.

        Excludes LF (10) and CR (13), and escapes Unicode bidirectional controls
        to prevent compilation warnings or terminal garbling.
        """
        out_s, x = _encode_nobidi(raw, _SAFE1_CPP_RS)
        assert ")qQ\"" not in out_s
        return out_s, x

    @staticmethod
    def encode_rs_nobidi(raw: bytes) -> Tuple[str, int]:
        """Encode binary stream into UTF-8 codepoints safe for Rust raw string literals.

        Excludes NUL (0), LF (10), CR (13), and Unicode bidirectional controls.
        Pairs with `rs_raw_hashes` (`r#\"...\"#`) so double-quotes stay inside `SAFE1`.
        """
        out_s, x = _encode_nobidi(raw, _SAFE1_CPP_RS)
        assert "\n" not in out_s and "\r" not in out_s and "\0" not in out_s
        return out_s, x

    @staticmethod
    def rs_raw_hashes(s: str) -> str:
        """Return minimal `#` run needed so `r{h}\"{s}\"{h}` never terminates early."""
        if '"' not in s:
            return ""
        hashes = "#"
        while '"' + hashes in s:
            hashes += "#"
        return hashes

    @staticmethod
    def encode_py_nobidi(raw: bytes) -> Tuple[str, int]:
        """Encode binary stream into UTF-8 codepoints safe for Python single-line strings.

        Excludes LF (10), CR (13), quote (34), backslash (92), and Unicode Bidi controls.
        """
        out_s, x = _encode_nobidi(raw, _SAFE1_PY)
        assert '"' not in out_s and '\\' not in out_s and '\n' not in out_s and '\r' not in out_s
        return out_s, x
