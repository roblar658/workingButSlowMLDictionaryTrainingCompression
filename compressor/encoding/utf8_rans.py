"""
UTF-8 Stream Encoder Module
===========================
Encodes arbitrary binary streams into valid, safe, non-bidi UTF-8 strings.
Provides variants for C++ raw string literals and Python string literals.
"""

from typing import Tuple

class UTF8Encoder:
    """Encodes compressed byte payloads into high-density non-bidi UTF-8 codepoints."""

    @staticmethod
    def encode_cpp_nobidi(raw: bytes) -> Tuple[str, int]:
        """Encode binary stream into UTF-8 codepoints safe for C++ R\"qQ(...)qQ\" literals.
        
        Excludes LF (10) and CR (13), and escapes Unicode bidirectional controls
        to prevent compilation warnings or terminal garbling.
        """
        SAFE1 = [i for i in range(1, 128) if i not in (10, 13)]
        W1 = len(SAFE1)
        W2 = 1919
        W3 = 6144 + 46992 + 8190
        R = W1 * 16384 + W2 * 128 + W3
        OFF2 = W1 * 16384
        OFF3 = OFF2 + W2 * 128

        L = 1 << 40
        x = L + int.from_bytes(raw[:5], "big")
        cps = []
        di = 5
        nd = len(raw)

        while True:
            q, d = divmod(x, R)
            if d < OFF2:
                idx, rem = divmod(d, 16384)
                x = q * 16384 + rem
                s = SAFE1[idx]
            elif d < OFF3:
                idx, rem = divmod(d - OFF2, 128)
                x = q * 128 + rem
                s = 0x0080 + idx + (1 if (0x0080 + idx) >= 0x061C else 0)
            else:
                idx = d - OFF3
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
            while x < L:
                if di >= nd:
                    done = True
                    break
                x = (x << 8) | raw[di]
                di += 1
            if done:
                break

        out_s = "".join(cps)
        assert ")qQ\"" not in out_s
        return out_s, x

    @staticmethod
    def encode_py_nobidi(raw: bytes) -> Tuple[str, int]:
        """Encode binary stream into UTF-8 codepoints safe for Python single-line strings.
        
        Excludes LF (10), CR (13), quote (34), backslash (92), and Unicode Bidi controls.
        """
        SAFE1 = [i for i in range(1, 128) if i not in (10, 13, 34, 92)]
        W1 = len(SAFE1)
        W2 = 1919
        W3 = 6144 + 46992 + 8190
        R = W1 * 16384 + W2 * 128 + W3
        OFF2 = W1 * 16384
        OFF3 = OFF2 + W2 * 128

        L = 1 << 40
        x = L + int.from_bytes(raw[:5], "big")
        cps = []
        di = 5
        nd = len(raw)

        while True:
            q, d = divmod(x, R)
            if d < OFF2:
                idx, rem = divmod(d, 16384)
                x = q * 16384 + rem
                s = SAFE1[idx]
            elif d < OFF3:
                idx, rem = divmod(d - OFF2, 128)
                x = q * 128 + rem
                s = 0x0080 + idx + (1 if (0x0080 + idx) >= 0x061C else 0)
            else:
                idx = d - OFF3
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
            while x < L:
                if di >= nd:
                    done = True
                    break
                x = (x << 8) | raw[di]
                di += 1
            if done:
                break

        out_s = "".join(cps)
        assert '"' not in out_s and '\\' not in out_s and '\n' not in out_s and '\r' not in out_s
        return out_s, x
