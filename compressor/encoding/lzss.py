"""
DP-Optimal LZSS Module
======================
4-pass Dynamic Programming optimal LZSS compressor with token/offset/length entropy encoding.
"""

from typing import List, Tuple, Dict
from collections import Counter, defaultdict
import math
from compressor.encoding.rans import RangeANS

class DPOptimalLZSS:
    """Computes rate-distortion optimal LZSS parse using backward dynamic programming."""

    @staticmethod
    def compress(data: bytes, prefix: bytes = b"") -> bytes:
        """Compress byte stream using 4-pass iterative cost dynamic programming LZSS."""
        full = prefix + data
        off = len(prefix)
        n = len(full)
        last = defaultdict(list)
        matches = [[] for _ in range(n)]

        for i in range(n - 2):
            h = full[i:i + 3]
            lst = last[h]
            if i >= off:
                for prev in reversed(lst[-128:]):
                    d = i - prev
                    if d > 32767:
                        break
                    l = 3
                    while l < 288 and i + l < n and full[prev + l] == full[i + l]:
                        l += 1
                    matches[i].append((d, l))
            lst.append(i)

        lit_c = [8.0] * 256
        len_c = [9.0] * 32
        d_c = [6.0] * 16

        for it in range(4):
            dp = [1e9] * (n + 1)
            dp[n] = 0.0
            choice = [None] * n

            for i in range(n - 1, off - 1, -1):
                best = lit_c[full[i]] + dp[i + 1]
                bc = (0, full[i])
                for d, l_max in matches[i]:
                    db = (d - 1).bit_length()
                    dcost = (d_c[db] + max(0, db - 1)) if it > 0 else (5.0 + max(0, db - 1))
                    for l in range(3, min(33, l_max) + 1):
                        sc = len_c[l - 3] + dcost + dp[i + l]
                        if sc < best:
                            best = sc
                            bc = (1, d, l)
                    if l_max > 33:
                        sc = len_c[31] + 8.0 + dcost + dp[i + l_max]
                        if sc < best:
                            best = sc
                            bc = (1, d, l_max)
                dp[i] = best
                choice[i] = bc

            syms = []
            dbkts = []
            i = off
            while i < n:
                ch = choice[i]
                if ch[0] == 0:
                    syms.append(ch[1])
                    i += 1
                else:
                    _, d, l = ch
                    syms.append(256 + min(31, l - 3))
                    dbkts.append((d - 1).bit_length())
                    i += l
            
            fc = Counter(syms)
            tot = len(syms)
            for s in range(256):
                lit_c[s] = -math.log2((fc.get(s, 0) + 0.5) / (tot + 128))
            for s in range(32):
                len_c[s] = -math.log2((fc.get(256 + s, 0) + 0.5) / (tot + 128))
            dc = Counter(dbkts)
            td = max(1, len(dbkts))
            for b in range(16):
                d_c[b] = -math.log2((dc.get(b, 0) + 0.5) / (td + 8))

        toks = []
        dbkts = []
        lex = bytearray()
        bits = []
        i = off
        while i < n:
            ch = choice[i]
            if ch[0] == 0:
                toks.append(ch[1])
                i += 1
            else:
                _, d, l = ch
                toks.append(256 + min(31, l - 3))
                if l >= 34:
                    lex.append(l - 34)
                db = (d - 1).bit_length()
                dbkts.append(db)
                eb = max(0, db - 1)
                if eb:
                    val = d - (1 + (1 << eb))
                    for b in range(eb):
                        bits.append((val >> b) & 1)
                i += l

        ex_b = bytearray((len(bits) + 7) // 8)
        for idx, b in enumerate(bits):
            if b:
                ex_b[idx >> 3] |= (1 << (idx & 7))

        b1 = RangeANS.encode_order0(toks, 2048)
        b2 = RangeANS.encode_order0(dbkts, 256)
        
        return (
            len(b1).to_bytes(2, "little")
            + len(b2).to_bytes(2, "little")
            + len(lex).to_bytes(2, "little")
            + b1
            + b2
            + bytes(lex)
            + bytes(ex_b)
        )
