"""
Asymmetric Numeral Systems (rANS) Module
========================================
Order-0 and Multi-context range Asymmetric Numeral Systems encoders.
"""

from typing import List, Dict, Tuple
from collections import Counter

class RangeANS:
    """Encodes symbol sequences using range Asymmetric Numeral Systems."""

    @staticmethod
    def encode_order0(syms: List[int], M: int = 2048) -> bytes:
        """Order-0 rANS encoder with quantized probability distribution header."""
        n = len(syms)
        if n == 0:
            return b"\x00" * 8
        fc = Counter(syms)
        used = sorted(fc.keys())
        fq = {s: max(1, int(round(fc[s] * M / n))) for s in used}
        cs = sum(fq.values())
        while cs > M:
            b = max(used, key=lambda s: fq[s] if fq[s] > 1 else -1)
            fq[b] -= 1
            cs -= 1
        while cs < M:
            b = max(used, key=lambda s: fc[s] * M / n - fq[s])
            fq[b] += 1
            cs += 1
        
        cum = {}
        pos = 0
        for s in used:
            cum[s] = pos
            pos += fq[s]
        
        h = bytearray([len(used) & 255, len(used) >> 8, n & 255, n >> 8])
        ps = 0
        for s in used:
            d = s - ps
            ps = s
            f = fq[s]
            if d < 255:
                h.append(d)
            else:
                h.append(255)
                h.append(d - 255)
            if f < 240:
                h.append(f)
            else:
                h.append(240 + (f >> 8))
                h.append(f & 255)
        
        L_b = 1 << 16
        mbits = M.bit_length() - 1
        x = L_b
        bs = bytearray()
        for s in reversed(syms):
            f = fq[s]
            c = cum[s]
            xm = ((L_b >> mbits) << 8) * f
            while x >= xm:
                bs.append(x & 255)
                x >>= 8
            x = (x // f) * M + (x % f) + c
        
        for _ in range(4):
            bs.append(x & 255)
            x >>= 8
        
        return bytes(h) + bytes(bs[::-1])

    @staticmethod
    def encode_multicontext_unified(
        seq_syms: List[int],
        seq_ctxs: List[int],
        n_ctx: int,
        M: int = 256
    ) -> Tuple[bytes, bytes]:
        """Multi-context rANS encoder with unified lookup tables."""
        ctx_cnt = [Counter() for _ in range(n_ctx)]
        for s, c in zip(seq_syms, seq_ctxs):
            ctx_cnt[c][s] += 1
        
        f_arr = []
        C_arr = []
        nus = bytearray()
        d_syms = bytearray()
        f_vals = bytearray()
        
        for c in range(n_ctx):
            cnt = ctx_cnt[c]
            if not cnt:
                f_arr.append({})
                C_arr.append({})
                nus.append(0)
                continue
            u = sorted(cnt.keys())
            tot = sum(cnt.values())
            f = {s: max(1, (cnt[s] * M) // tot) for s in u}
            diff = M - sum(f.values())
            while diff != 0:
                if diff > 0:
                    f[max(u, key=lambda s: cnt[s] / f[s])] += 1
                    diff -= 1
                else:
                    f[max([s for s in u if f[s] > 1], key=lambda s: f[s] / cnt[s])] -= 1
                    diff += 1
            C = {}
            ps_ = 0
            for s in u:
                C[s] = ps_
                ps_ += f[s]
            f_arr.append(f)
            C_arr.append(C)
            nus.append(len(u))
            pv = 0
            for s in u:
                d_syms.append(s - pv)
                pv = s
                f_vals.append(f[s] - 1)
        
        L_b = 1 << 16
        mbits = M.bit_length() - 1
        x = L_b
        out = bytearray()
        for s, c in zip(reversed(seq_syms), reversed(seq_ctxs)):
            fs = f_arr[c][s]
            xm = ((L_b >> mbits) << 8) * fs
            while x >= xm:
                out.append(x & 255)
                x >>= 8
            x = ((x // fs) * M) + C_arr[c][s] + (x % fs)
        
        for _ in range(4):
            out.append(x & 255)
            x >>= 8
        out.reverse()
        
        table_bytes = bytes(nus) + bytes(d_syms) + bytes(f_vals)
        stream_bytes = bytes(out)
        return table_bytes, stream_bytes
