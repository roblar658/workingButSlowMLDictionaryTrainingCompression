"""
Context Router Module
=====================
Decomposes base words into (p, c0, rem, inf) and routes causal contexts.
"""

from typing import Dict, List, Tuple
from collections import Counter
import numpy as np

class ContextRouter:
    """Extracts causal context indices for p, c0, inf, and rem."""

    BOS: int = 54
    EOW: int = 53

    @staticmethod
    def mix(h: int) -> int:
        """32-bit Knuth multiplicative integer hash."""
        h = (h * 0x9E3779B1) & 0xFFFFFFFF
        return (h ^ (h >> 16)) & 0xFFFFFFFF

    @classmethod
    def decompose_base_words(
        cls,
        raw_items: List[Tuple[int, str, int]],
        up_inf_map: Dict[int, int],
        c2i: Dict[str, int]
    ):
        """Extract prefix lengths, first characters, suffix remainder tokens, and inflections."""
        freq_rem = [0] * 54
        ps = []
        p_ctxs = []
        c0s = []
        c0_ctxs = []
        infs = []
        inf_lc12 = []
        rem_words = []

        pc_pad = [cls.BOS] * 5
        plen = 0
        pp = 0

        for p, suf, inf in raw_items:
            w_ids = pc_pad[5:5 + p] + [c2i[c] for c in suf]
            wc_pad = [cls.BOS] * 5 + w_ids
            ps.append(p)
            
            u_prev = 1 if (plen > 0 and 1 <= pc_pad[5] <= 26) else 0
            p_ctxs.append((min(plen, 11) << 2) | ((1 if pp > 0 else 0) << 1) | u_prev)
            
            c0 = w_ids[p]
            loc = cls.BOS if p == plen else pc_pad[5 + p]
            c0s.append(c0)
            c0_ctxs.append(loc)
            
            tokens = w_ids[p + 1:] + [cls.EOW]
            for tok in tokens:
                freq_rem[tok] += 1
            rem_words.append(tokens)

            up0 = 1 if (1 <= w_ids[0] <= 26) else 0
            infs.append(up_inf_map[inf] if up0 else inf)
            inf_lc12.append((up0, wc_pad[-1], wc_pad[-2]))

            pc_pad = wc_pad
            plen = len(w_ids)
            pp = p

        return {
            "ps": ps,
            "p_ctxs": p_ctxs,
            "c0s": c0s,
            "c0_ctxs": c0_ctxs,
            "infs": infs,
            "inf_lc12": inf_lc12,
            "rem_words": rem_words,
            "freq_rem": freq_rem,
        }

    @classmethod
    def build_inf_contexts(cls, inf_lc12: List[Tuple[int, int, int]]):
        """Select top inf contexts and build lookup structures."""
        lc_cnt = Counter([(-1 if u else c1) for u, c1, c2 in inf_lc12])
        bg_cnt = Counter([(-1 if u else (c2, c1)) for u, c1, c2 in inf_lc12])
        
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
                dense[("bg", (c2, c1))] if (c2, c1) in top_bg_set
                else (dense[("lc", c1)] if ("lc", c1) in dense else 1)
            )
            for u, c1, c2 in inf_lc12
        ]
        
        ic1_55 = bytearray(dense[("lc", c1)] if ("lc", c1) in dense else 1 for c1 in range(55))
        bg_triples = bytearray([len(top_bg)])
        for c2, c1 in sorted(top_bg_set):
            bg_triples.extend([c2, c1, dense[("bg", (c2, c1))]])
            
        return ctxs_inf, ic1_55, bg_triples, nxt
