"""
Context Router Module
=====================
Decomposes base words into (p, c0, rem, inf) and routes causal contexts.
"""

from typing import Any, Dict, List, Tuple
from collections import Counter, defaultdict
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
        n_ch = len(c2i)
        eow = n_ch
        bos = n_ch + 1
        s_stride = n_ch + 2
        is_up = [0] * s_stride
        for ch, idx in c2i.items():
            if ch.isupper():
                is_up[idx] = 1

        freq_rem = [0] * (n_ch + 1)
        ps = []
        p_ctxs = []
        c0s = []
        c0_ctxs = []
        c0_feat = []
        infs = []
        inf_lc12 = []
        inf_feat = []
        rem_words = []

        pc_pad = [bos] * 5
        plen = 0
        pp = 0

        for p, suf, inf in raw_items:
            w_ids = pc_pad[5:5 + p] + [c2i[c] for c in suf]
            wc_pad = [bos] * 5 + w_ids
            ps.append(p)

            u_prev = 1 if (plen > 0 and is_up[pc_pad[5]]) else 0
            p_ctxs.append((min(plen, 11) << 2) | ((1 if pp > 0 else 0) << 1) | u_prev)

            c0 = w_ids[p]
            loc = bos if p == plen else pc_pad[5 + p]
            c0s.append(c0)
            c0_ctxs.append(loc)
            c0_feat.append((loc, pc_pad[4 + p]))

            tokens = w_ids[p + 1:] + [eow]
            for tok in tokens:
                freq_rem[tok] += 1
            rem_words.append(tokens)

            up0 = is_up[w_ids[0]]
            infs.append(up_inf_map[inf] if up0 else inf)
            inf_lc12.append((up0, wc_pad[-1], wc_pad[-2]))
            inf_feat.append((up0, wc_pad[-1], wc_pad[-2], wc_pad[-3], wc_pad[-4], len(w_ids) >= 7))

            pc_pad = wc_pad
            plen = len(w_ids)
            pp = p

        return {
            "n_ch": n_ch,
            "eow": eow,
            "bos": bos,
            "s_stride": s_stride,
            "is_up": is_up,
            "ps": ps,
            "p_ctxs": p_ctxs,
            "c0s": c0s,
            "c0_ctxs": c0_ctxs,
            "c0_feat": c0_feat,
            "infs": infs,
            "inf_lc12": inf_lc12,
            "inf_feat": inf_feat,
            "rem_words": rem_words,
            "freq_rem": freq_rem,
        }

    @classmethod
    def build_inf_contexts(cls, inf_lc12: List[Tuple[int, int, int]], s_stride: int = 55):
        """Select top inf contexts and build lookup structures."""
        lc_cnt = Counter([(-1 if u else c1) for u, c1, c2 in inf_lc12])
        bg_cnt = Counter([(-1 if u else (c2, c1)) for u, c1, c2 in inf_lc12])

        top_bg = [k for k, _ in bg_cnt.most_common(13) if k != -1][:12]
        top_bg_set = set(top_bg)
        kept_lc = sorted(
            k for k, v in lc_cnt.most_common()
            if k >= 0 and v >= 400
        )[:240]

        dense = {-1: 0, "other": 1}
        nxt = 2
        for k in sorted(kept_lc):
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

        ic1 = bytearray(dense[("lc", c1)] if ("lc", c1) in dense else 1 for c1 in range(s_stride))
        bg_triples = bytearray([len(top_bg)])
        for c2, c1 in sorted(top_bg_set):
            bg_triples.extend([c2, c1, dense[("bg", (c2, c1))]])

        return ctxs_inf, ic1, bg_triples, nxt

    @classmethod
    def build_kl_c0_inf_maps(
        cls,
        decomp: Dict[str, Any],
        n_masks: int,
        c0_clusters: int = 48,
        inf_clusters: int = 32,
    ) -> Dict[str, Any]:
        """Cluster c0 and inflection contexts by minimum KL increase and pack lookup tables."""
        from compressor.models.zero_paq import ZeroPAQModel

        n_ch = max(1, decomp["n_ch"])
        s_stride = decomp["s_stride"]
        c0s = decomp["c0s"]
        c0_feat = decomp["c0_feat"]
        infs = decomp["infs"]
        inf_feat = decomp["inf_feat"]
        inf_sym_range = max(1, n_masks)

        # Stage A: c0 KL clustering over (loc, prev_char_x1)
        fb_c: Dict[int, Counter] = defaultdict(Counter)
        bg_c: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
        for s, (l, x1) in zip(c0s, c0_feat):
            fb_c[l][s] += 1
            bg_c[(l, x1)][s] += 1
        kept_c = {
            k for k, c in bg_c.items()
            if sum(c.values()) >= 8 and ZeroPAQModel.kl_gain_bits(c, fb_c[k[0]], n_ch) >= 14.0
        }
        lc_c: Dict[Tuple, Counter] = defaultdict(Counter)
        for s, (l, x1) in zip(c0s, c0_feat):
            lc_c[("bg", l, x1) if (l, x1) in kept_c else ("FB", l)][s] += 1
        cl_c, ac0 = ZeroPAQModel.kl_cluster_vecs(
            [
                ([k], np.array([c.get(i, 0) for i in range(n_ch)], dtype=np.float64))
                for k, c in sorted(lc_c.items())
            ],
            c0_clusters,
        )
        km0: Dict[Tuple, int] = {}
        fm0: Dict[int, int] = {}
        for cid, ks in enumerate(cl_c):
            for k in ks:
                if k[0] == "FB":
                    fm0[k[1]] = cid
                else:
                    km0[k] = cid
        c0_sc = [km0.get(("bg", l, x1), fm0.get(l, 0)) for l, x1 in c0_feat]
        rb0 = [(k[1], k[2], cid) for k, cid in sorted(km0.items()) if cid != fm0.get(k[1], 0)]
        c0_map = (
            bytes(fm0.get(i, 0) for i in range(s_stride))
            + len(rb0).to_bytes(2, "little")
            + bytes(x for t in rb0 for x in t)
        )

        # Stage B: inflection KL clustering over (u0, c1..c4, char_len >= 7)
        fb_i: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
        bg_i: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
        tg_i: Dict[Tuple[int, int, int], Counter] = defaultdict(Counter)
        qg_i: Dict[Tuple[int, int, int, int], Counter] = defaultdict(Counter)
        wl_i: Dict[Tuple[bool, int], Counter] = defaultdict(Counter)
        for s, (u0, c1, c2, c3, c4, wl7) in zip(infs, inf_feat):
            fb_i[(u0, 0 if u0 else c1)][s] += 1
            if u0 == 0:
                bg_i[(c2, c1)][s] += 1
                tg_i[(c3, c2, c1)][s] += 1
                qg_i[(c4, c3, c2, c1)][s] += 1
                wl_i[(wl7, c1)][s] += 1

        k_bg = {
            k for k, c in bg_i.items()
            if sum(c.values()) >= 8 and ZeroPAQModel.kl_gain_bits(c, fb_i[(0, k[1])], inf_sym_range) >= 13.0
        }
        k_tg = {
            k for k, c in tg_i.items()
            if sum(c.values()) >= 9 and ZeroPAQModel.kl_gain_bits(
                c, bg_i[(k[1], k[2])] if (k[1], k[2]) in k_bg else fb_i[(0, k[2])], inf_sym_range
            ) >= 14.0
        }
        k_qg = {
            k for k, c in qg_i.items()
            if sum(c.values()) >= 10 and ZeroPAQModel.kl_gain_bits(
                c,
                tg_i[(k[1], k[2], k[3])]
                if (k[1], k[2], k[3]) in k_tg
                else (bg_i[(k[2], k[3])] if (k[2], k[3]) in k_bg else fb_i[(0, k[3])]),
                inf_sym_range,
            ) >= 16.0
        }
        k_wl = {
            k for k, c in wl_i.items()
            if sum(c.values()) >= 12 and ZeroPAQModel.kl_gain_bits(c, fb_i[(0, k[1])], inf_sym_range) >= 15.0
        }

        lc_i: Dict[Tuple, Counter] = defaultdict(Counter)
        for s, (u0, c1, c2, c3, c4, wl7) in zip(infs, inf_feat):
            if u0:
                k: Tuple = ("FB", (1, 0))
            elif (c4, c3, c2, c1) in k_qg:
                k = ("qg", c4, c3, c2, c1)
            elif (c3, c2, c1) in k_tg:
                k = ("tg", c3, c2, c1)
            elif (c2, c1) in k_bg:
                k = ("bg", c2, c1)
            elif (wl7, c1) in k_wl:
                k = ("wl", wl7, c1)
            else:
                k = ("FB", (0, c1))
            lc_i[k][s] += 1

        cl_i, aci = ZeroPAQModel.kl_cluster_vecs(
            [
                ([k], np.array([c.get(i, 0) for i in range(inf_sym_range)], dtype=np.float64))
                for k, c in sorted(lc_i.items())
            ],
            inf_clusters,
        )
        kmi: Dict[Tuple, int] = {}
        fmi: Dict[Tuple[int, int], int] = {}
        for cid, ks in enumerate(cl_i):
            for k in ks:
                if k[0] == "FB":
                    fmi[k[1]] = cid
                else:
                    kmi[k] = cid

        inf_sc: List[int] = []
        for u0, c1, c2, c3, c4, wl7 in inf_feat:
            if u0:
                inf_sc.append(fmi.get((1, 0), 0))
            elif ("qg", c4, c3, c2, c1) in kmi:
                inf_sc.append(kmi[("qg", c4, c3, c2, c1)])
            elif ("tg", c3, c2, c1) in kmi:
                inf_sc.append(kmi[("tg", c3, c2, c1)])
            elif ("bg", c2, c1) in kmi:
                inf_sc.append(kmi[("bg", c2, c1)])
            elif ("wl", wl7, c1) in kmi:
                inf_sc.append(kmi[("wl", wl7, c1)])
            else:
                inf_sc.append(fmi.get((0, c1), 0))

        bl_i = [(k[1], k[2], cid) for k, cid in sorted(kmi.items()) if k[0] == "bg"]
        tl_i = [(k[1], k[2], k[3], cid) for k, cid in sorted(kmi.items()) if k[0] == "tg"]
        ql_i = [(k[1], k[2], k[3], k[4], cid) for k, cid in sorted(kmi.items()) if k[0] == "qg"]
        inf_map = (
            bytes(kmi.get(("wl", False, i), fmi.get((0, i), 0)) for i in range(s_stride))
            + bytes([fmi.get((1, 0), 0)])
            + bytes(kmi.get(("wl", True, i), fmi.get((0, i), 0)) for i in range(s_stride))
            + len(bl_i).to_bytes(2, "little") + bytes(x for t in bl_i for x in t)
            + len(tl_i).to_bytes(2, "little") + bytes(x for t in tl_i for x in t)
            + len(ql_i).to_bytes(2, "little") + bytes(x for t in ql_i for x in t)
        )

        ex_need = ZeroPAQModel._simulate_ex_slots(
            s_stride, bl_i, tl_i, ql_i, [], {}, {}, [], []
        )

        return {
            "c0_sc": c0_sc,
            "inf_sc": inf_sc,
            "ac0": ac0,
            "aci": aci,
            "c0_map": c0_map,
            "inf_map": inf_map,
            "ex_cap": max(16, ex_need + 8),
        }
