"""
Zero-PAQ Model Module
=====================
Implements KL-agglomerative clustered multi-order Markov modeling and
32-grid quantized rANS coding for ultra-fast C-in-Python and pure-Python
dictionary decompression on arbitrary UTF-8 alphabets.
"""

import math
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple
import numpy as np

from compressor.morphology.operators import MorphologicalRules


class ZeroPAQModel:
    """Zero-PAQ KL-clustered Markov/rANS compressor for arbitrary UTF-8 wordlists."""

    GRID_32: Tuple[int, ...] = (
        1, 2, 3, 4, 5, 6, 7, 8,
        10, 12, 14, 16, 18, 20, 24, 28,
        32, 36, 44, 52, 60, 72, 84, 96,
        112, 128, 144, 160, 176, 192, 216, 240,
    )

    @staticmethod
    def kl_cluster_vecs(
        groups: List[Tuple[List[Tuple], np.ndarray]],
        n_clusters: int,
    ) -> Tuple[List[List[Tuple]], int]:
        """Merge context count vectors greedily by minimum KL divergence increase."""
        if not groups:
            return [], 0
        n_clusters = max(1, n_clusters)

        if len(groups) > n_clusters:
            m = len(groups)
            vecs = np.array([v for _, v in groups], dtype=np.float64)
            tots = vecs.sum(axis=1)

            def calc_ents(V: np.ndarray, T: np.ndarray) -> np.ndarray:
                return T * np.log2(np.maximum(T, 1.0)) - (
                    V * np.log2(np.maximum(V, 1.0))
                ).sum(axis=-1)

            ents = calc_ents(vecs, tots)
            dist = np.full((m, m), 1e18, dtype=np.float64)
            for ii in range(m - 1):
                vs = vecs[ii + 1 :] + vecs[ii]
                ts = tots[ii + 1 :] + tots[ii]
                c = calc_ents(vs, ts) - (ents[ii + 1 :] + ents[ii])
                dist[ii, ii + 1 :] = c
                dist[ii + 1 :, ii] = c

            alive = np.ones(m, dtype=bool)
            n_alive = m
            while n_alive > n_clusters:
                flat = int(np.argmin(dist))
                i, j = divmod(flat, m)
                if i > j:
                    i, j = j, i
                vecs[i] += vecs[j]
                tots[i] += tots[j]
                ents[i] = float(calc_ents(vecs[i : i + 1], tots[i : i + 1])[0])
                groups[i][0].extend(groups[j][0])
                alive[j] = False
                n_alive -= 1
                dist[j, :] = 1e18
                dist[:, j] = 1e18
                idx = np.flatnonzero(alive)
                idx = idx[idx != i]
                if len(idx):
                    vs = vecs[idx] + vecs[i]
                    ts = tots[idx] + tots[i]
                    c = calc_ents(vs, ts) - (ents[idx] + ents[i])
                    dist[i, idx] = c
                    dist[idx, i] = c
                dist[i, i] = 1e18
            act_list = list(np.flatnonzero(alive))
        else:
            act_list = list(range(len(groups)))
            vecs = np.array([v for _, v in groups], dtype=np.float64)
            tots = vecs.sum(axis=1)

        # Nearest-neighbor L1 ordering improves LZSS compressibility of frequency/bitmap metadata.
        probs = [vecs[idx] / max(1.0, float(tots[idx])) for idx in act_list]
        used = [False] * len(act_list)
        cur = 0
        used[0] = True
        perm = [act_list[0]]
        for _ in range(len(act_list) - 1):
            p_cur = probs[cur]
            best_d = 1e18
            best_n = -1
            for k in range(len(act_list)):
                if not used[k]:
                    d = float(np.abs(p_cur - probs[k]).sum())
                    if d < best_d:
                        best_d = d
                        best_n = k
            used[best_n] = True
            perm.append(act_list[best_n])
            cur = best_n
        return [groups[idx][0] for idx in perm], len(perm)

    @staticmethod
    def kl_gain_bits(cnt: Counter, parent_cnt: Counter, sym_range: int) -> float:
        """Compute smoothed KL information gain in bits over the parent context."""
        tk = sum(cnt.values())
        tp = sum(parent_cnt.values())
        if tk == 0 or tp == 0:
            return 0.0
        sym_range = max(1, sym_range)
        g = 0.0
        alpha = 0.5
        denom_k = tk + alpha * sym_range
        denom_p = tp + alpha * sym_range
        for s, v in cnt.items():
            g += v * (
                math.log2((v + alpha) / denom_k)
                - math.log2((parent_cnt.get(s, 0) + alpha) / denom_p)
            )
        return g

    @staticmethod
    def _quantize_ctx_freqs(
        cnt: Counter,
        u: List[int],
        grid: Sequence[int],
        g2idx: Dict[int, int],
        M: int,
    ) -> Dict[int, int]:
        """Quantize per-symbol counts onto the 32-level frequency grid."""
        if len(u) == 1:
            return {u[0]: M}

        tot = sum(cnt.values())
        f0 = {s: max(1, (cnt[s] * M) // tot) for s in u}
        diff = M - sum(f0.values())
        while diff != 0:
            if diff > 0:
                f0[max(u, key=lambda s: cnt[s] / f0[s])] += 1
                diff -= 1
            else:
                f0[max((s for s in u if f0[s] > 1), key=lambda s: f0[s] / cnt[s])] -= 1
                diff += 1

        f: Dict[int, int] = {}
        rem_s = u[:-1]
        for s in rem_s:
            f[s] = min(grid, key=lambda gv: abs(math.log((gv + 0.1) / (f0[s] + 0.1))))

        cur_sum = sum(f.values())
        while cur_sum >= M:
            cands = [s for s in rem_s if f[s] > 1]
            s_adj = min(cands, key=lambda s: cnt[s] * math.log(f[s] / grid[g2idx[f[s]] - 1]))
            f[s_adj] = grid[g2idx[f[s_adj]] - 1]
            cur_sum = sum(f.values())
        f[u[-1]] = M - cur_sum

        improved = True
        while improved:
            improved = False
            for s in rem_s:
                idx = g2idx[f[s]]
                best_k = idx
                best_delta = 0.0
                for nid in (idx - 1, idx + 1):
                    if 0 <= nid < len(grid):
                        nf = grid[nid]
                        nl = f[u[-1]] + f[s] - nf
                        if nl >= 1:
                            delta = cnt[s] * math.log(nf / f[s]) + cnt[u[-1]] * math.log(nl / f[u[-1]])
                            if delta > best_delta + 1e-9:
                                best_delta = delta
                                best_k = nid
                if best_k != idx:
                    nf = grid[best_k]
                    f[u[-1]] += f[s] - nf
                    f[s] = nf
                    improved = True
        return f

    @classmethod
    def build_ctx_rans_quant(
        cls,
        seq_syms: List[int],
        seq_ctxs: List[int],
        n_ctx: int,
        grid: Sequence[int],
        off_c0: int,
        off_inf: int,
        off_rem: int,
        nb_p: int,
        nb_c0: int,
        nb_r: int,
        M: int = 256,
    ) -> Tuple[bytes, bytes, bytes]:
        """Encode multi-context symbol stream using adaptive bitmap/delta headers and GRID_32 freqs."""
        g2idx = {v: i for i, v in enumerate(grid)}
        ctx_cnt = [Counter() for _ in range(n_ctx)]
        for s, c in zip(seq_syms, seq_ctxs):
            ctx_cnt[c][s] += 1

        f_arr: List[Dict[int, int]] = []
        C_arr: List[Dict[int, int]] = []
        f_vals = bytearray()
        bm = bytearray()

        for c in range(n_ctx):
            nb = (
                nb_p if c < off_c0
                else (nb_c0 if c < off_inf else (0 if c < off_rem else nb_r))
            )
            cnt = ctx_cnt[c]
            if not cnt:
                f_arr.append({})
                C_arr.append({})
                bm.extend(bytes(nb) if nb > 0 else b"\x00")
                continue

            u = sorted(cnt.keys())
            if nb > 0:
                mask = 0
                for k in u:
                    mask |= 1 << k
                bm.extend(mask.to_bytes(nb, "little"))
            else:
                bm.append(len(u))
                pv = 0
                for k in u:
                    bm.append(k - pv)
                    pv = k

            f = cls._quantize_ctx_freqs(cnt, u, grid, g2idx, M)
            C = {}
            ps_ = 0
            for s in u:
                C[s] = ps_
                ps_ += f[s]
            f_arr.append(f)
            C_arr.append(C)
            for s in u[:-1]:
                f_vals.append(g2idx[f[s]])

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
        return bytes(out), bytes(bm), bytes(f_vals)

    @staticmethod
    def _simulate_ex_slots(
        s_stride: int,
        bl_i: List[Tuple[int, int, int]],
        tl_i: List[Tuple[int, int, int, int]],
        ql_i: List[Tuple[int, int, int, int, int]],
        bg_lst: List[Tuple[int, int, int]],
        tg_g: Dict[Tuple[int, int], List[Tuple[int, int]]],
        qg_g: Dict[Tuple[int, int, int], List[Tuple[int, int]]],
        ubg_lst: List[Tuple[int, int, int]],
        utg_lst: List[Tuple[int, int, int, int]],
    ) -> int:
        """Compute exact EX table row count needed by C/Python context lookup trie."""
        n2 = s_stride * s_stride
        IS = [0] * n2
        IL = [0] * n2
        RL = [0] * n2
        RU = [0] * n2
        EX: List[List[int]] = []

        def ae(arr: List[int], ix: int, k: int, v: int) -> None:
            if arr[ix] < 512:
                EX.append([arr[ix]] * s_stride)
                arr[ix] = 511 + len(EX)
            EX[arr[ix] - 512][k] = v

        def ae2(arr: List[int], ix: int, k1: int, k2: int, v: int) -> None:
            if arr[ix] < 512:
                EX.append([arr[ix]] * s_stride)
                arr[ix] = 511 + len(EX)
            ae(EX[arr[ix] - 512], k1, k2, v)

        for c2, c1, cid in bl_i:
            IS[c2 * s_stride + c1] = 1
            IL[c2 * s_stride + c1] = 1
        for c3, c2, c1, cid in tl_i:
            ae(IS, c2 * s_stride + c1, c3, 1)
            ae(IL, c2 * s_stride + c1, c3, 1)
        for c4, c3, c2, c1, cid in ql_i:
            ae2(IS, c2 * s_stride + c1, c3, c4, 1)
            ae2(IL, c2 * s_stride + c1, c3, c4, 1)
        for (c2, c1), items in sorted(tg_g.items()):
            for c3, cid in items:
                ae(RL, c2 * s_stride + c1, c3, 1)
                ae(RU, c2 * s_stride + c1, c3, 1)
        for (c3, c2, c1), items in sorted(qg_g.items()):
            for c4, cid in items:
                ae2(RL, c2 * s_stride + c1, c3, c4, 1)
                ae2(RU, c2 * s_stride + c1, c3, c4, 1)
        for c3, c2, c1, cid in utg_lst:
            ae(RU, c2 * s_stride + c1, c3, 1)

        return len(EX)

    @classmethod
    def encode(
        cls,
        raw_items: List[Tuple[int, str, int]],
        masks_rem: List[int],
        *,
        c0_clusters: int = 56,
        inf_clusters: int = 32,
        rem_clusters: int = 112,
    ) -> dict:
        """Build generalized Zero-PAQ model metadata and quantized rANS payloads."""
        chars = sorted(set(c for _, s, _ in raw_items for c in s))
        c2i = {c: i for i, c in enumerate(chars)}
        n_ch = len(chars)
        eow = n_ch
        bos = n_ch + 1
        s_stride = n_ch + 2
        if s_stride > 255:
            raise ValueError(f"Alphabet too large for 1-byte symbol tables: s_stride={s_stride} > 255")

        n_masks = len(masks_rem)
        if n_masks > 255:
            raise ValueError(f"Too many active masks for Zero-PAQ byte tables: n_masks={n_masks} > 255")

        is_up = [1 if c.isupper() else 0 for c in chars] + [0, 0]
        up_inf_list = [k for k, m in enumerate(masks_rem) if m < 4]
        up_inf_map = {k: i for i, k in enumerate(up_inf_list)}

        mops_packed = bytearray()
        for m in masks_rem:
            ops = []
            mm = m
            while mm:
                lsb = mm & -mm
                ops.append(lsb.bit_length() - 1)
                mm ^= lsb
            mops_packed.append(len(ops))
            mops_packed.extend(ops)
        mops_b = bytes(mops_packed)

        ps: List[int] = []
        p_ctxs: List[int] = []
        c0s: List[int] = []
        c0_feat: List[Tuple[int, int]] = []
        infs: List[int] = []
        inf_feat: List[Tuple[int, int, int, int, int, bool]] = []
        rem_words: List[Tuple[List[int], int, int, int, int, int]] = []

        total_words = 0
        total_out_bytes = 0
        max_wbytes = 1
        max_bchars = 1
        prev_w = ""
        pc_pad = [bos] * 5
        plen = 0
        pp = 0

        for p, suf, inf in raw_items:
            w = prev_w[:p] + suf
            prev_w = w
            max_bchars = max(max_bchars, len(w))
            wb_l = len(w.encode("utf-8"))
            total_words += 1
            total_out_bytes += wb_l + 1
            max_wbytes = max(max_wbytes, wb_l)

            mm = masks_rem[inf]
            while mm:
                lsb = mm & -mm
                for dw in MorphologicalRules.apply_rule(lsb.bit_length() - 1, w):
                    dl = len(dw.encode("utf-8"))
                    total_words += 1
                    total_out_bytes += dl + 1
                    max_wbytes = max(max_wbytes, dl)
                mm ^= lsb

            w_ids = pc_pad[5 : 5 + p] + [c2i[c] for c in suf]
            wc_pad = [bos] * 5 + w_ids
            u_prev = 1 if (plen > 0 and is_up[pc_pad[5]]) else 0
            ps.append(p)
            p_ctxs.append((min(plen, 11) << 2) | ((1 if pp > 0 else 0) << 1) | u_prev)

            c0 = w_ids[p]
            loc = bos if p == plen else pc_pad[5 + p]
            x1 = pc_pad[4 + p]
            c0s.append(c0)
            c0_feat.append((loc, x1))

            u0 = is_up[w_ids[0]]
            infs.append(up_inf_map[inf] if u0 else inf)
            inf_feat.append((u0, wc_pad[-1], wc_pad[-2], wc_pad[-3], wc_pad[-4], len(w_ids) >= 7))
            rem_words.append((w_ids[p + 1 :] + [eow], w_ids[p], wc_pad[4 + p], wc_pad[3 + p], wc_pad[2 + p], u0))

            pc_pad = wc_pad
            plen = len(w_ids)
            pp = p

        max_p = max(ps, default=0)
        if max_p > 255:
            raise ValueError(f"Prefix length {max_p} exceeds 255 symbol limit")

        rem_events: List[Tuple[int, int, int, int, int, int]] = []
        for toks, c0, c1_p, c2_p, c3_p, u0 in rem_words:
            c1, c2, c3, c4 = c0, c1_p, c2_p, c3_p
            for tok in toks:
                rem_events.append((tok, u0, c4, c3, c2, c1))
                c4, c3, c2, c1 = c3, c2, c1, tok

        # Stage 1: c0 KL clustering
        fb_c: Dict[int, Counter] = defaultdict(Counter)
        bg_c: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
        for s, (loc, x1) in zip(c0s, c0_feat):
            fb_c[loc][s] += 1
            bg_c[(loc, x1)][s] += 1
        kept_c0 = {
            k for k, c in bg_c.items()
            if sum(c.values()) >= 8 and cls.kl_gain_bits(c, fb_c[k[0]], max(1, n_ch)) >= 14.0
        }
        lc_c0: Dict[Tuple, Counter] = defaultdict(Counter)
        for s, (loc, x1) in zip(c0s, c0_feat):
            lc_c0[("bg", loc, x1) if (loc, x1) in kept_c0 else ("FB", loc)][s] += 1
        cl_c0, ac0 = cls.kl_cluster_vecs(
            [
                ([k], np.array([c.get(i, 0) for i in range(max(1, n_ch))], dtype=np.float64))
                for k, c in sorted(lc_c0.items())
            ],
            c0_clusters,
        )
        km0: Dict[Tuple, int] = {}
        fm0: Dict[int, int] = {}
        for cid, ks in enumerate(cl_c0):
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

        # Stage 2: inflections KL clustering
        inf_sym_range = max(1, n_masks)
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

        k_bg_i = {
            k for k, c in bg_i.items()
            if sum(c.values()) >= 8 and cls.kl_gain_bits(c, fb_i[(0, k[1])], inf_sym_range) >= 13.0
        }
        k_tg_i = {
            k for k, c in tg_i.items()
            if sum(c.values()) >= 9 and cls.kl_gain_bits(
                c, bg_i[(k[1], k[2])] if (k[1], k[2]) in k_bg_i else fb_i[(0, k[2])], inf_sym_range
            ) >= 14.0
        }
        k_qg_i = {
            k for k, c in qg_i.items()
            if sum(c.values()) >= 10 and cls.kl_gain_bits(
                c,
                tg_i[(k[1], k[2], k[3])]
                if (k[1], k[2], k[3]) in k_tg_i
                else (bg_i[(k[2], k[3])] if (k[2], k[3]) in k_bg_i else fb_i[(0, k[3])]),
                inf_sym_range,
            ) >= 16.0
        }
        k_wl_i = {
            k for k, c in wl_i.items()
            if sum(c.values()) >= 12 and cls.kl_gain_bits(c, fb_i[(0, k[1])], inf_sym_range) >= 15.0
        }

        lc_i: Dict[Tuple, Counter] = defaultdict(Counter)
        for s, (u0, c1, c2, c3, c4, wl7) in zip(infs, inf_feat):
            if u0:
                k: Tuple = ("FB", (1, 0))
            elif (c4, c3, c2, c1) in k_qg_i:
                k = ("qg", c4, c3, c2, c1)
            elif (c3, c2, c1) in k_tg_i:
                k = ("tg", c3, c2, c1)
            elif (c2, c1) in k_bg_i:
                k = ("bg", c2, c1)
            elif (wl7, c1) in k_wl_i:
                k = ("wl", wl7, c1)
            else:
                k = ("FB", (0, c1))
            lc_i[k][s] += 1

        cl_i, aci = cls.kl_cluster_vecs(
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

        # Stage 3: remainder tokens KL clustering
        rem_sym_range = eow + 1
        fb_r: Dict[int, Counter] = defaultdict(Counter)
        ufb_r: Dict[int, Counter] = defaultdict(Counter)
        bg_r: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
        tg_r: Dict[Tuple[int, int, int], Counter] = defaultdict(Counter)
        qg_r: Dict[Tuple[int, int, int, int], Counter] = defaultdict(Counter)
        ubg_r: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
        utg_r: Dict[Tuple[int, int, int], Counter] = defaultdict(Counter)
        for tok, u0, c4, c3, c2, c1 in rem_events:
            if u0:
                ufb_r[c1][tok] += 1
                ubg_r[(c2, c1)][tok] += 1
                utg_r[(c3, c2, c1)][tok] += 1
            else:
                fb_r[c1][tok] += 1
            bg_r[(c2, c1)][tok] += 1
            tg_r[(c3, c2, c1)][tok] += 1
            qg_r[(c4, c3, c2, c1)][tok] += 1

        all_fb_r = {i: (fb_r[i] + ufb_r[i]) for i in range(s_stride)}
        k_bg_r = {
            k for k, c in bg_r.items()
            if sum(c.values()) >= 6 and cls.kl_gain_bits(c, all_fb_r[k[1]], rem_sym_range) >= 9.0
        }

        def bpar_r(c2: int, c1: int) -> Counter:
            return bg_r[(c2, c1)] if (c2, c1) in k_bg_r else all_fb_r[c1]

        k_tg_r = {
            k for k, c in tg_r.items()
            if sum(c.values()) >= 8 and cls.kl_gain_bits(c, bpar_r(k[1], k[2]), rem_sym_range) >= 14.0
        }

        def tpar_r(c3: int, c2: int, c1: int) -> Counter:
            return tg_r[(c3, c2, c1)] if (c3, c2, c1) in k_tg_r else bpar_r(c2, c1)

        k_qg_r = {
            k for k, c in qg_r.items()
            if sum(c.values()) >= 10 and cls.kl_gain_bits(c, tpar_r(k[1], k[2], k[3]), rem_sym_range) >= 17.0
        }
        k_ubg_r = {
            k for k, c in ubg_r.items()
            if sum(c.values()) >= 7 and cls.kl_gain_bits(c, bpar_r(k[0], k[1]), rem_sym_range) >= 14.0
        }
        k_utg_r = {
            k for k, c in utg_r.items()
            if sum(c.values()) >= 8 and cls.kl_gain_bits(c, tpar_r(k[0], k[1], k[2]), rem_sym_range) >= 16.0
        }

        lc_r: Dict[Tuple, Counter] = defaultdict(Counter)
        eks: List[Tuple] = []
        for tok, u0, c4, c3, c2, c1 in rem_events:
            if u0 and (c3, c2, c1) in k_utg_r:
                k = ("utg", c3, c2, c1)
            elif u0 and (c2, c1) in k_ubg_r:
                k = ("ubg", c2, c1)
            elif (c4, c3, c2, c1) in k_qg_r:
                k = ("qg", c4, c3, c2, c1)
            elif (c3, c2, c1) in k_tg_r:
                k = ("tg", c3, c2, c1)
            elif (c2, c1) in k_bg_r:
                k = ("bg", c2, c1)
            elif u0:
                k = ("UFB", c1)
            else:
                k = ("FB", c1)
            eks.append(k)
            lc_r[k][tok] += 1

        cl_r, acr = cls.kl_cluster_vecs(
            [
                ([k], np.array([c.get(i, 0) for i in range(rem_sym_range)], dtype=np.float64))
                for k, c in sorted(lc_r.items())
            ],
            rem_clusters,
        )
        kmr: Dict[Tuple, int] = {}
        fmr: Dict[int, int] = {}
        ufmr: Dict[int, int] = {}
        for cid, ks in enumerate(cl_r):
            for k in ks:
                if k[0] == "FB":
                    fmr[k[1]] = cid
                elif k[0] == "UFB":
                    ufmr[k[1]] = cid
                else:
                    kmr[k] = cid
        rem_c = [
            fmr.get(k[1], 0)
            if k[0] == "FB"
            else (ufmr.get(k[1], fmr.get(k[1], 0)) if k[0] == "UFB" else kmr[k])
            for k in eks
        ]

        rmap = bytearray(fmr.get(i, 0) for i in range(s_stride))
        rmap.extend(ufmr.get(i, fmr.get(i, 0)) for i in range(s_stride))
        bg_lst = [(k[1], k[2], cid) for k, cid in sorted(kmr.items()) if k[0] == "bg"]
        tg_g: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
        for k, cid in kmr.items():
            if k[0] == "tg":
                tg_g[(k[2], k[3])].append((k[1], cid))
        qg_g: Dict[Tuple[int, int, int], List[Tuple[int, int]]] = defaultdict(list)
        for k, cid in kmr.items():
            if k[0] == "qg":
                qg_g[(k[2], k[3], k[4])].append((k[1], cid))
        ubg_lst = [(k[1], k[2], cid) for k, cid in sorted(kmr.items()) if k[0] == "ubg"]
        utg_lst = [(k[1], k[2], k[3], cid) for k, cid in sorted(kmr.items()) if k[0] == "utg"]

        rmap.extend(len(bg_lst).to_bytes(2, "little"))
        for t in bg_lst:
            rmap.extend(t)
        rmap.extend(len(tg_g).to_bytes(2, "little"))
        for (c2, c1), l in sorted(tg_g.items()):
            rmap.extend([c2, c1, len(l)])
            for c3, cid in sorted(l):
                rmap.extend([c3, cid])
        rmap.extend(len(qg_g).to_bytes(2, "little"))
        for (c3, c2, c1), l in sorted(qg_g.items()):
            rmap.extend([c3, c2, c1, len(l)])
            for c4, cid in sorted(l):
                rmap.extend([c4, cid])
        rmap.extend(len(ubg_lst).to_bytes(2, "little"))
        for t in ubg_lst:
            rmap.extend(t)
        rmap.extend(len(utg_lst).to_bytes(2, "little"))
        for t in utg_lst:
            rmap.extend(t)

        acp = 48
        off_c0 = acp
        off_inf = off_c0 + ac0
        off_rem = off_inf + aci
        n_tot_ctx = off_rem + acr
        assert n_tot_ctx < 512, f"Context count {n_tot_ctx} exceeds EX sentinel 512"

        # Adaptive bitmask byte-widths per section; 0 selects compact delta-list encoding
        nb_p = (max_p // 8 + 1) if max_p < 64 else 0
        nb_c0 = ((max(1, n_ch) - 1) // 8 + 1) if n_ch <= 64 else 0
        nb_r = (eow // 8 + 1) if eow < 64 else 0

        u_s: List[int] = []
        u_c: List[int] = []
        ri = 0
        for i in range(len(raw_items)):
            u_s.append(ps[i])
            u_c.append(p_ctxs[i])
            u_s.append(c0s[i])
            u_c.append(off_c0 + c0_sc[i])
            for tok in rem_words[i][0]:
                u_s.append(tok)
                u_c.append(off_rem + rem_c[ri])
                ri += 1
            u_s.append(infs[i])
            u_c.append(off_inf + inf_sc[i])

        pb_raw, bm, fv = cls.build_ctx_rans_quant(
            u_s, u_c, n_tot_ctx, cls.GRID_32,
            off_c0, off_inf, off_rem, nb_p, nb_c0, nb_r, M=256,
        )

        cs_utf8 = "".join(chars).encode("utf-8")
        up_bytes = bytes(is_up[:n_ch])
        meta_raw = (
            mops_b
            + bytes(up_inf_list)
            + MorphologicalRules.AP_BLOB
            + cs_utf8
            + up_bytes
            + c0_map
            + inf_map
            + bytes(rmap)
            + bm
        )

        ex_need = cls._simulate_ex_slots(
            s_stride, bl_i, tl_i, ql_i, bg_lst, tg_g, qg_g, ubg_lst, utg_lst
        )

        return {
            "meta_raw": meta_raw,
            "fv": fv,
            "pb_raw": pb_raw,
            "n_ch": n_ch,
            "eow": eow,
            "bos": bos,
            "s_stride": s_stride,
            "n_masks": n_masks,
            "n_up_inf": len(up_inf_list),
            "n_base": len(raw_items),
            "total_words": total_words,
            "total_out_bytes": total_out_bytes,
            "max_bchars": max_bchars,
            "max_wbytes": max_wbytes,
            "max_p": max_p,
            "nb_p": nb_p,
            "nb_c0": nb_c0,
            "nb_r": nb_r,
            "off_c0": off_c0,
            "off_inf": off_inf,
            "off_rem": off_rem,
            "n_tot_ctx": n_tot_ctx,
            "ex_cap": max(16, ex_need + 8),
            "fv_len": len(fv),
            "u_cap": max(n_ch + 2, n_masks, max_p + 2, 64),
            "o_up": len(mops_b),
            "o_ap": len(mops_b) + len(up_inf_list),
        }

