"""
Shared Compression Pipeline Module
==================================
Unifies dataset loading, morphological decomposition, PAQ FSM / neural logistic
training, and rANS/LZSS encoding for both C++ and Python solver generators.
"""

import gc
import os
import pickle
from typing import Optional, Dict, Any

from compressor.morphology import MorphologicalRules, DatasetBuilder
from compressor.models import PAQBitFSM, ContextRouter, NeuralLogisticTrainer
from compressor.encoding import CanonicalHuffman, RangeANS, DPOptimalLZSS


class DictionaryCompressionPipeline:
    """Builds quantized NN+PAQ compressed artifacts from a dictionary dataset."""

    @staticmethod
    def build(
        dict_path: str,
        dataset_path: Optional[str] = None,
        *,
        m3b: int = 20,
        fsm_st: int = 80,
        f1_k: int = 4,
        qs: int = 6,
        epochs: int = 500,
    ) -> Dict[str, Any]:
        """Load or construct dataset, train neural logistic PAQ model, and encode payloads."""
        if dataset_path and os.path.exists(dataset_path):
            print(f"Loading dataset: {dataset_path} ...")
            with open(dataset_path, "rb") as f:
                ds = pickle.load(f)
        else:
            builder = DatasetBuilder(dict_path)
            raw_items, masks_rem = builder.build()
            ds = {
                "raw_items": raw_items,
                "masks_rem": masks_rem,
                "n_bases": len(raw_items),
            }
            if dataset_path:
                with open(dataset_path, "wb") as f:
                    pickle.dump(ds, f, protocol=pickle.HIGHEST_PROTOCOL)

        raw_items = ds["raw_items"]
        masks_rem = ds["masks_rem"]

        chars = sorted(set(c for _, s, _ in raw_items for c in s))
        c2i = {c: i for i, c in enumerate(chars)}
        up_inf_list = [k for k, m in enumerate(masks_rem) if m < 4]
        up_inf_map = {k: i for i, k in enumerate(up_inf_list)}

        decomp = ContextRouter.decompose_base_words(raw_items, up_inf_map, c2i)
        n_ch = decomp["n_ch"]
        s_stride = decomp["s_stride"]
        is_up = decomp["is_up"]
        ps = decomp["ps"]
        p_ctxs = decomp["p_ctxs"]
        c0s = decomp["c0s"]
        infs = decomp["infs"]
        freq_rem = decomp["freq_rem"]

        kl = ContextRouter.build_kl_c0_inf_maps(
            decomp, len(masks_rem), c0_clusters=48, inf_clusters=32
        )
        c0_sc = kl["c0_sc"]
        inf_sc = kl["inf_sc"]
        ac0 = kl["ac0"]
        aci = kl["aci"]
        trem = CanonicalHuffman.build_canonical_tree(
            CanonicalHuffman.compute_lengths(freq_rem)
        )

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

        print("Building model & rANS contexts...")
        states, nx0, nx1 = PAQBitFSM.build_fsm(fsm_st)
        n_fsm = len(states)
        n_tot_ctx = 48 + ac0 + aci

        seq_s = []
        seq_c = []
        for i in range(len(raw_items)):
            seq_s.append(ps[i])
            seq_c.append(p_ctxs[i])
            seq_s.append(c0s[i])
            seq_c.append(48 + c0_sc[i])
            seq_s.append(infs[i])
            seq_c.append(48 + ac0 + inf_sc[i])

        tab_u, pci_b = RangeANS.encode_multicontext_unified(seq_s, seq_c, n_tot_ctx, 256)

        h0 = trem["n_int"]
        h_stride = 1 << max(6, (h0 - 1).bit_length() if h0 > 1 else 0)
        cb_shift = max(6, (s_stride - 1).bit_length())
        use_h2_hash = (s_stride * s_stride * h0 > 393216)

        a_nid, a_s1, a_s2, a_s3, a_y = NeuralLogisticTrainer.collect_bit_events(
            raw_items, trem, f1_k, m3b, nx0, nx1, c2i, ContextRouter.mix
        )

        print(f"Collected {len(a_y):,} PAQ rem bits. Training model ({epochs} epochs)...")
        params = NeuralLogisticTrainer.train_adam(
            a_nid, a_s1, a_s2, a_s3, a_y, states, h0, epochs=epochs
        )

        print("Quantizing parameters...")
        w_bytes, ml_bin, q_info = NeuralLogisticTrainer.quantize_and_encode(
            params, a_nid, a_s1, a_s2, a_s3, a_y, qs=qs, qw=16
        )

        cs_utf8 = "".join(chars).encode("utf-8")
        up_bytes = bytes(is_up[:n_ch])

        # Exact output bounds so arbitrary UTF-8 dictionaries never overflow static buffers
        total_words = 0
        total_out_bytes = 0
        max_wbytes = 1
        max_bchars = 1
        prev_w = ""
        for p, suf, inf_idx in raw_items:
            w = prev_w[:p] + suf
            prev_w = w
            max_bchars = max(max_bchars, len(w))
            wb_l = len(w.encode("utf-8"))
            total_words += 1
            total_out_bytes += wb_l + 1
            max_wbytes = max(max_wbytes, wb_l)
            mm = masks_rem[inf_idx]
            while mm:
                lsb = mm & -mm
                for dw in MorphologicalRules.apply_rule(lsb.bit_length() - 1, w):
                    dl = len(dw.encode("utf-8"))
                    total_words += 1
                    total_out_bytes += dl + 1
                    max_wbytes = max(max_wbytes, dl)
                mm ^= lsb

        meta_raw = (
            w_bytes
            + bytes(nx0)
            + bytes(nx1)
            + bytes(trem["lens"])
            + bytes(up_inf_list)
            + mops_b
            + MorphologicalRules.AP_BLOB
            + cs_utf8
            + up_bytes
            + kl["c0_map"]
            + kl["inf_map"]
            + tab_u
        )
        meta_lz = DPOptimalLZSS.compress(meta_raw)
        print(
            f"  sizes: meta_lz={len(meta_lz):,} pci_b={len(pci_b):,} "
            f"ml_bin={len(ml_bin):,} w_bytes={len(w_bytes):,} tab_u={len(tab_u):,}"
        )

        del a_nid, a_s1, a_s2, a_s3, a_y, params
        gc.collect()

        return {
            "meta_lz": meta_lz,
            "pci_b": pci_b,
            "ml_bin": ml_bin,
            "meta_raw_len": len(meta_raw),
            "w_bytes_len": len(w_bytes),
            "n_fsm": n_fsm,
            "f1_k": f1_k,
            "m3b": m3b,
            "n_tot_ctx": n_tot_ctx,
            "sig_pad": q_info["sig_pad"],
            "sig_cut": q_info["sig_cut"],
            "denom": q_info["denom"],
            "n_ch": n_ch,
            "s_stride": s_stride,
            "ac0": ac0,
            "aci": aci,
            "ex_cap": kl["ex_cap"],
            "n_masks": len(masks_rem),
            "n_up_inf": len(up_inf_list),
            "n_base": len(raw_items),
            "total_words": total_words,
            "total_out_bytes": total_out_bytes,
            "max_bchars": max_bchars,
            "max_wbytes": max_wbytes,
            "use_h2_hash": use_h2_hash,
            "cb_shift": cb_shift,
            "h_stride": h_stride,
        }

