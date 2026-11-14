"""
Shared Solver Layout Helper
===========================
Computes shared buffer capacities, bit-packing widths, and metadata offsets
used by both the C++20 and Rust solver generators.
"""

from typing import Any, Dict


def compute_solver_layout(
    u_str: str,
    w_bytes_len: int,
    n_fsm: int,
    m3b: int,
    n_tot_ctx: int,
    n_ch: int,
    n_masks: int,
    n_up_inf: int,
    total_words: int,
    total_out_bytes: int,
    max_bchars: int,
    max_wbytes: int,
    blob_len: int,
    meta_raw_len: int,
    use_h2_hash: bool,
    ex_cap: int,
) -> Dict[str, Any]:
    """Compute shared decompression layout constants and bit-packing widths."""
    stride = 96 if n_fsm <= 96 else 128
    m3 = 1 << m3b
    mask3 = m3 - 1
    eow = n_ch
    bos = n_ch + 1
    s_stride = n_ch + 2
    h0 = max(1, n_ch)
    n_ch_safe = max(1, n_ch)
    n_masks_safe = max(1, n_masks)
    ex_rows = max(16, ex_cap)
    m2_size = ((1 << 18) + h0) if use_h2_hash else (s_stride * s_stride * h0)

    w_buf = max(total_out_bytes + 64, 1024)
    o_buf = max(total_out_bytes + 64, 1024)
    nw_cap = max(total_words + 16, 64)
    len_bits = max(6, max_wbytes.bit_length())
    len_mask = (1 << len_bits) - 1
    id_bits = max(17, total_words.bit_length())
    id_mask_hex = hex((1 << id_bits) - 1)
    cmp_start = (64 - id_bits) // 8
    is_ol_32 = (w_buf.bit_length() + len_bits) <= 32

    x0o = w_bytes_len
    x1o = x0o + n_fsm
    lno = x1o + n_fsm
    upo = lno + n_ch + 1
    mpo = upo + n_up_inf

    return {
        "stride": stride,
        "m3": m3,
        "mask3": mask3,
        "eow": eow,
        "bos": bos,
        "s_stride": s_stride,
        "ss": s_stride * s_stride,
        "h0": h0,
        "n_ch_safe": n_ch_safe,
        "n_masks_safe": n_masks_safe,
        "ex_rows": ex_rows,
        "rs_ex_rows": 2 * s_stride + ex_rows,
        "m2_size": m2_size,
        "w_buf": w_buf,
        "o_buf": o_buf,
        "nw_cap": nw_cap,
        "len_bits": len_bits,
        "len_mask": len_mask,
        "id_bits": id_bits,
        "id_mask_hex": id_mask_hex,
        "cmp_start": cmp_start,
        "is_ol_32": is_ol_32,
        "mb_buf": max(meta_raw_len + 128, 1024),
        "an_start": max(blob_len + 128, 1024),
        "b_buf": max(blob_len + 128, 1024) + 16,
        "cp_buf": max(len(u_str) + 16, 256),
        "wv_buf": max(meta_raw_len + 128, 1024),
        "db_buf": max(meta_raw_len + 128, 512),
        "pc_buf": max(max_bchars + 16, 64),
        "wb_buf": max(max_wbytes + 16, 64),
        "n_tot_ctx_safe": max(1, n_tot_ctx),
        "x0o": x0o,
        "x1o": x1o,
        "lno": lno,
        "upo": upo,
        "mpo": mpo,
        "q1o": h0,
        "q2o": h0 + n_fsm,
        "q3o": h0 + 2 * n_fsm,
        "qwo": h0 + 3 * n_fsm,
    }
