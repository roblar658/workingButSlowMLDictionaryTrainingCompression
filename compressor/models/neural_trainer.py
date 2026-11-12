"""
Neural Logistic Trainer Module
==============================
Trains multi-context PAQ-mixing logistic models using Adam, quantizes weights to int8,
and encodes bitstreams with a 32-bit multi-precision range encoder.
"""

from typing import Dict, List, Tuple, Any
import math
import numpy as np
import gc

class NeuralLogisticTrainer:
    """Trains, quantizes, and range-encodes 3-order PAQ bit-history neural models."""

    @staticmethod
    def collect_bit_events(
        raw_items: List[Tuple[int, str, int]],
        trem: Dict[str, Any],
        f1_k: int,
        m3b: int,
        nx0: np.ndarray,
        nx1: np.ndarray,
        c2i: Dict[str, int],
        mix_fn,
        bos: int = 54,
        eow: int = 53
    ):
        """Pass through raw items and collect bit-level events and FSM state histories."""
        H0 = trem["n_int"]
        M3 = 1 << m3b
        MASK3 = M3 - 1

        FS1 = np.zeros(55 * f1_k * 64, dtype=np.uint8)
        FS2 = np.zeros(225280, dtype=np.uint8)
        FS3 = np.zeros(M3 + H0, dtype=np.uint8)

        bit_nid = []
        bit_s1 = []
        bit_s2 = []
        bit_s3 = []
        bit_y = []

        pc_pad = [bos] * 5
        for p, suf, inf in raw_items:
            w_ids = pc_pad[5:5 + p] + [c2i[c] for c in suf]
            wc_pad = [bos] * 5 + w_ids
            c1 = w_ids[p]
            c2 = wc_pad[5 + p - 1]
            c3 = wc_pad[5 + p - 2]
            wl = 0
            for tok in w_ids[p + 1:] + [eow]:
                c1_off = (c1 * f1_k + min(wl, f1_k - 1)) * 64
                h2_b = (c2 * 64 + c1) * 64
                h3_b = mix_fn((c3 << 12) ^ (c2 << 6) ^ c1 ^ 0x8888) & MASK3
                for u, b in trem["codes"][tok]:
                    k1 = c1_off + u
                    k2 = h2_b + u
                    k3 = h3_b + u
                    s1 = FS1[k1]
                    s2 = FS2[k2]
                    s3 = FS3[k3]
                    bit_nid.append(u)
                    bit_s1.append(s1)
                    bit_s2.append(s2)
                    bit_s3.append(s3)
                    bit_y.append(b)
                    FS1[k1] = nx1[s1] if b else nx0[s1]
                    FS2[k2] = nx1[s2] if b else nx0[s2]
                    FS3[k3] = nx1[s3] if b else nx0[s3]
                if tok != eow:
                    c3, c2, c1 = c2, c1, tok
                    wl += 1
            pc_pad = wc_pad

        del FS1, FS2, FS3
        return (
            np.array(bit_nid, dtype=np.uint16),
            np.array(bit_s1, dtype=np.uint8),
            np.array(bit_s2, dtype=np.uint8),
            np.array(bit_s3, dtype=np.uint8),
            np.array(bit_y, dtype=np.float32)
        )

    @staticmethod
    def train_adam(
        a_nid: np.ndarray,
        a_s1: np.ndarray,
        a_s2: np.ndarray,
        a_s3: np.ndarray,
        a_y: np.ndarray,
        states: np.ndarray,
        h0: int,
        epochs: int = 500
    ) -> Dict[str, np.ndarray]:
        """Train weights with Adam optimizer and cosine learning rate decay."""
        n_fsm = len(states)
        s_init = np.log((states[:, 1] + 0.5) / (states[:, 0] + 0.5)).astype(np.float32)
        params = {
            "w0": np.zeros(h0, dtype=np.float32),
            "S1": s_init.copy(),
            "S2": s_init.copy(),
            "S3": s_init.copy(),
            "Wm": np.tile(np.array([0.6, 0.7, 0.9], dtype=np.float32), (h0, 1))
        }
        m = {k: np.zeros_like(v) for k, v in params.items()}
        v = {k: np.zeros_like(v) for k, v in params.items()}

        for ep in range(1, epochs + 1):
            wm0 = params["Wm"][a_nid, 0]
            wm1 = params["Wm"][a_nid, 1]
            wm2 = params["Wm"][a_nid, 2]
            z = params["w0"][a_nid] + wm0 * params["S1"][a_s1] + wm1 * params["S2"][a_s2] + wm2 * params["S3"][a_s3]
            pr = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
            gz = (pr - a_y).astype(np.float32)

            gwm = np.zeros_like(params["Wm"])
            gwm[:, 0] = np.bincount(a_nid, weights=gz * params["S1"][a_s1], minlength=h0)
            gwm[:, 1] = np.bincount(a_nid, weights=gz * params["S2"][a_s2], minlength=h0)
            gwm[:, 2] = np.bincount(a_nid, weights=gz * params["S3"][a_s3], minlength=h0)

            gr = {
                "w0": np.bincount(a_nid, weights=gz, minlength=h0).astype(np.float32),
                "S1": np.bincount(a_s1, weights=gz * wm0, minlength=n_fsm).astype(np.float32),
                "S2": np.bincount(a_s2, weights=gz * wm1, minlength=n_fsm).astype(np.float32),
                "S3": np.bincount(a_s3, weights=gz * wm2, minlength=n_fsm).astype(np.float32),
                "Wm": gwm
            }

            cur_lr = 0.0005 + 0.0495 * 0.5 * (1 + math.cos(math.pi * ep / epochs))
            b1t = 1 - 0.9**ep
            b2t = 1 - 0.999**ep
            for k in params:
                g = gr[k]
                m[k] = 0.9 * m[k] + 0.1 * g
                v[k] = 0.999 * v[k] + 0.001 * (g * g)
                params[k] -= cur_lr * (m[k] / b1t) / (np.sqrt(v[k] / b2t) + 1e-8)

        return params

    @staticmethod
    def quantize_and_encode(
        params: Dict[str, np.ndarray],
        a_nid: np.ndarray,
        a_s1: np.ndarray,
        a_s2: np.ndarray,
        a_s3: np.ndarray,
        a_y: np.ndarray,
        qs: int = 6,
        qw: int = 16
    ) -> Tuple[bytes, bytes, Dict[str, Any]]:
        """Quantize parameters to int8 and encode the bitstream via 32-bit range encoder."""
        denom = qs * qw
        qw0 = np.clip(np.round(params["w0"] * qs), -127, 127).astype(np.int32)
        qS1 = np.clip(np.round(params["S1"] * qs), -127, 127).astype(np.int32)
        qS2 = np.clip(np.round(params["S2"] * qs), -127, 127).astype(np.int32)
        qS3 = np.clip(np.round(params["S3"] * qs), -127, 127).astype(np.int32)
        qWm = np.clip(np.round(params["Wm"] * qw), -64, 64).astype(np.int32)

        wm_q = qWm[a_nid]
        logit_d = qw0[a_nid] * qw + wm_q[:, 0] * qS1[a_s1] + wm_q[:, 1] * qS2[a_s2] + wm_q[:, 2] * qS3[a_s3]
        sig_pad = max(2048, int(-logit_d.min()) + 2, int(logit_d.max()) + 2)
        arg = np.clip(np.arange(-sig_pad, sig_pad + 1) / float(denom), -40.0, 40.0)
        SIG_ARR = np.clip(np.rint(4096.0 / (1.0 + np.exp(arg))), 1, 4095).astype(np.int32)
        sig_cut = min(sig_pad, 850)
        SIG_ARR[sig_pad + 1 + sig_cut:] = 1
        SIG_ARR[:sig_pad - sig_cut] = 4095

        p_int = SIG_ARR[logit_d + sig_pad]

        # 32-bit range encoding
        TOP = 1 << 32
        BOT = 1 << 24
        low = 0
        rng_e = TOP
        out = bytearray()
        for b, pv in zip(a_y, p_int):
            r1 = (rng_e >> 12) * int(pv)
            if b == 0:
                rng_e = r1
            else:
                low += r1
                rng_e -= r1
            while rng_e < BOT:
                d = low >> 24
                if d >= 256:
                    for idx in range(len(out) - 1, -1, -1):
                        out[idx] = (out[idx] + 1) & 255
                        if out[idx] != 0:
                            break
                    d &= 255
                out.append(d)
                low = (low & 0xFFFFFF) << 8
                rng_e <<= 8

        for _ in range(5):
            d = low >> 24
            if d >= 256:
                for idx in range(len(out) - 1, -1, -1):
                    out[idx] = (out[idx] + 1) & 255
                    if out[idx] != 0:
                        break
                d &= 255
            out.append(d)
            low = (low & 0xFFFFFF) << 8

        ml_bin = bytes(out)
        w_list = list(qw0) + list(qS1) + list(qS2) + list(qS3) + list(qWm.ravel())
        w_bytes = bytes((v & 255) for v in w_list)

        meta_info = {
            "qw0": qw0,
            "qS1": qS1,
            "qS2": qS2,
            "qS3": qS3,
            "qWm": qWm,
            "sig_pad": sig_pad,
            "sig_cut": sig_cut,
            "denom": denom,
        }
        return w_bytes, ml_bin, meta_info
