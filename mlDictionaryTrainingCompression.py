import os, re, math, time, lzma, struct, pickle, subprocess, gc
from collections import Counter
import numpy as np

DATASET_PATH = r"C:\dev\ML\ml_dataset.pkl" 
DICT_PATH = r"C:\dev\ML\dict.txt"

import heapq

def build_paq_fsm(max_states=160):
    states = [(0, 0)]; s2i = {(0, 0): 0}; q = [(0, 0)]
    RUNS = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,26,34,44,58,76,100,140,200]
    def snap(r):
        best = 0
        for x in RUNS:
            if x <= r: best = x
        return best
    def nst(c0, c1, b):
        if b == 0:
            if c1 == 0: return (snap(c0 + 1), 0)
            if c1 == 1: return (min(c0 + 1, 16), 1)
            nc0 = c0 + 1; nc1 = c1 if c1 < 4 else (c1 * 2 // 3 if c1 > 4 else 3)
        else:
            if c0 == 0: return (0, snap(c1 + 1))
            if c0 == 1: return (1, min(c1 + 1, 16))
            nc1 = c1 + 1; nc0 = c0 if c0 < 4 else (c0 * 2 // 3 if c0 > 4 else 3)
        if nc0 + nc1 > 40: nc0 = max(1, (nc0 * 3) // 4); nc1 = max(1, (nc1 * 3) // 4)
        return (nc0, nc1)
    idx = 0
    while idx < len(q) and len(states) < max_states:
        st = q[idx]; idx += 1
        for b in (0, 1):
            t = nst(st[0], st[1], b)
            if t not in s2i and len(states) < max_states:
                s2i[t] = len(states); states.append(t); q.append(t)
    NX0 = [0]*len(states); NX1 = [0]*len(states)
    for i, st in enumerate(states):
        for b, NX in ((0, NX0), (1, NX1)):
            t = nst(st[0], st[1], b)
            while t not in s2i: t = (max(0, t[0]-1), max(0, t[1]-1))
            NX[i] = s2i[t]
    return np.array(states, dtype=np.float32), np.array(NX0, dtype=np.uint8), np.array(NX1, dtype=np.uint8)

def huffman_lengths(freqs):
    K = len(freqs); pq = []; parent = [-1] * (2 * K - 1)
    for s in range(K): heapq.heappush(pq, (max(1, freqs[s]), s, s))
    uid = K
    while len(pq) > 1:
        f1, _, i1 = heapq.heappop(pq); f2, _, i2 = heapq.heappop(pq)
        parent[i1] = uid; parent[i2] = uid
        heapq.heappush(pq, (f1 + f2, uid, uid)); uid += 1
    lens = [0] * K
    for s in range(K):
        u = s; d = 0
        while parent[u] != -1: u = parent[u]; d += 1
        lens[s] = d
    return lens

def canon_tree_from_lens(lens):
    K = len(lens); L = [0]*(K-1); R = [0]*(K-1); V = [0]*K
    nxt = 1; li = 0; c = 0; pl = 0; codes = {}
    for l, s in sorted((lens[i], i) for i in range(K)):
        c <<= (l - pl); pl = l; u = 0; path = []
        for bp in range(l - 1, 0, -1):
            bit = (c >> bp) & 1; path.append((u, bit))
            if bit:
                if R[u] == 0: R[u] = nxt; nxt += 1
                u = R[u]
            else:
                if L[u] == 0: L[u] = nxt; nxt += 1
                u = L[u]
        bit = c & 1; path.append((u, bit)); leaf = K - 1 + li
        if bit: R[u] = leaf
        else: L[u] = leaf
        V[li] = s; codes[s] = path; li += 1; c += 1
    lm = [0]*(2*K - 1)
    for i in range(K): lm[K - 1 + i] = 1 << V[i]
    for i in range(K - 2, -1, -1): lm[i] = lm[L[i]] | lm[R[i]]
    return {"n_int": K - 1, "left": L, "right": R, "leaf_val": V, "leaf_mask": lm, "codes": codes, "lens": lens}

def build_alphabetic_preorder(freqs):
    K = len(freqs); w = [max(1, f) for f in freqs]; pref = [0]*(K+1)
    for i in range(K): pref[i+1] = pref[i] + w[i]
    INF = 10**18; dp = [[0]*K for _ in range(K)]; opt = [[0]*K for _ in range(K)]
    for length in range(2, K + 1):
        for i in range(K - length + 1):
            j = i + length - 1; best = INF; bk = i; sum_ij = pref[j+1] - pref[i]
            lo_k = opt[i][j-1] if length > 2 else i; hi_k = opt[i+1][j] if length > 2 else j - 1
            for k in range(max(i, lo_k), min(j-1, hi_k) + 1):
                c = dp[i][k] + dp[k+1][j] + sum_ij
                if c < best: best = c; bk = k
            dp[i][j] = best; opt[i][j] = bk
    mids = []
    def collect(i, j):
        if i == j: return
        k = opt[i][j]; mids.append(k); collect(i, k); collect(k+1, j)
    collect(0, K - 1)
    L=[]; R=[]; M=[]; V=[]; p=[0]; codes={}
    def go(a, b, path):
        if a == b:
            v = len(V); V.append(a); codes[a] = path; return (K - 1) + v
        u = len(L); k = mids[p[0]]; p[0] += 1; L.append(0); R.append(0); M.append(k)
        L[u] = go(a, k, path + [(u, 0)]); R[u] = go(k + 1, b, path + [(u, 1)])
        return u
    go(0, K - 1, [])
    return {"n_int": K - 1, "left": L, "right": R, "mids": M, "leaf_val": V, "codes": codes}

def mix(h):
    h = (h * 0x9E3779B1) & 0xFFFFFFFF
    return h ^ (h >> 16)

with open(DATASET_PATH, "rb") as f: DS = pickle.load(f)
RAW_ITEMS = DS["raw_items"]; MASKS_REM = DS["masks_rem"]
CHARS = sorted(set(c for _, s, _ in RAW_ITEMS for c in s))
C2I = {c: i for i, c in enumerate(CHARS)}
EOW = 53; BOS = 54

freq_p = [0]*20; freq_c0 = [0]*53; freq_rem = [0]*54; freq_inf_low = [0]*200
UP_INF_LIST = [k for k, m in enumerate(MASKS_REM) if m < 4]
UP_INF_MAP = {k: i for i, k in enumerate(UP_INF_LIST)}
freq_inf_up = [0]*4
prev = ""
for p, suf, inf in RAW_ITEMS:
    w = prev[:p] + suf
    freq_p[p] += 1; freq_c0[C2I[suf[0]]] += 1
    for c in suf[1:]: freq_rem[C2I[c]] += 1
    freq_rem[EOW] += 1
    if w[0].isupper(): freq_inf_up[UP_INF_MAP[inf]] += 1
    else: freq_inf_low[inf] += 1
    prev = w

tree_p = canon_tree_from_lens(huffman_lengths(freq_p))
tree_c0 = build_alphabetic_preorder(freq_c0)
tree_rem = canon_tree_from_lens(huffman_lengths(freq_rem))
tree_inf_low = canon_tree_from_lens(huffman_lengths(freq_inf_low))
tree_inf_up = canon_tree_from_lens(huffman_lengths(freq_inf_up))

OFF_P = 0; OFF_C0 = tree_p["n_int"]; OFF_REM = OFF_C0 + tree_c0["n_int"]
OFF_INF_L = OFF_REM + tree_rem["n_int"]; OFF_INF_U = OFF_INF_L + tree_inf_low["n_int"]
H0 = OFF_INF_U + tree_inf_up["n_int"]

mops_packed = bytearray()
for m in MASKS_REM:
    ops = []; mm = m
    while mm: lsb = mm & -mm; ops.append(lsb.bit_length() - 1); mm ^= lsb
    mops_packed.append(len(ops)); mops_packed.extend(ops)
mops_bytes = bytes(mops_packed)

def encode_utf8_rans(raw):
    SAFE1 = [i for i in range(1, 128) if i not in (10, 13, 34, 92)]
    W1, W2 = 1 << 14, 1 << 7; N1, N2 = 123, 1920
    R = N1 * W1 + N2 * W2 + 61440
    OFF2 = N1 * W1; OFF3 = OFF2 + N2 * W2
    L = 1 << 40
    x = L + int.from_bytes(raw[:5], "big"); di = 5; nd = len(raw); cps = []
    while True:
        q, d = divmod(x, R)
        if d < OFF2: idx, rem = divmod(d, W1); x = q * W1 + rem; s = SAFE1[idx]
        elif d < OFF3: idx, rem = divmod(d - OFF2, W2); x = q * W2 + rem; s = 128 + idx
        else: idx = d - OFF3; x = q; s = 2048 + idx if idx < 53248 else 57344 + idx - 53248
        cps.append(chr(s)); done = False
        while x < L:
            if di >= nd: done = True; break
            x = (x << 8) | raw[di]; di += 1
        if done: break
    return "".join(cps), x

def build_and_eval(D, max_fsm, out_path, epochs=165, l1_w=0.80, U_MAX=112):
    STATES, NX0, NX1 = build_paq_fsm(max_fsm)
    N_FSM = len(STATES)

    M_FS2 = 1 << 18; M_FS3 = 1 << 19
    MASK2 = M_FS2 - 1; MASK3 = M_FS3 - 1
    FS1 = np.zeros(55 * 512, dtype=np.uint8)
    FS2 = np.zeros(M_FS2 + H0, dtype=np.uint8)
    FS3 = np.zeros(M_FS3 + H0, dtype=np.uint8)

    sym_feats = []; bit_sym_idx = []; bit_nid = []; bit_s1 = []; bit_s2 = []; bit_s3 = []; bit_y = []
    nid_counts = Counter()

    pc_pad = [BOS]*5; pp = 0; pi = 2; psl = 0; plen = 0
    for p, suf, inf in RAW_ITEMS:
        w_ids = pc_pad[5 : 5 + p] + [C2I[c] for c in suf]
        wlen = len(w_ids); wc_pad = [BOS]*5 + w_ids

        vp = 0
        for k in range(min(20, plen + 1)):
            if k == plen or pc_pad[5 + k] != 52: vp |= (1 << k)
        pp_b = min(pp, 19); plen_b = min(plen, 19); psl_b = min(psl, 15)
        c_end1 = pc_pad[-1]; c_pre1 = pc_pad[5] if plen >= 1 else BOS
        c_pre2 = pc_pad[6] if plen >= 2 else BOS; c_pre3 = pc_pad[7] if plen >= 3 else BOS
        c_pre4 = pc_pad[8] if plen >= 4 else BOS; c_at_pp = pc_pad[5 + pp] if pp < plen else BOS
        sym_idx = len(sym_feats)
        sym_feats.append((c_end1, c_at_pp, c_pre3, pp_b, plen_b, psl_b))
        c1_off = pp_b << 9
        h2_b = mix((pp_b << 12) ^ (plen_b << 6) ^ c_at_pp ^ 0x1111) & MASK2
        h3_b = mix((c_pre1 << 24) ^ (c_pre2 << 18) ^ (c_pre3 << 12) ^ (c_pre4 << 6) ^ pp_b ^ 0x2222) & MASK3
        lm_p = tree_p["leaf_mask"]
        for u, b in tree_p["codes"][p]:
            if (lm_p[tree_p["left"][u]] & vp) == 0 or (lm_p[tree_p["right"][u]] & vp) == 0: continue
            nid = OFF_P + u; nid_counts[nid] += 1
            k1 = c1_off + nid; k2 = h2_b + nid; k3 = h3_b + nid
            s1 = FS1[k1]; s2 = FS2[k2]; s3 = FS3[k3]
            bit_sym_idx.append(sym_idx); bit_nid.append(nid); bit_s1.append(s1); bit_s2.append(s2); bit_s3.append(s3); bit_y.append(b)
            FS1[k1] = NX1[s1] if b else NX0[s1]; FS2[k2] = NX1[s2] if b else NX0[s2]; FS3[k3] = NX1[s3] if b else NX0[s3]

        c0_id = w_ids[p]
        lo = 0 if p == plen else pc_pad[5 + p] + 1
        lo_c = BOS if p == plen else pc_pad[5 + p]
        c_p1 = pc_pad[4 + p]; c_p2 = pc_pad[3 + p]
        sym_idx = len(sym_feats)
        sym_feats.append((c_p1, c_p2, lo_c, 20 + min(p, 19), 20 + min(plen - p, 19), 16 + (1 if p == plen else 0) + 2*(p >= pp)))
        c1_off = c_p1 << 9
        h2_b = mix((lo_c << 12) ^ (c_p1 << 6) ^ min(p, 19) ^ 0x4444) & MASK2
        h3_b = mix((c_p2 << 18) ^ (c_p1 << 12) ^ (lo_c << 6) ^ min(p, 19) ^ 0x5555) & MASK3
        sm_c0 = tree_c0["mids"]
        for u, b in tree_c0["codes"][c0_id]:
            if lo > sm_c0[u]: continue
            nid = OFF_C0 + u; nid_counts[nid] += 1
            k1 = c1_off + nid; k2 = h2_b + nid; k3 = h3_b + nid
            s1 = FS1[k1]; s2 = FS2[k2]; s3 = FS3[k3]
            bit_sym_idx.append(sym_idx); bit_nid.append(nid); bit_s1.append(s1); bit_s2.append(s2); bit_s3.append(s3); bit_y.append(b)
            FS1[k1] = NX1[s1] if b else NX0[s1]; FS2[k2] = NX1[s2] if b else NX0[s2]; FS3[k3] = NX1[s3] if b else NX0[s3]

        tokens = w_ids[p + 1:] + [EOW]
        is_up0 = (1 <= w_ids[0] <= 26)
        for pos_idx, tok in enumerate(tokens):
            idx_end = 5 + p + 1 + pos_idx
            c1 = wc_pad[idx_end - 1]; c2 = wc_pad[idx_end - 2]; c3 = wc_pad[idx_end - 3]
            cur_len = p + 1 + pos_idx
            sym_idx = len(sym_feats)
            sym_feats.append((c1, c2, c3, 40 + min(cur_len, 19), 40 + min(pos_idx + 1, 19), 32 + is_up0))
            c1_off = c1 << 9
            h2_b = mix((c2 << 6) ^ c1 ^ 0x7777) & MASK2
            h3_b = mix((c3 << 12) ^ (c2 << 6) ^ c1 ^ 0x8888) & MASK3
            for u, b in tree_rem["codes"][tok]:
                nid = OFF_REM + u; nid_counts[nid] += 1
                k1 = c1_off + nid; k2 = h2_b + nid; k3 = h3_b + nid
                s1 = FS1[k1]; s2 = FS2[k2]; s3 = FS3[k3]
                bit_sym_idx.append(sym_idx); bit_nid.append(nid); bit_s1.append(s1); bit_s2.append(s2); bit_s3.append(s3); bit_y.append(b)
                FS1[k1] = NX1[s1] if b else NX0[s1]; FS2[k2] = NX1[s2] if b else NX0[s2]; FS3[k3] = NX1[s3] if b else NX0[s3]

        c1 = wc_pad[-1]; c2 = wc_pad[-2]; c3 = wc_pad[-3]
        pi_f = 1 if pi != 2 else 0
        sym_idx = len(sym_feats)
        sym_feats.append((c1, c2, c3, 60 + min(wlen, 19), 60 + min(p, 19), 48 + (2 if is_up0 else 0) + pi_f))
        c1_off = c1 << 9
        h2_b = mix((c2 << 6) ^ c1 ^ 0xAAAA) & MASK2
        h3_b = mix((c3 << 12) ^ (c2 << 6) ^ c1 ^ 0xBBBB) & MASK3
        seq_inf = [(OFF_INF_U + u, b) for u, b in tree_inf_up["codes"][UP_INF_MAP[inf]]] if is_up0 else [(OFF_INF_L + u, b) for u, b in tree_inf_low["codes"][inf]]
        for nid, b in seq_inf:
            nid_counts[nid] += 1
            k1 = c1_off + nid; k2 = h2_b + nid; k3 = h3_b + nid
            s1 = FS1[k1]; s2 = FS2[k2]; s3 = FS3[k3]
            bit_sym_idx.append(sym_idx); bit_nid.append(nid); bit_s1.append(s1); bit_s2.append(s2); bit_s3.append(s3); bit_y.append(b)
            FS1[k1] = NX1[s1] if b else NX0[s1]; FS2[k2] = NX1[s2] if b else NX0[s2]; FS3[k3] = NX1[s3] if b else NX0[s3]

        pc_pad = wc_pad; plen = wlen; pp = p; pi = inf; psl = len(suf)

    del FS1, FS2, FS3
    globals().pop("DS", None); globals().pop("RAW_ITEMS", None)
    gc.collect()
    sf_np = np.array(sym_feats, dtype=np.uint8)
    a_sym = np.array(bit_sym_idx, dtype=np.int32)
    a_nid = np.array(bit_nid, dtype=np.uint16)
    a_s1 = np.array(bit_s1, dtype=np.uint8)
    a_s2 = np.array(bit_s2, dtype=np.uint8)
    a_s3 = np.array(bit_s3, dtype=np.uint8)
    a_y = np.array(bit_y, dtype=np.float32)
    del sym_feats, bit_sym_idx, bit_nid, bit_s1, bit_s2, bit_s3, bit_y; gc.collect()

    top_nids = [nid for nid, _ in nid_counts.most_common(U_MAX - 1)]
    del nid_counts
    nid_to_u = {nid: i for i, nid in enumerate(top_nids)}
    a_unid = np.array([nid_to_u.get(int(n), U_MAX - 1) for n in a_nid], dtype=np.uint16)
    unid_by_nid = [nid_to_u.get(n, U_MAX - 1) for n in range(H0)]
    sf0 = sf_np[:,0].copy(); sf1 = sf_np[:,1].copy(); sf2 = sf_np[:,2].copy()
    sf3 = sf_np[:,3].copy(); sf4 = sf_np[:,4].copy(); sf5 = sf_np[:,5].copy()
    S_cnt = len(sf_np)
    del sf_np; gc.collect()

    rng = np.random.default_rng(42)
    s_init = np.log((STATES[:, 1] + 0.5) / (STATES[:, 0] + 0.5)).astype(np.float32)
    params = {
        "w0": np.zeros(H0, dtype=np.float32),
        "S1": s_init.copy(), "S2": s_init.copy(), "S3": s_init.copy(),
        "Wm": np.tile(np.array([0.6, 0.7, 0.9], dtype=np.float32), (U_MAX, 1)),
    }
    for nm, shp in [("Ec0",(55,D)),("Ec1",(55,D)),("Ec2",(55,D)),("Ep0",(80,D)),("Ep1",(80,D)),("Eaux",(56,D)),("U",(U_MAX,D))]:
        params[nm] = (rng.standard_normal(shp) * 0.05).astype(np.float32)

    adam_m = {k: np.zeros_like(v) for k, v in params.items()}
    adam_v = {k: np.zeros_like(v) for k, v in params.items()}
    lr = 0.045

    f1 = np.float32(1.0); f0 = np.float32(0.0); fn20 = np.float32(-20.0); f20 = np.float32(20.0)
    for ep in range(1, epochs + 1):
        wm = params["Wm"][a_unid]
        z = params["w0"][a_nid] + wm[:,0]*params["S1"][a_s1] + wm[:,1]*params["S2"][a_s2] + wm[:,2]*params["S3"][a_s3]
        h_pre = params["Ec0"][sf0] + params["Ec1"][sf1] + params["Ec2"][sf2] + params["Ep0"][sf3] + params["Ep1"][sf4] + params["Eaux"][sf5]
        h = np.maximum(f0, h_pre)
        h_b = h[a_sym]; u_b = params["U"][a_unid]
        z += np.sum(u_b * h_b, axis=1)
        p = f1 / (f1 + np.exp(-np.clip(z, fn20, f20)))
        gz = (p - a_y).astype(np.float32)
        del z, p
        grads = {
            "w0": np.bincount(a_nid, weights=gz, minlength=H0).astype(np.float32),
            "S1": np.bincount(a_s1, weights=gz * wm[:,0], minlength=N_FSM).astype(np.float32),
            "S2": np.bincount(a_s2, weights=gz * wm[:,1], minlength=N_FSM).astype(np.float32),
            "S3": np.bincount(a_s3, weights=gz * wm[:,2], minlength=N_FSM).astype(np.float32),
        }
        del wm
        gwm = np.zeros_like(params["Wm"])
        gwm[:,0] = np.bincount(a_unid, weights=gz * params["S1"][a_s1], minlength=U_MAX)
        gwm[:,1] = np.bincount(a_unid, weights=gz * params["S2"][a_s2], minlength=U_MAX)
        gwm[:,2] = np.bincount(a_unid, weights=gz * params["S3"][a_s3], minlength=U_MAX)
        grads["Wm"] = gwm
        gU = np.zeros_like(params["U"]); gh = np.zeros((S_cnt, D), dtype=np.float32)
        for d in range(D):
            gU[:, d] = np.bincount(a_unid, weights=gz * h_b[:, d], minlength=U_MAX)
            gh[:, d] = np.bincount(a_sym, weights=gz * u_b[:, d], minlength=S_cnt)
        del h_b, u_b, gz, h
        gh *= (h_pre > 0)
        del h_pre
        for nm, idx_arr, sz in [("Ec0",sf0,55),("Ec1",sf1,55),("Ec2",sf2,55),("Ep0",sf3,80),("Ep1",sf4,80),("Eaux",sf5,56)]:
            ge = np.zeros((sz, D), dtype=np.float32)
            for d in range(D): ge[:, d] = np.bincount(idx_arr, weights=gh[:, d], minlength=sz)
            grads[nm] = ge + l1_w * np.sign(params[nm])
        del gh
        grads["U"] = gU + l1_w * np.sign(params["U"])
        del gU, gwm
        cur_lr = 0.001 + 0.5 * (lr - 0.001) * (1.0 + math.cos(math.pi * ep / epochs))
        b1t = 1.0 - 0.9 ** ep; b2t = 1.0 - 0.999 ** ep
        for k, val in params.items():
            g = grads[k]
            adam_m[k] = 0.9 * adam_m[k] + 0.1 * g
            adam_v[k] = 0.999 * adam_v[k] + 0.001 * (g * g)
            params[k] = val - cur_lr * (adam_m[k] / b1t) / (np.sqrt(adam_v[k] / b2t) + 1e-8)
        del grads

    sc_s = 8; sc_w = 16; sc_h = 16; sc_u = 8
    qw0 = np.clip(np.round(params["w0"] * sc_s), -127, 127).astype(np.int32)
    qS1 = np.clip(np.round(params["S1"] * sc_s), -127, 127).astype(np.int32)
    qS2 = np.clip(np.round(params["S2"] * sc_s), -127, 127).astype(np.int32)
    qS3 = np.clip(np.round(params["S3"] * sc_s), -127, 127).astype(np.int32)
    qWm = np.clip(np.round(params["Wm"] * sc_w), -64, 64).astype(np.int32)
    qE = {nm: np.clip(np.round(params[nm] * sc_h), -127, 127).astype(np.int32) for nm in ("Ec0","Ec1","Ec2","Ep0","Ep1","Eaux")}
    qU = np.clip(np.round(params["U"] * sc_u), -127, 127).astype(np.int32)

    wm_q = qWm[a_unid]
    fsm_q = wm_q[:,0]*qS1[a_s1] + wm_q[:,1]*qS2[a_s2] + wm_q[:,2]*qS3[a_s3]
    hq = np.maximum(0, qE["Ec0"][sf0]+qE["Ec1"][sf1]+qE["Ec2"][sf2]+qE["Ep0"][sf3]+qE["Ep1"][sf4]+qE["Eaux"][sf5])
    nnq = np.sum(qU[a_unid] * hq[a_sym], axis=1)
    logit_128 = qw0[a_nid]*sc_w + fsm_q + nnq

    sig_pad = max(2048, int(-logit_128.min()) + 2, int(logit_128.max()) + 2)
    SIG_ARR = np.clip(np.round((1.0 / (1.0 + np.exp(np.arange(-sig_pad, sig_pad + 1) / 128.0))) * 4096.0), 1, 4095).astype(np.int32)
    p_int = SIG_ARR[logit_128 + sig_pad]

    TOP = 1 << 32; BOT = 1 << 24
    low = 0; rng_e = TOP; out = bytearray()
    for b, pv in zip(a_y.astype(np.int32).tolist(), p_int.tolist()):
        r1 = (rng_e >> 12) * pv
        if b == 0: rng_e = r1
        else: low += r1; rng_e -= r1
        while rng_e < BOT:
            d = low >> 24
            if d >= 256:
                carry, d = d >> 8, d & 255; idx_c = len(out) - 1
                while carry:
                    t = out[idx_c] + carry; out[idx_c] = t & 255; carry = t >> 8; idx_c -= 1
            out.append(d); low = (low & 0xFFFFFF) << 8; rng_e <<= 8
    cur = low + (rng_e >> 1)
    d3 = cur & 255; cur >>= 8; d2 = cur & 255; cur >>= 8; d1 = cur & 255; cur >>= 8; d0 = cur & 255; carry = cur >> 8
    if carry:
        idx_c = len(out) - 1
        while carry: t = out[idx_c] + carry; out[idx_c] = t & 255; carry = t >> 8; idx_c -= 1
    out.extend([d0, d1, d2, d3, 0, 0, 0, 0, 0, 0, 0, 0])
    ml_bytes = bytes(out)

    # Byte layout of B:
    # 0..326: qw0(i8)
    # qs1(ns), qs2(ns), qs3(ns)
    # qwm(3*U_MAX)
    # Ec0(55*D), Ec1(55*D), Ec2(55*D), Ep0(80*D), Ep1(80*D), Eaux(56*D)
    # qu(U_MAX*D)
    # X0(ns), X1(ns), unid(326)
    # lens: 20+54+200+4, mids: 52, up_inf: 4, mops_bytes
    chunks = [
        qw0.astype(np.int8).tobytes(),
        qS1.astype(np.int8).tobytes(), qS2.astype(np.int8).tobytes(), qS3.astype(np.int8).tobytes(),
        qWm.astype(np.int8).tobytes(),
        qE["Ec0"].astype(np.int8).tobytes(), qE["Ec1"].astype(np.int8).tobytes(), qE["Ec2"].astype(np.int8).tobytes(),
        qE["Ep0"].astype(np.int8).tobytes(), qE["Ep1"].astype(np.int8).tobytes(), qE["Eaux"].astype(np.int8).tobytes(), qU.astype(np.int8).tobytes(),
        bytes(NX0.tolist()), bytes(NX1.tolist()),
        bytes(unid_by_nid),
        bytes(tree_p["lens"] + tree_rem["lens"] + tree_inf_low["lens"] + tree_inf_up["lens"]),
        bytes(tree_c0["mids"]),
        bytes(UP_INF_LIST),
        mops_bytes,
    ]
    bin_tables = b"".join(chunks)

    # Offsets inside B for C:
    o_s1 = 326; o_s2 = o_s1 + N_FSM; o_s3 = o_s2 + N_FSM; o_wm = o_s3 + N_FSM
    o_e = o_wm + 3 * U_MAX
    o_ec1 = o_e + 55*D; o_ec2 = o_ec1 + 55*D; o_ep0 = o_ec2 + 55*D; o_ep1 = o_ep0 + 80*D; o_eaux = o_ep1 + 80*D
    o_qu = o_eaux + 56*D
    o_x0 = o_qu + U_MAX*D; o_x1 = o_x0 + N_FSM; o_un = o_x1 + N_FSM

    u_vars = ",".join(f"u{d}" for d in range(D))
    u_dot = "+".join(f"u{d}*h{d}" for d in range(D))
    def nn_sym(f0, f1, f2, f3, f4, f5):
        a_v = ",".join(f"a{d}" for d in range(D)); b_v = ",".join(f"b{d}" for d in range(D))
        c_v = ",".join(f"cv{d}" for d in range(D)); d_v = ",".join(f"dv{d}" for d in range(D))
        e_v = ",".join(f"ev{d}" for d in range(D)); g_v = ",".join(f"gv{d}" for d in range(D))
        hs = ";".join(f"h{d}=a{d}+b{d}+cv{d}+dv{d}+ev{d}+gv{d};h{d}=0 if h{d}<0 else h{d}" for d in range(D))
        return f"{a_v}=Ec0[{f0}];{b_v}=Ec1[{f1}];{c_v}=Ec2[{f2}];{d_v}=Ep0[{f3}];{e_v}=Ep1[{f4}];{g_v}=Eaux[{f5}];{hs}"

    def step_c(nid_expr, lu, ru, ind="   "):
        return (
            f"r1,r2,r3,{u_vars}=N[{nid_expr}];k1=o1+{nid_expr};k2=k2b+{nid_expr};k3=k3b+{nid_expr};"
            f"s1=F1[k1];s2=F2[k2];s3=F3[k3];rv=(R>>12)*S[r1[s1]+r2[s2]+r3[s3]+{u_dot}]\n"
            f"{ind}if L<rv:R=rv;u={lu};F1[k1]=X0[s1];F2[k2]=X0[s2];F3[k3]=X0[s3]\n"
            f"{ind}else:L-=rv;R-=rv;u={ru};F1[k1]=X1[s1];F2[k2]=X1[s2];F3[k3]=X1[s3]\n"
            f"{ind}while R<16777216:R<<=8;L=(L<<8)|Q[di];di+=1"
        )

    c_hvars = " ".join(f"int h{d};" for d in range(D))
    c_nn_body = " ".join(
        f"h{d}=ec0[(a)*{D}+{d}]+ec1[(b)*{D}+{d}]+ec2[(c)*{D}+{d}]+ep0[(e)*{D}+{d}]+ep1[(f)*{D}+{d}]+eaux[(g)*{D}+{d}];if(h{d}<0)h{d}=0;"
        for d in range(D)
    )
    c_udot = "+".join(f"uq[{d}]*h{d}" for d in range(D))

    C_SRC = (
        f"typedef unsigned char u8;typedef signed char i8;typedef unsigned short u16;typedef unsigned int u32;typedef unsigned long long u64;\n"
        f"static u8 F1[55*512],F2[(1<<18)+326],F3[(1<<19)+326],WL[102400];\n"
        f"static char W[1050000];static u32 WO[102400],WI[102400],wp,nw;static u64 WK[102400];\n"
        f"static u32 mix(u32 h){{h*=0x9E3779B1u;return h^(h>>16);}}\n"
        f"static void addw(const char*s,int l){{u32 o=wp;for(int i=0;i<l;i++)W[wp++]=s[i];WO[nw]=o;WL[nw]=(u8)l;u64 k=0;int m=l<8?l:8;for(int i=0;i<m;i++)k|=((u64)(u8)s[i])<<(56-8*i);WK[nw]=k;WI[nw]=nw;nw++;}}\n"
        f"static int vow(char c){{return c=='a'||c=='e'||c=='i'||c=='o'||c=='u';}}\n"
        f"static int cy(const char*w,int l){{return l>1&&w[l-1]=='y'&&!vow(w[l-2]);}}\n"
        f"static int mk(char*d,const char*w,int l,int ch,const char*s,int sl){{int k=l-ch;for(int i=0;i<k;i++)d[i]=w[i];for(int i=0;i<sl;i++)d[k+i]=s[i];return k+sl;}}\n"
        f"static int mkp(char*d,const char*p,int pl,const char*w,int l){{for(int i=0;i<pl;i++)d[i]=p[i];for(int i=0;i<l;i++)d[pl+i]=w[i];return pl+l;}}\n"
        f"static int c_pl(char*d,const char*w,int l){{if(!(w[0]>='A'&&w[0]<='Z')){{if(w[l-1]=='s'||w[l-1]=='x'||w[l-1]=='z'||(l>=2&&w[l-1]=='h'&&(w[l-2]=='c'||w[l-2]=='s')))return mk(d,w,l,0,\"es\",2);if(cy(w,l))return mk(d,w,l,1,\"ies\",3);}}return mk(d,w,l,0,\"s\",1);}}\n"
        f"static int c_ed(char*d,const char*w,int l){{return w[l-1]=='e'?mk(d,w,l,0,\"d\",1):(cy(w,l)?mk(d,w,l,1,\"ied\",3):mk(d,w,l,0,\"ed\",2));}}\n"
        f"static int c_ing(char*d,const char*w,int l){{return(w[l-1]=='e'&&!(l>=2&&(w[l-2]=='e'||w[l-2]=='y'||w[l-2]=='o')))?mk(d,w,l,1,\"ing\",3):mk(d,w,l,0,\"ing\",3);}}\n"
        f"static int c_er(char*d,const char*w,int l){{return w[l-1]=='e'?mk(d,w,l,0,\"r\",1):(cy(w,l)?mk(d,w,l,1,\"ier\",3):mk(d,w,l,0,\"er\",2));}}\n"
        f"static int c_est(char*d,const char*w,int l){{return w[l-1]=='e'?mk(d,w,l,0,\"st\",2):(cy(w,l)?mk(d,w,l,1,\"iest\",4):mk(d,w,l,0,\"est\",3));}}\n"
        f"static int c_ly(char*d,const char*w,int l){{if(cy(w,l))return mk(d,w,l,1,\"ily\",3);if(l>2&&w[l-1]=='e'&&w[l-2]=='l'&&!vow(w[l-3]))return mk(d,w,l,1,\"y\",1);if(l>=2&&w[l-2]=='i'&&w[l-1]=='c')return mk(d,w,l,0,\"ally\",4);return mk(d,w,l,0,\"ly\",2);}}\n"
        f"static int c_al(char*d,const char*w,int l){{return w[l-1]=='e'?mk(d,w,l,1,\"al\",2):(w[l-1]=='y'?mk(d,w,l,1,\"ial\",3):mk(d,w,l,0,\"al\",2));}}\n"
        f"static int c_ic(char*d,const char*w,int l){{return(w[l-1]=='e'||w[l-1]=='y')?mk(d,w,l,1,\"ic\",2):mk(d,w,l,0,\"ic\",2);}}\n"
        f"static int c_ab(char*d,const char*w,int l){{return(w[l-1]=='e'&&!(l>=2&&(w[l-2]=='e'||w[l-2]=='c'||w[l-2]=='g')))?mk(d,w,l,1,\"able\",4):mk(d,w,l,0,\"able\",4);}}\n"
        f"static void c_app(const char*w,int l,int op){{char b1[64],b2[64];\n"
        f"if(op>=37){{static const char*P[]={{\"un\",\"re\",\"dis\",\"in\",\"de\",\"over\",\"mis\",\"non\",\"pre\",\"out\",\"under\",\"sub\",\"im\",\"en\",\"be\"}};const char*q=P[op-37];int ql=0;while(q[ql])ql++;addw(b1,mkp(b1,q,ql,w,l));return;}}\n"
        f"if(op==0){{addw(b1,mk(b1,w,l,0,\"'s\",2));return;}}if(op==1){{addw(b1,c_pl(b1,w,l));return;}}if(op==2){{addw(b1,c_ed(b1,w,l));return;}}if(op==3){{addw(b1,c_ing(b1,w,l));return;}}\n"
        f"if(op==4){{int n=c_er(b1,w,l);addw(b1,n);addw(b2,mk(b2,b1,n,0,\"'s\",2));addw(b2,c_pl(b2,b1,n));return;}}\n"
        f"if(op==5){{addw(b1,c_er(b1,w,l));return;}}if(op==6){{addw(b1,c_est(b1,w,l));return;}}if(op==7){{addw(b1,c_ly(b1,w,l));return;}}\n"
        f"if(op==8){{int n=cy(w,l)?mk(b1,w,l,1,\"iness\",5):mk(b1,w,l,0,\"ness\",4);addw(b1,n);addw(b2,mk(b2,b1,n,0,\"'s\",2));return;}}\n"
        f"if(op==9){{int n=mk(b1,w,l,w[l-1]=='e',\"ion\",3);addw(b1,n);addw(b2,mk(b2,b1,n,0,\"'s\",2));addw(b2,c_pl(b2,b1,n));return;}}\n"
        f"if(op==10){{int n=mk(b1,w,l,0,\"ment\",4);addw(b1,n);addw(b2,mk(b2,b1,n,0,\"'s\",2));addw(b2,c_pl(b2,b1,n));return;}}\n"
        f"if(op==11){{addw(b1,c_ab(b1,w,l));return;}}\n"
        f"if(op<15){{int x=op-12;int n=x==0?c_pl(b1,w,l):(x==1?c_ed(b1,w,l):c_ing(b1,w,l));addw(b2,mkp(b2,\"un\",2,b1,n));return;}}\n"
        f"if(op<18){{int x=op-15;int n=x==0?c_pl(b1,w,l):(x==1?c_ed(b1,w,l):c_ing(b1,w,l));addw(b2,mkp(b2,\"re\",2,b1,n));return;}}\n"
        f"if(op==18){{int n=c_pl(b1,w,l);addw(b2,mkp(b2,\"dis\",3,b1,n));return;}}\n"
        f"if(op==19){{int n=mk(b1,w,l,(cy(w,l)||w[l-1]=='e'),\"ity\",3);addw(b1,n);addw(b2,mk(b2,b1,n,0,\"'s\",2));addw(b2,c_pl(b2,b1,n));return;}}\n"
        f"if(op==20){{int n=mk(b1,w,l,w[l-1]=='e',\"ize\",3);addw(b1,n);addw(b2,c_pl(b2,b1,n));addw(b2,c_ed(b2,b1,n));addw(b2,c_ing(b2,b1,n));return;}}\n"
        f"if(op<27){{char dc=w[l-1];int dl=mk(b1,w,l,0,&dc,1);if(op==23){{int n=mk(b2,b1,dl,0,\"er\",2);addw(b2,n);addw(b1,mk(b1,b2,n,0,\"'s\",2));addw(b1,c_pl(b1,b2,n));return;}}static const char*S6[]={{\"ed\",\"ing\",\"\",\"er\",\"est\",\"y\"}};const char*s=S6[op-21];int sl=0;while(s[sl])sl++;addw(b2,mk(b2,b1,dl,0,s,sl));return;}}\n"
        f"if(op==27){{addw(b1,mk(b1,w,l,w[l-1]=='e',\"y\",1));return;}}if(op==28){{addw(b1,c_al(b1,w,l));return;}}\n"
        f"if(op==29){{int n=c_al(b1,w,l);addw(b1,n);addw(b2,c_ly(b2,b1,n));return;}}if(op==30){{addw(b1,c_ic(b1,w,l));return;}}\n"
        f"if(op==31){{int n=c_ic(b1,w,l);addw(b1,n);addw(b2,mk(b2,b1,n,0,\"al\",2));addw(b2,mk(b2,b1,n,0,\"ally\",4));return;}}\n"
        f"if(op<34){{int n=mk(b1,w,l,w[l-1]=='e',op==32?\"ive\":\"ous\",3);addw(b1,n);addw(b2,c_ly(b2,b1,n));return;}}\n"
        f"if(op==34){{int n=c_ing(b1,w,l);addw(b2,mk(b2,b1,n,0,\"ly\",2));return;}}if(op==35){{int n=c_ab(b1,w,l);addw(b2,mk(b2,b1,n,1,\"y\",1));return;}}\n"
        f"int n=c_ly(b1,w,l);addw(b2,mkp(b2,\"un\",2,b1,n));}}\n"
        f"static int cmpw(u32 a,u32 b){{if(WK[a]!=WK[b])return WK[a]<WK[b]?-1:1;int la=WL[a],lb=WL[b],m=la<lb?la:lb;const char*pa=W+WO[a],*pb=W+WO[b];for(int i=8;i<m;i++)if(pa[i]!=pb[i])return(u8)pa[i]<(u8)pb[i]?-1:1;return la<lb?-1:(la>lb?1:0);}}\n"
        f"static void qs(int l,int r){{while(l<r){{int i=l,j=r;u32 p=WI[(l+r)>>1];while(i<=j){{while(cmpw(WI[i],p)<0)i++;while(cmpw(WI[j],p)>0)j--;if(i<=j){{u32 t=WI[i];WI[i]=WI[j];WI[j]=t;i++;j--;}}}}if(j-l<r-i){{if(l<j)qs(l,j);l=i;}}else{{if(i<r)qs(i,r);r=j;}}}}}}\n"
        f"#define NN(a,b,c,e,f,g) {c_nn_body}\n"
        f"#define ST(nid,lu,ru) nd=(nid);ui=un[nd];k1=o1+nd;k2=k2b+nd;k3=k3b+nd;s1=F1[k1];s2=F2[k2];s3=F3[k3];wm=qwm+3*ui;uq=bqu+{D}*ui;"
        f"rv=(R>>12)*S[B[nd]*16+{sig_pad}+wm[0]*qs1[s1]+wm[1]*qs2[s2]+wm[2]*qs3[s3]+{c_udot}];"
        f"if(L<rv){{R=rv;u=(lu);F1[k1]=x0[s1];F2[k2]=x0[s2];F3[k3]=x0[s3];}}"
        f"else{{L-=rv;R-=rv;u=(ru);F1[k1]=x1[s1];F2[k2]=x1[s2];F3[k3]=x1[s3];}}"
        f"while(R<16777216ULL){{R<<=8;L=(L<<8)|Q[di++];}}\n"
        f"int dec(const u8*Q,const i8*B,const u16*T,const u16*S,const u8*MP,u8*O){{\n"
        f"wp=0;nw=0;u32 mo[200];int mq=0;for(int i=0;i<200;i++){{mo[i]=mq;mq+=1+MP[mq];}}\n"
        f"const char*CH=\"'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\";\n"
        f"const i8*qs1=B+{o_s1},*qs2=B+{o_s2},*qs3=B+{o_s3},*qwm=B+{o_wm},*ec0=B+{o_e},*ec1=B+{o_ec1},*ec2=B+{o_ec2},*ep0=B+{o_ep0},*ep1=B+{o_ep1},*eaux=B+{o_eaux},*bqu=B+{o_qu};\n"
        f"const u8*x0=(const u8*)B+{o_x0},*x1=(const u8*)B+{o_x1},*un=(const u8*)B+{o_un};\n"
        f"const u16*lp=T,*rp=T+19,*vp=T+38,*lr=T+58,*rr=T+111,*vr=T+164,*lil=T+218,*ril=T+417,*vil=T+616,*liu=T+816,*riu=T+819,*viu=T+822,*lc0=T+826,*rc0=T+878,*mc0=T+930,*vc0=T+982,*upi=T+1035;\n"
        f"u32 lmp[39];for(int i=0;i<20;i++)lmp[19+i]=1u<<vp[i];for(int i=18;i>=0;i--)lmp[i]=lmp[lp[i]]|lmp[rp[i]];\n"
        f"u64 R=1ULL<<32,L=((u64)Q[0]<<24)|((u64)Q[1]<<16)|((u64)Q[2]<<8)|Q[3],rv;\n"
        f"u32 di=4,pc[64],wc[64],o1,k2b,k3b,k1,k2,k3,vpm;char WB[64],PB[64];int pp=0,pi=2,psl=0,plen=0;\n"
        f"for(int i=0;i<5;i++)pc[i]=54;\n"
        f"for(int _w=0;_w<35771;_w++){{\n"
        f"int ppb=pp<19?pp:19,plb=plen<19?plen:19,psb=psl<15?psl:15;vpm=0;\n"
        f"int km=plen+1>20?20:plen+1;for(int k=0;k<km;k++)if(k==plen||pc[5+k]!=52)vpm|=1u<<k;\n"
        f"u32 ce1=pc[5+plen-1],cp1=plen>=1?pc[5]:54,cp2=plen>=2?pc[6]:54,cp3=plen>=3?pc[7]:54,cp4=plen>=4?pc[8]:54,capp=pp<plen?pc[5+pp]:54;\n"
        f"{c_hvars}const i8*wm,*uq;int nd,ui,s1,s2,s3,u;\n"
        f"NN(ce1,capp,cp3,ppb,plb,psb)\n"
        f"o1=ppb<<9;k2b=mix((ppb<<12)^(plb<<6)^capp^0x1111u)&((1<<18)-1);k3b=mix((cp1<<24)^(cp2<<18)^(cp3<<12)^(cp4<<6)^ppb^0x2222u)&((1<<19)-1);\n"
        f"u=0;while(u<19){{u16 cl=lp[u],cr=rp[u];if(!(lmp[cl]&vpm)){{u=cr;continue;}}if(!(lmp[cr]&vpm)){{u=cl;continue;}}ST(u,cl,cr)}}\n"
        f"int p=vp[u-19];\n"
        f"int lo=p==plen?0:pc[5+p]+1,loc=p==plen?54:pc[5+p],x1c=pc[4+p],x2c=pc[3+p],pb19=p<19?p:19,lp19=(plen-p)<19?(plen-p):19;\n"
        f"NN(x1c,x2c,loc,20+pb19,20+lp19,16+(p==plen)+(p>=pp)*2)\n"
        f"o1=x1c<<9;k2b=mix((loc<<12)^(x1c<<6)^pb19^0x4444u)&((1<<18)-1);k3b=mix((x2c<<18)^(x1c<<12)^(loc<<6)^pb19^0x5555u)&((1<<19)-1);\n"
        f"u=0;while(u<52){{if(lo>mc0[u]){{u=rc0[u];continue;}}ST(19+u,lc0[u],rc0[u])}}\n"
        f"int c0i=vc0[u-52];\n"
        f"for(int i=0;i<5+p;i++)wc[i]=pc[i];wc[5+p]=c0i;\n"
        f"for(int i=0;i<p;i++)WB[i]=PB[i];WB[p]=CH[c0i];\n"
        f"int up0=(wc[5]>=1&&wc[5]<=26),wlen=p+1;\n"
        f"u32 c1=c0i,c2=wc[5+p-1],c3=wc[5+p-2];int cln=p+1,pi2=1;\n"
        f"for(;;){{\n"
        f"int cl19=cln<19?cln:19,ps19=pi2<19?pi2:19;\n"
        f"NN(c1,c2,c3,40+cl19,40+ps19,32+up0)\n"
        f"o1=c1<<9;k2b=mix((c2<<6)^c1^0x7777u)&((1<<18)-1);k3b=mix((c3<<12)^(c2<<6)^c1^0x8888u)&((1<<19)-1);\n"
        f"u=0;while(u<53){{ST(71+u,lr[u],rr[u])}}\n"
        f"int tok=vr[u-53];if(tok==53)break;\n"
        f"wc[5+cln++]=tok;WB[wlen++]=CH[tok];c3=c2;c2=c1;c1=tok;pi2++;\n"
        f"}}\n"
        f"int sl=pi2,pif=pi!=2,wl19=cln<19?cln:19;\n"
        f"NN(c1,c2,c3,60+wl19,60+pb19,48+up0*2+pif)\n"
        f"o1=c1<<9;k2b=mix((c2<<6)^c1^0xAAAAu)&((1<<18)-1);k3b=mix((c3<<12)^(c2<<6)^c1^0xBBBBu)&((1<<19)-1);\n"
        f"int lm=up0?3:199,bo=up0?323:124;const u16*lt=up0?liu:lil,*rt=up0?riu:ril;\n"
        f"u=0;while(u<lm){{ST(bo+u,lt[u],rt[u])}}\n"
        f"int inf=up0?upi[viu[u-3]]:vil[u-199];\n"
        f"addw(WB,wlen);const u8*mp=MP+mo[inf];int nk=mp[0];for(int k=1;k<=nk;k++)c_app(WB,wlen,mp[k]);\n"
        f"for(int i=0;i<5+wlen;i++)pc[i]=wc[i];for(int i=0;i<wlen;i++)PB[i]=WB[i];plen=wlen;pp=p;pi=inf;psl=sl;\n"
        f"}}qs(0,(int)nw-1);int oi=0;for(u32 i=0;i<nw;i++){{u32 id=WI[i];int l=WL[id];const char*s=W+WO[id];for(int j=0;j<l;j++)O[oi++]=(u8)s[j];O[oi++]='\\n';}}return oi;}}\n"
    )

    INNER_CODE = f'''def solve(Q,B,pr,sp_f):
 import struct,sys,math,ctypes
 ns={N_FSM}
 def ct(ln):
  K=len(ln);L=[0]*(K-1);R=[0]*(K-1);V=[0]*K;nt=1;li=c=pl=0
  for l,s in sorted((ln[i],i) for i in range(K)):
   c<<=l-pl;pl=l;u=0
   for bp in range(l-1,0,-1):
    if(c>>bp)&1:
     if R[u]==0:R[u]=nt;nt+=1
     u=R[u]
    else:
     if L[u]==0:L[u]=nt;nt+=1
     u=L[u]
   if c&1:R[u]=K-1+li
   else:L[u]=K-1+li
   V[li]=s;li+=1;c+=1
  return L,R,V
 def bc0(m):
  L=[];R=[];M=[];V=[];p=[0]
  def go(a,b):
   if a==b:v=len(V);V.append(a);return 52+v
   u=len(L);k=m[p[0]];p[0]+=1;L.append(0);R.append(0);M.append(k);L[u]=go(a,k);R[u]=go(k+1,b);return u
  go(0,52);return L,R,M,V
 o_t={o_un+326}
 lp,rp,vp=ct(B[o_t:o_t+20]);lr,rr,vr=ct(B[o_t+20:o_t+74]);lil,ril,vil=ct(B[o_t+74:o_t+274]);liu,riu,viu=ct(B[o_t+274:o_t+278])
 lc0,rc0,mc0,vc0=bc0(B[o_t+278:o_t+330]);up_inf=B[o_t+330:o_t+334]
 sp={sig_pad};S=tuple(min(4095,max(1,round(4096.0/(1.0+math.exp(x/128.0))))) for x in range(-sp,sp+1))
 try:
  if not pr or pr.wait():raise OSError
  Ta=lp+rp+vp+lr+rr+vr+lil+ril+vil+liu+riu+viu+lc0+rc0+mc0+vc0+list(up_inf)
  T=struct.pack(f"{{len(Ta)}}H",*Ta)
  Sb=struct.pack(f"{{len(S)}}H",*S)
  O=bytearray(1000000);ob=(ctypes.c_uint8*1000000).from_buffer(O)
  lib=ctypes.CDLL(sp_f)
  fn=lib.dec;fn.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_char_p,ctypes.c_char_p,ctypes.c_char_p,ctypes.c_void_p]
  n=fn(Q,B[:o_t],T,Sb,B[o_t+334:],ob)
  sys.stdout.buffer.write(memoryview(O)[:n]);return
 except Exception:
  o=o_t+334;MOPS=[];D=[]
  for _ in range(200):k=B[o];MOPS.append(tuple(B[o+1:o+1+k]));o+=1+k
  PRE="un re dis in de over mis non pre out under sub im en be".split()
  def cy(w):return len(w)>1 and w[-1]=="y" and w[-2] not in "aeiou"
  def pl(w):
   if not w[0].isupper():
    if w.endswith(("s","x","z","ch","sh")):return w+"es"
    if cy(w):return w[:-1]+"ies"
   return w+"s"
  def ed(w):return w+"d" if w[-1]=="e" else((w[:-1]+"ied") if cy(w) else w+"ed")
  def ing(w):return(w[:-1]+"ing") if(w[-1]=="e" and not w.endswith(("ee","ye","oe"))) else w+"ing"
  def er(w):return w+"r" if w[-1]=="e" else((w[:-1]+"ier") if cy(w) else w+"er")
  def est(w):return w+"st" if w[-1]=="e" else((w[:-1]+"iest") if cy(w) else w+"est")
  def ly(w):
   if cy(w):return w[:-1]+"ily"
   if w.endswith("le") and len(w)>2 and w[-3] not in "aeiou":return w[:-1]+"y"
   return w+"ally" if w.endswith("ic") else w+"ly"
  def al(w):return(w[:-1]+"al") if w[-1]=="e" else((w[:-1]+"ial") if w[-1]=="y" else w+"al")
  def ic(w):return(w[:-1]+"ic") if w[-1] in "ey" else w+"ic"
  def ab(w):return(w[:-1]+"able") if(w[-1]=="e" and not w.endswith(("ee","ce","ge"))) else w+"able"
  def app(w,op):
   if op>=37:return(PRE[op-37]+w,)
   if op<4:return(w+"\'s",pl(w),ed(w),ing(w))[op:op+1]
   if op==4:e=er(w);return(e,e+"\'s",pl(e))
   if op<8:return(er(w),est(w),ly(w))[op-5:op-4]
   if op==8:n=(w[:-1]+"iness") if cy(w) else w+"ness";return(n,n+"\'s")
   if op==9:j=(w[:-1] if w[-1]=="e" else w)+"ion";return(j,j+"\'s",pl(j))
   if op==10:t=w+"ment";return(t,t+"\'s",pl(t))
   if op==11:return(ab(w),)
   if op<15:return("un"+(pl,ed,ing)[op-12](w),)
   if op<18:return("re"+(pl,ed,ing)[op-15](w),)
   if op==18:return("dis"+pl(w),)
   if op==19:t=(w[:-1] if(cy(w) or w[-1]=="e") else w)+"ity";return(t,t+"\'s",pl(t))
   if op==20:z=(w[:-1] if w[-1]=="e" else w)+"ize";return(z,pl(z),ed(z),ing(z))
   if op<27:
    d=w+w[-1]
    if op==23:e=d+"er";return(e,e+"\'s",pl(e))
    return(d+("ed","ing","","er","est","y")[op-21],)
   if op==27:return((w[:-1] if w[-1]=="e" else w)+"y",)
   if op==28:return(al(w),)
   if op==29:a=al(w);return(a,ly(a))
   if op==30:return(ic(w),)
   if op==31:z=ic(w);return(z,z+"al",z+"ally")
   if op<34:z=(w[:-1] if w[-1]=="e" else w)+("ive" if op==32 else "ous");return(z,ly(z))
   return(ing(w)+"ly",) if op==34 else((ab(w)[:-1]+"y",) if op==35 else("un"+ly(w),))
  def i8(o,n):return struct.unpack_from(f"{{n}}b",B,o),o+n
  o=0;qw0,o=i8(o,326);qs1,o=i8(o,ns);qs2,o=i8(o,ns);qs3,o=i8(o,ns);qwm,o=i8(o,{3*U_MAX})
  Es=[]
  for k in(55,55,55,80,80,56):a,o=i8(o,k*{D});Es.append(tuple(a[i*{D}:i*{D}+{D}] for i in range(k)))
  Ec0,Ec1,Ec2,Ep0,Ep1,Eaux=Es;qu,o=i8(o,{U_MAX*D});X0=B[o:o+ns];X1=B[o+ns:o+2*ns];o+=2*ns;unid=B[o:o+326]
  lmp=[0]*39
  for i in range(20):lmp[19+i]=1<<vp[i]
  for i in range(18,-1,-1):lmp[i]=lmp[lp[i]]|lmp[rp[i]]
  w0b=[qw0[n]*16+sp for n in range(326)]
  R2=[tuple(qwm[ui*3+1]*qs2[s] for s in range(ns)) for ui in range({U_MAX})]
  R3=[tuple(qwm[ui*3+2]*qs3[s] for s in range(ns)) for ui in range({U_MAX})]
  N=tuple((tuple(w0b[nid]+qwm[unid[nid]*3]*qs1[s] for s in range(ns)),R2[unid[nid]],R3[unid[nid]],*qu[unid[nid]*{D}:unid[nid]*{D}+{D}]) for nid in range(326))
  CH="\'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";VPA=tuple((1<<(i+1 if i<19 else 20))-1 for i in range(20))
  M1=0x9E3779B1;M9=0xFFFFFFFF;F1=bytearray(55*512);M2=(1<<18)-1;M3=(1<<19)-1;F2=bytearray((1<<18)+326);F3=bytearray((1<<19)+326)
  R=1<<32;L=(Q[0]<<24)|(Q[1]<<16)|(Q[2]<<8)|Q[3];di=4
  pc=[54,54,54,54,54];pp=0;pi=2;psl=0;plen=0;prev=""
  for _w in range(35771):
   pp_b=pp if pp<19 else 19;plen_b=plen if plen<19 else 19;psl_b=psl if psl<15 else 15
   if "z" not in prev[:20]:vpm=VPA[plen_b]
   else:
    vpm=0
    for k in range(20 if plen+1>20 else plen+1):
     if k==plen or pc[5+k]!=52:vpm|=1<<k
   ce1=pc[-1];cp1=pc[5] if plen>=1 else 54;cp2=pc[6] if plen>=2 else 54;cp3=pc[7] if plen>=3 else 54;cp4=pc[8] if plen>=4 else 54;capp=pc[5+pp] if pp<plen else 54
   {nn_sym("ce1","capp","cp3","pp_b","plen_b","psl_b")}
   o1=pp_b<<9;t=((pp_b<<12)^(plen_b<<6)^capp^0x1111)*M1&M9;k2b=(t^(t>>16))&M2;t=((cp1<<24)^(cp2<<18)^(cp3<<12)^(cp4<<6)^pp_b^0x2222)*M1&M9;k3b=(t^(t>>16))&M3
   u=0
   while u<19:
    cl=lp[u];cr=rp[u]
    if(lmp[cl]&vpm)==0:u=cr;continue
    if(lmp[cr]&vpm)==0:u=cl;continue
    {step_c("u","cl","cr","    ")}
   p=vp[u-19]
   lo=0 if p==plen else pc[5+p]+1;loc=54 if p==plen else pc[5+p]
   x1=pc[4+p];x2=pc[3+p];pb19=p if p<19 else 19;lp19=(plen-p) if plen-p<19 else 19
   {nn_sym("x1","x2","loc","20+pb19","20+lp19","16+(1 if p==plen else 0)+2*(p>=pp)")}
   o1=x1<<9;t=((loc<<12)^(x1<<6)^pb19^0x4444)*M1&M9;k2b=(t^(t>>16))&M2;t=((x2<<18)^(x1<<12)^(loc<<6)^pb19^0x5555)*M1&M9;k3b=(t^(t>>16))&M3
   u=0
   while u<52:
    if lo>mc0[u]:u=rc0[u];continue
    nid=19+u
    {step_c("nid","lc0[u]","rc0[u]","    ")}
   c0i=vc0[u-52]
   wc=pc[:5+p];wc.append(c0i);sc=[CH[c0i]];up0=(1<=wc[5]<=26)
   c1=c0i;c2=wc[-2];c3=wc[-3];cln=p+1;pi2=1
   while True:
    cl19=cln if cln<19 else 19;ps19=pi2 if pi2<19 else 19
    {nn_sym("c1","c2","c3","40+cl19","40+ps19","32+up0")}
    o1=c1<<9;t=((c2<<6)^c1^0x7777)*M1&M9;k2b=(t^(t>>16))&M2;t=((c3<<12)^(c2<<6)^c1^0x8888)*M1&M9;k3b=(t^(t>>16))&M3
    u=0
    while u<53:
     nid=71+u
     {step_c("nid","lr[u]","rr[u]","     ")}
    tok=vr[u-53]
    if tok==53:break
    wc.append(tok);sc.append(CH[tok]);c3=c2;c2=c1;c1=tok;cln+=1;pi2+=1
   pi_f=1 if pi!=2 else 0;wl19=cln if cln<19 else 19
   {nn_sym("c1","c2","c3","60+wl19","60+pb19","48+(2 if up0 else 0)+pi_f")}
   o1=c1<<9;t=((c2<<6)^c1^0xAAAA)*M1&M9;k2b=(t^(t>>16))&M2;t=((c3<<12)^(c2<<6)^c1^0xBBBB)*M1&M9;k3b=(t^(t>>16))&M3
   if up0:lm,bo,lt,rt=3,323,liu,riu
   else:lm,bo,lt,rt=199,124,lil,ril
   u=0
   while u<lm:
    nid=bo+u
    {step_c("nid","lt[u]","rt[u]","    ")}
   inf=up_inf[viu[u-3]] if up0 else vil[u-199]
   w=prev[:p]+"".join(sc);D.append(w)
   for op in MOPS[inf]:D.extend(app(w,op))
   pc=wc;prev=w;plen=cln;pp=p;pi=inf;psl=pi2
 D.sort()
 sys.stdout.buffer.write(("\\n".join(D)+"\\n").encode("ascii"))
'''

    cs_lz = lzma.compress(C_SRC.encode("utf-8"), format=lzma.FORMAT_ALONE, filters=[{"id": lzma.FILTER_LZMA1, "preset": 9, "dict_size": 1 << 17}])
    cs_str, x_cs = encode_utf8_rans(cs_lz)

    hdr_raw = struct.pack("<H", len(INNER_CODE)) + INNER_CODE.encode("utf-8") + bin_tables
    hdr_lz = lzma.compress(hdr_raw, format=lzma.FORMAT_ALONE, filters=[{"id": lzma.FILTER_LZMA1, "preset": 9, "dict_size": 1 << 17}])

    full_bin = struct.pack("<H", len(hdr_lz)) + hdr_lz + ml_bytes
    u_str, x_end = encode_utf8_rans(full_bin)

    OUTER = (
        "# -*- coding: utf-8 -*-\n"
        "import gc,lzma,os,subprocess,sys,tempfile;gc.disable()\n"
        f"U=\"{u_str}\"\n"
        f"CS=\"{cs_str}\"\n"
        "C=[0]*128;k=0\n"
        "for i in range(1,128):\n"
        " if i not in(10,13,34,92):C[i]=k<<14;k+=1\n"
        "def D(s,x):\n"
        " R=2322432;O2=2015232;O3=2260992;U0=(1<<40)//R;B1=U0<<22;B2=U0<<15;B3=U0<<8;o=bytearray();ap=o.append\n"
        " for c in map(ord,s[::-1]):\n"
        "  if c<128:\n"
        "   while x>=B1:ap(x&255);x>>=8\n"
        "   x=(x>>14)*R+C[c]+(x&16383)\n"
        "  elif c<2048:\n"
        "   while x>=B2:ap(x&255);x>>=8\n"
        "   x=(x>>7)*R+O2+((c-128)<<7)+(x&127)\n"
        "  else:\n"
        "   while x>=B3:ap(x&255);x>>=8\n"
        "   x=x*R+O3+(c-2048 if c<55296 else c-4096)\n"
        " o.extend((x-(1<<40)).to_bytes(5,'big')[::-1]);return bytes(o[::-1])\n"
        "sp_f=os.path.join(tempfile.gettempdir(),'dc_'+str(os.getpid())+('.dll' if os.name=='nt' else '.so'));pr=None\n"
        "try:\n"
        f" cb=lzma.decompress(D(CS,{x_cs}))\n"
        " pr=subprocess.Popen(['gcc','-O0','-pipe','-shared']+([] if os.name=='nt' else ['-fPIC'])+['-x','c','-','-o',sp_f],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
        " pr.stdin.write(cb);pr.stdin.close()\n"
        "except Exception:pr=None\n"
        f"A=D(U,{x_end})\n"
        "hl=A[0]|(A[1]<<8)\n"
        "B=lzma.decompress(A[2:2+hl])\n"
        "L=B[0]|(B[1]<<8);ns={}\n"
        "exec(B[2:2+L],ns)\n"
        "ns['solve'](A[2+hl:],B[2+L:],pr,sp_f)\n"
    )
    with open(out_path, "wb") as f:
        f.write(OUTER.encode("utf-8"))

    sz = os.path.getsize(out_path)
    print(f"\n=== UTF-8 C+Py D={D} N_FSM={N_FSM} U={U_MAX} | ml_bin={len(ml_bytes):,}B hdr_lz={len(hdr_lz):,}B (code_raw={len(INNER_CODE):,}) bin_tot={len(full_bin):,}B => UTF-8 FILE={sz:,} B ({sz/1024:.2f} KiB) ===", flush=True)

    del sf0, sf1, sf2, sf3, sf4, sf5, a_sym, a_nid, a_s1, a_s2, a_s3, a_y, a_unid, p_int, logit_128, fsm_q, hq, nnq, params, adam_m, adam_v
    gc.collect()

    with open(DICT_PATH, "rb") as f: EXP_BYTES = f.read()
    times = []
    for r in range(6):
        t0 = time.perf_counter()
        res = subprocess.run(["python", out_path], capture_output=True, timeout=5)
        dt = (time.perf_counter() - t0) * 1000
        times.append(dt)
        ok = (res.returncode == 0 and res.stdout == EXP_BYTES)
        print(f"  Run {r+1}: {dt:.1f} ms | rc={res.returncode} | exact_match={ok}", flush=True)
        if not ok:
            print("  ERR:", res.stderr.decode("utf-8", "replace")[:400], flush=True)
            return
    st = sorted(times)
    print(f"  -> min={min(times):.1f} ms | median={(st[2]+st[3])/2:.1f} ms | max={max(times):.1f} ms", flush=True)

build_and_eval(3, 160, r"C:\dev\ML\solve.py", epochs=165, l1_w=0.80, U_MAX=112)
