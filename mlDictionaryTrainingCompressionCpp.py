import os, re, math, time, struct, pickle, subprocess, gc
from collections import Counter, defaultdict
import heapq
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = HERE if os.path.exists(os.path.join(HERE, "dict.txt")) else r"C:\dev\ML"
DATASET_PATH = os.path.join(BASE_DIR, "ml_dataset.pkl")
DICT_PATH = os.path.join(BASE_DIR, "dict.txt")
OUT_CPP = os.path.join(BASE_DIR, "solve.cpp")
OUT_EXE = os.path.join(os.getenv("TEMP") or "/tmp", f"solve_cpp_{os.getpid()}" + (".exe" if os.name == "nt" else ""))

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
N_BASE = len(RAW_ITEMS); assert MASKS_REM[2] == 0
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

def encode_utf8_rans_nobidi(raw):
    # Safe 1-byte ASCII inside R"!(...)!" (124): 1..127 except \n(10), \r(13), )(41)
    # Safe 2-byte non-bidi (1296): U+0080..U+058F
    # Safe 3-byte non-bidi (59648): U+0900..U+1FFF (5888) + U+2100..U+D7FF (46848) + U+E000..U+FAFF (6912)
    SAFE1 = [i for i in range(1, 128) if i not in (10, 13, 41)]
    W1, W2 = 1 << 14, 1 << 7
    N1, N2, N3 = 124, 1296, 59648
    R = N1 * W1 + N2 * W2 + N3
    OFF2 = N1 * W1; OFF3 = OFF2 + N2 * W2
    L = 1 << 40
    x = L + int.from_bytes(raw[:5], "big"); di = 5; nd = len(raw); cps = []
    while True:
        q, d = divmod(x, R)
        if d < OFF2:
            idx, rem = divmod(d, W1); x = q * W1 + rem; s = SAFE1[idx]
        elif d < OFF3:
            idx, rem = divmod(d - OFF2, W2); x = q * W2 + rem; s = 0x0080 + idx
        else:
            idx = d - OFF3; x = q
            s = (0x0900 + idx) if idx < 5888 else ((0x2100 + idx - 5888) if idx < 52736 else (0xE000 + idx - 52736))
        cps.append(chr(s)); done = False
        while x < L:
            if di >= nd: done = True; break
            x = (x << 8) | raw[di]; di += 1
        if done: break
    return "".join(cps), x

def rans_enc(syms, M=2048):
    n = len(syms); fc = Counter(syms); used = sorted(fc.keys())
    fq = {s: max(1, int(round(fc[s] * M / n))) for s in used}
    cs = sum(fq.values())
    while cs > M: b = max(used, key=lambda s: fq[s] if fq[s] > 1 else -1); fq[b] -= 1; cs -= 1
    while cs < M: b = max(used, key=lambda s: fc[s]*M/n - fq[s]); fq[b] += 1; cs += 1
    cum = {}; pos = 0
    for s in used: cum[s] = pos; pos += fq[s]
    h = bytearray([len(used)&255, len(used)>>8, n&255, n>>8])
    ps = 0
    for s in used:
        d = s - ps; ps = s; f = fq[s]
        if d < 255: h.append(d)
        else: h.append(255); h.append(d - 255)
        if f < 240: h.append(f)
        else: h.append(240 + (f >> 8)); h.append(f & 255)
    L_b = 1 << 16; mbits = M.bit_length() - 1
    x = L_b; bs = bytearray()
    for s in reversed(syms):
        f = fq[s]; c = cum[s]; xm = ((L_b >> mbits) << 8) * f
        while x >= xm: bs.append(x & 255); x >>= 8
        x = (x // f) * M + (x % f) + c
    for _ in range(4): bs.append(x & 255); x >>= 8
    return bytes(h) + bytes(bs[::-1])

def dp_lzss_enc(data, prefix=b""):
    full = prefix + data; off = len(prefix); n = len(full)
    last = defaultdict(list); matches = [[] for _ in range(n)]
    for i in range(n - 2):
        h = full[i:i+3]; lst = last[h]
        if i >= off:
            for prev in reversed(lst[-128:]):
                d = i - prev
                if d > 32767: break
                l = 3
                while l < 288 and i + l < n and full[prev + l] == full[i + l]: l += 1
                matches[i].append((d, l))
        lst.append(i)
    lit_c = [8.0]*256; len_c = [9.0]*32; d_c = [6.0]*16
    for it in range(4):
        dp = [1e9]*(n + 1); dp[n] = 0.0; choice = [None]*n
        for i in range(n - 1, off - 1, -1):
            best = lit_c[full[i]] + dp[i + 1]; bc = (0, full[i])
            for d, l_max in matches[i]:
                db = (d - 1).bit_length()
                dcost = (d_c[db] + max(0, db - 1)) if it > 0 else (5.0 + max(0, db - 1))
                for l in range(3, min(33, l_max) + 1):
                    sc = len_c[l - 3] + dcost + dp[i + l]
                    if sc < best: best = sc; bc = (1, d, l)
                if l_max > 33:
                    sc = len_c[31] + 8.0 + dcost + dp[i + l_max]
                    if sc < best: best = sc; bc = (1, d, l_max)
            dp[i] = best; choice[i] = bc
        syms = []; dbkts = []; i = off
        while i < n:
            ch = choice[i]
            if ch[0] == 0: syms.append(ch[1]); i += 1
            else:
                _, d, l = ch
                syms.append(256 + min(31, l - 3))
                dbkts.append((d - 1).bit_length()); i += l
        fc = Counter(syms); tot = len(syms)
        for s in range(256): lit_c[s] = -math.log2((fc.get(s,0)+0.5)/(tot+128))
        for s in range(32): len_c[s] = -math.log2((fc.get(256+s,0)+0.5)/(tot+128))
        dc = Counter(dbkts); td = max(1, len(dbkts))
        for b in range(16): d_c[b] = -math.log2((dc.get(b,0)+0.5)/(td+8))
    toks = []; dbkts = []; lex = bytearray(); bits = []; i = off
    while i < n:
        ch = choice[i]
        if ch[0] == 0: toks.append(ch[1]); i += 1
        else:
            _, d, l = ch
            toks.append(256 + min(31, l - 3))
            if l >= 34: lex.append(l - 34)
            db = (d - 1).bit_length(); dbkts.append(db); eb = max(0, db - 1)
            if eb:
                val = d - (1 + (1 << eb))
                for b in range(eb): bits.append((val >> b) & 1)
            i += l
    ex_b = bytearray((len(bits) + 7) // 8)
    for idx, b in enumerate(bits):
        if b: ex_b[idx >> 3] |= (1 << (idx & 7))
    b1 = rans_enc(toks, 2048); b2 = rans_enc(dbkts, 256)
    return len(b1).to_bytes(2,'little') + len(b2).to_bytes(2,'little') + len(lex).to_bytes(2,'little') + b1 + b2 + bytes(lex) + bytes(ex_b)

D = 3; max_fsm = 160; epochs = 190; l1_w = 0.95; U_MAX = 112
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

del FS1, FS2, FS3, DS, RAW_ITEMS
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

w_bytes = b"".join([
    qw0.astype(np.int8).tobytes(),
    qS1.astype(np.int8).tobytes(), qS2.astype(np.int8).tobytes(), qS3.astype(np.int8).tobytes(),
    qWm.astype(np.int8).tobytes(),
    qE["Ec0"].astype(np.int8).tobytes(), qE["Ec1"].astype(np.int8).tobytes(), qE["Ec2"].astype(np.int8).tobytes(),
    qE["Ep0"].astype(np.int8).tobytes(), qE["Ep1"].astype(np.int8).tobytes(), qE["Eaux"].astype(np.int8).tobytes(), qU.astype(np.int8).tobytes(),
])
meta_bytes = b"".join([
    bytes(NX0.tolist()), bytes(NX1.tolist()),
    bytes(unid_by_nid),
    bytes(tree_p["lens"] + tree_rem["lens"] + tree_inf_low["lens"] + tree_inf_up["lens"]),
    bytes(tree_c0["mids"]),
    bytes(UP_INF_LIST),
    mops_bytes,
])

o_s1 = 326; o_s2 = o_s1 + N_FSM; o_s3 = o_s2 + N_FSM; o_wm = o_s3 + N_FSM
o_e = o_wm + 3 * U_MAX
o_ec1 = o_e + 55*D; o_ec2 = o_ec1 + 55*D; o_ep0 = o_ec2 + 55*D; o_ep1 = o_ep0 + 80*D; o_eaux = o_ep1 + 80*D
o_qu = o_eaux + 56*D
w_len = o_qu + U_MAX*D
assert len(w_bytes) == w_len

w_b = rans_enc(list(w_bytes), 2048)
m_b = dp_lzss_enc(meta_bytes)
full_bin = struct.pack("<2H", len(w_b), len(m_b)) + w_b + m_b + ml_bytes
u_str, x_end = encode_utf8_rans_nobidi(full_bin)

del sf0, sf1, sf2, sf3, sf4, sf5, a_sym, a_nid, a_s1, a_s2, a_s3, a_y, a_unid, p_int, logit_128, fsm_q, hq, nnq, params, adam_m, adam_v
gc.collect()

# Split u_str cleanly across 2 literals (each < 50 KB) to avoid MSVC string limits if any
mid = len(u_str) // 2
u1, u2 = u_str[:mid], u_str[mid:]

cpp_src = f"""#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif
using namespace std;using u8=uint8_t;using i8=int8_t;using u16=uint16_t;using u32=uint32_t;using u64=uint64_t;
#define F(i,n) for(int i=0;i<(n);i++)
#define Z memcpy
#define X(a,b) ((a)<(b)?(a):(b))
#define K(x) (mix(x)&262143)
#define J(x) (mix(x)&524287)
const char*U=R"!({u1})!" R"!({u2})!";
u8 F1[55*512],F2[262470],F3[524614],WL[102400],MB[4096],BX[80000];i8 WB[4096];
char W[1050000],OT[1000000];u16 S[{2*sig_pad+1}],WV[4096],DB[4096];u64 WK[102400];
u32 RT[2048],CP[80000],WO[102400],WI[102400],wp,nw,lp[19],rp[19],vp[20],lr[53],rr[53],vr[54],lil[199],ril[199],vil[200],liu[3],riu[3],viu[4],lc0[52],rc0[52],mc0[52],vc0[53],mo[200];
u32 mix(u32 h){{return h*=0x9E3779B1u,h^(h>>16);}}
int rd(const u8*B,int bl,int M,u16*out){{int nu=B[0]|(B[1]<<8),nl=B[2]|(B[3]<<8),o=4,ps=0,mb=M==2048?11:8,mk=M-1,tn=0;F(i,nu){{int d=B[o++];if(d==255)d+=B[o++];int s=ps+d;ps=s;int f=B[o++];if(f>=240)f=((f-240)<<8)|B[o++];u32 hi=((u32)s<<(mb+12))|((u32)f<<mb);F(j,f)RT[tn++]=hi+j;}}u32 x=((u32)B[o]<<24)|((u32)B[o+1]<<16)|((u32)B[o+2]<<8)|B[o+3];o+=4;F(i,nl){{u32 v=RT[x&mk];x=((v>>mb)&4095)*(x>>mb)+(v&mk);while(x<65536&&o<bl)x=(x<<8)|B[o++];out[i]=v>>(mb+12);}}return nl;}}
void dlz(const u8*B,u8*out){{int l1=B[0]|(B[1]<<8),l2=B[2]|(B[3]<<8),lx=B[4]|(B[5]<<8),o=6,nt=rd(B+o,l1,2048,WV);o+=l1;rd(B+o,l2,256,DB);o+=l2;const u8*lxb=B+o,*ex=B+o+lx;int di=0,li=0,bp=0,n=0;F(i,nt){{u16 s=WV[i];if(s<256)out[n++]=(u8)s;else{{int l=s==287?34+lxb[li++]:3+s-256,b=DB[di++],d;if(b<2)d=b+1;else{{int eb=b-1;d=1+(1<<eb);F(k,eb){{if((ex[bp>>3]>>(bp&7))&1)d+=(1<<k);bp++;}}}}F(k,l)out[n+k]=out[n-d+k];n+=l;}}}}}}
void ct(const u8*ln,int q,u32*L,u32*R,u32*V){{u32 nt=1,li=0,c=0,pl=0;for(int l=1;l<=32;l++)F(s,q)if(ln[s]==l){{c<<=(l-pl);pl=l;u32 u=0;for(int bp=l-1;bp>0;bp--){{if((c>>bp)&1){{if(!R[u])R[u]=nt++;u=R[u];}}else{{if(!L[u])L[u]=nt++;u=L[u];}}}}if(c&1)R[u]=q-1+li;else L[u]=q-1+li;V[li++]=s;c++;}}}}
u32 bc0(const u8*m,int a,int b,int&p,int&v){{if(a==b){{vc0[v++]=a;return 52+v-1;}}u32 id=p;int k=m[p++];mc0[id]=k;lc0[id]=bc0(m,a,k,p,v);rc0[id]=bc0(m,k+1,b,p,v);return id;}}
void G(const char*s,int l){{u32 o=wp;Z(W+wp,s,l);wp+=l;WO[nw]=o;WL[nw]=(u8)l;u64 k=0;int m=X(l,8);F(i,m)k|=((u64)(u8)s[i])<<(56-8*i);WK[nw]=k;WI[nw]=nw;nw++;}}
int V(char c){{return c=='a'||c=='e'||c=='i'||c=='o'||c=='u';}}
int Y(const char*w,int l){{return l>1&&w[l-1]=='y'&&!V(w[l-2]);}}
int M(char*d,const char*w,int l,int ch,const char*s,int sl){{int k=l-ch;Z(d,w,k);Z(d+k,s,sl);return k+sl;}}
int MP_(char*d,const char*p,int pl,const char*w,int l){{Z(d,p,pl);Z(d+pl,w,l);return pl+l;}}
#define _A(x,y) M(d,w,l,x,y,sizeof(y)-1)
int PL(char*d,const char*w,int l){{if(!(w[0]>='A'&&w[0]<='Z')){{if(w[l-1]=='s'||w[l-1]=='x'||w[l-1]=='z'||(l>=2&&w[l-1]=='h'&&(w[l-2]=='c'||w[l-2]=='s')))return _A(0,"es");if(Y(w,l))return _A(1,"ies");}}return _A(0,"s");}}
int ED(char*d,const char*w,int l){{return w[l-1]=='e'?_A(0,"d"):(Y(w,l)?_A(1,"ied"):_A(0,"ed"));}}
int IN(char*d,const char*w,int l){{return(w[l-1]=='e'&&!(l>=2&&(w[l-2]=='e'||w[l-2]=='y'||w[l-2]=='o')))?_A(1,"ing"):_A(0,"ing");}}
int ER(char*d,const char*w,int l){{return w[l-1]=='e'?_A(0,"r"):(Y(w,l)?_A(1,"ier"):_A(0,"er"));}}
int ES(char*d,const char*w,int l){{return w[l-1]=='e'?_A(0,"st"):(Y(w,l)?_A(1,"iest"):_A(0,"est"));}}
int LY(char*d,const char*w,int l){{if(Y(w,l))return _A(1,"ily");if(l>2&&w[l-1]=='e'&&w[l-2]=='l'&&!V(w[l-3]))return _A(1,"y");if(l>=2&&w[l-2]=='i'&&w[l-1]=='c')return _A(0,"ally");return _A(0,"ly");}}
int AL(char*d,const char*w,int l){{return w[l-1]=='e'?_A(1,"al"):(w[l-1]=='y'?_A(1,"ial"):_A(0,"al"));}}
int IC(char*d,const char*w,int l){{return(w[l-1]=='e'||w[l-1]=='y')?_A(1,"ic"):_A(0,"ic");}}
int AB(char*d,const char*w,int l){{return(w[l-1]=='e'&&!(l>=2&&(w[l-2]=='e'||w[l-2]=='c'||w[l-2]=='g')))?_A(1,"able"):_A(0,"able");}}
void AW(const char*w,int l,int f){{G(w,l);char b[64];if(f&1)G(b,M(b,w,l,0,"'s",2));if(f&2)G(b,PL(b,w,l));}}
#define _B(a,c,s) M(a,w,l,c,s,sizeof(s)-1)
#define _C(n,x) if(op==n){{x;return;}}
void AP(const char*w,int l,int op){{char a[64],b[64];
if(op>=37){{const char*P[]={{"un","re","dis","in","de","over","mis","non","pre","out","under","sub","im","en","be"}};const char*q=P[op-37];G(a,MP_(a,q,(int)strlen(q),w,l));return;}}
_C(0,G(a,_B(a,0,"'s")))_C(1,G(a,PL(a,w,l)))_C(2,G(a,ED(a,w,l)))_C(3,G(a,IN(a,w,l)))_C(4,AW(a,ER(a,w,l),3))_C(5,G(a,ER(a,w,l)))_C(6,G(a,ES(a,w,l)))_C(7,G(a,LY(a,w,l)))
_C(8,AW(a,Y(w,l)?_B(a,1,"iness"):_B(a,0,"ness"),1))_C(9,AW(a,_B(a,w[l-1]=='e',"ion"),3))_C(10,AW(a,_B(a,0,"ment"),3))_C(11,G(a,AB(a,w,l)))
if(op<18){{int f=op<15,x=op-(f?12:15),n=x==0?PL(a,w,l):(x==1?ED(a,w,l):IN(a,w,l));G(b,MP_(b,f?"un":"re",2,a,n));return;}}
_C(18,G(b,MP_(b,"dis",3,a,PL(a,w,l))))_C(19,AW(a,_B(a,(Y(w,l)||w[l-1]=='e'),"ity"),3))
_C(20,int n=_B(a,w[l-1]=='e',"ize");G(a,n);G(b,PL(b,a,n));G(b,ED(b,a,n));G(b,IN(b,a,n)))
if(op<27){{char dc=w[l-1];int dl=M(a,w,l,0,&dc,1);_C(23,AW(b,M(b,a,dl,0,"er",2),3))const char*S6[]={{"ed","ing","","er","est","y"}};const char*s=S6[op-21];G(b,M(b,a,dl,0,s,(int)strlen(s)));return;}}
_C(27,G(a,_B(a,w[l-1]=='e',"y")))_C(28,G(a,AL(a,w,l)))_C(29,int n=AL(a,w,l);G(a,n);G(b,LY(b,a,n)))_C(30,G(a,IC(a,w,l)))
_C(31,int n=IC(a,w,l);G(a,n);G(b,M(b,a,n,0,"al",2));G(b,M(b,a,n,0,"ally",4)))
if(op<34){{int n=_B(a,w[l-1]=='e',op==32?"ive":"ous");G(a,n);G(b,LY(b,a,n));return;}}
_C(34,G(b,M(b,a,IN(a,w,l),0,"ly",2)))_C(35,G(b,M(b,a,AB(a,w,l),1,"y",1)))G(b,MP_(b,"un",2,a,LY(a,w,l)));
}}
int cw(u32 a,u32 b){{if(WK[a]!=WK[b])return WK[a]<WK[b]?-1:1;int la=WL[a],lb=WL[b],m=X(la,lb);const char*pa=W+WO[a],*pb=W+WO[b];for(int i=8;i<m;i++)if(pa[i]!=pb[i])return(u8)pa[i]<(u8)pb[i]?-1:1;return la<lb?-1:(la>lb?1:0);}}
void qs(int l,int r){{while(l<r){{int i=l,j=r;u32 p=WI[(l+r)>>1];u64 pk=WK[p];while(i<=j){{while(WK[WI[i]]<pk||(WK[WI[i]]==pk&&cw(WI[i],p)<0))i++;while(WK[WI[j]]>pk||(WK[WI[j]]==pk&&cw(WI[j],p)>0))j--;if(i<=j){{u32 t=WI[i];WI[i]=WI[j];WI[j]=t;i++;j--;}}}}if(j-l<r-i){{if(l<j)qs(l,j);l=i;}}else{{if(i<r)qs(i,r);r=j;}}}}}}
#define NN(a,b,c,e,f,g,o,x,y) {{const i8*pa=ec0+(a)*3,*pb=ec1+(b)*3,*pc_=ec2+(c)*3,*pe=ep0+(e)*3,*pf=ep1+(f)*3,*pg=eaux+(g)*3;h0=pa[0]+pb[0]+pc_[0]+pe[0]+pf[0]+pg[0];if(h0<0)h0=0;h1=pa[1]+pb[1]+pc_[1]+pe[1]+pf[1]+pg[1];if(h1<0)h1=0;h2=pa[2]+pb[2]+pc_[2]+pe[2]+pf[2]+pg[2];if(h2<0)h2=0;o1=(o);k2b=K(x);k3b=J(y);}}
#define ST(nid,lu,ru) nd=(nid);ui=un[nd];k1=o1+nd;k2=k2b+nd;k3=k3b+nd;s1=F1[k1];s2=F2[k2];s3=F3[k3];wm=qwm+3*ui;uq=bqu+3*ui;rv=(R>>12)*S[w0p[nd]+wm[0]*qs1[s1]+wm[1]*qs2[s2]+wm[2]*qs3[s3]+uq[0]*h0+uq[1]*h1+uq[2]*h2];if(L<rv){{R=rv;u=(lu);F1[k1]=x0[s1];F2[k2]=x0[s2];F3[k3]=x0[s3];}}else{{L-=rv;R-=rv;u=(ru);F1[k1]=x1[s1];F2[k2]=x1[s2];F3[k3]=x1[s3];}}while(R<16777216ULL){{R<<=8;L=(L<<8)|Qd[di++];}}
int main(){{
#ifdef _WIN32
_setmode(_fileno(stdout),_O_BINARY);
#endif
u32 C[128]={{}};int ck=0,cn=0,an=0;F(i,128)if(i&&i!=10&&i!=13&&i!=41)C[i]=ck++<<14;
for(auto s=(const u8*)U;*s;)CP[cn++]=*s<128?*s++:(*s<224?(s+=2,((s[-2]&31)<<6)|(s[-1]&63)):(s+=3,((s[-3]&15)<<12)|((s[-2]&63)<<6)|(s[-1]&63)));
u64 R=2257152,O2=2031616,O3=2197504,U0=(1ULL<<40)/R,B1=U0<<22,B2=U0<<15,B3=U0<<8,x={x_end}ULL;
for(int i=cn-1;i>=0;i--){{u32 c=CP[i];if(c<128){{while(x>=B1){{BX[an++]=x&255;x>>=8;}}x=(x>>14)*R+C[c]+(x&16383);}}else if(c<2048){{while(x>=B2){{BX[an++]=x&255;x>>=8;}}x=(x>>7)*R+O2+((c-128)<<7)+(x&127);}}else{{while(x>=B3){{BX[an++]=x&255;x>>=8;}}x=x*R+O3+(c<0x2100?c-0x0900:(c<0xE000?c-0x2100+5888:c-0xE000+52736));}}}}
u64 t=x-(1ULL<<40);F(i,5){{BX[an++]=t&255;t>>=8;}}F(i,an/2){{u8 z=BX[i];BX[i]=BX[an-1-i];BX[an-1-i]=z;}}
int lw=BX[0]|(BX[1]<<8),lm=BX[2]|(BX[3]<<8),nwv=rd(BX+4,lw,2048,WV);F(i,nwv)WB[i]=(i8)(u8)WV[i];
dlz(BX+4+lw,MB);const u8*Qd=BX+4+lw+lm,*x0=MB,*x1=MB+{N_FSM},*un=x1+{N_FSM},*ln=un+326,*m0=ln+278,*upi=m0+52,*MP=upi+4;
ct(ln,20,lp,rp,vp);ct(ln+20,54,lr,rr,vr);ct(ln+74,200,lil,ril,vil);ct(ln+274,4,liu,riu,viu);int p0=0,v0=0;bc0(m0,0,52,p0,v0);
F(i,{2*sig_pad+1}){{double v=rint(4096.0/(1.0+exp((i-{sig_pad})/128.0)));S[i]=(u16)(v<1?1:(v>4095?4095:(int)v));}}
int mq=0;F(i,200){{mo[i]=mq;mq+=1+MP[mq];}}int w0p[326];F(i,326)w0p[i]=WB[i]*16+{sig_pad};
const char*CH="'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const i8*qs1=WB+{o_s1},*qs2=WB+{o_s2},*qs3=WB+{o_s3},*qwm=WB+{o_wm},*ec0=WB+{o_e},*ec1=WB+{o_ec1},*ec2=WB+{o_ec2},*ep0=WB+{o_ep0},*ep1=WB+{o_ep1},*eaux=WB+{o_eaux},*bqu=WB+{o_qu};
u32 lmp[39];F(i,20)lmp[19+i]=1u<<vp[i];for(int i=18;i>=0;i--)lmp[i]=lmp[lp[i]]|lmp[rp[i]];
R=1ULL<<32;u64 L=((u64)Qd[0]<<24)|((u64)Qd[1]<<16)|((u64)Qd[2]<<8)|Qd[3],rv;
u32 di=4,pc[64],wc[64],o1,k2b,k3b,k1,k2,k3,vpm;char Wb[64],PB[64];int pp=0,pi=2,psl=0,plen=0;F(i,5)pc[i]=54;
F(_w,{N_BASE}){{int ppb=X(pp,19),plb=X(plen,19),psb=X(psl,15);vpm=0;int km=X(plen+1,20);F(k,km)if(k==plen||pc[5+k]!=52)vpm|=1u<<k;u32 ce1=pc[5+plen-1],cp1=plen>=1?pc[5]:54,cp2=plen>=2?pc[6]:54,cp3=plen>=3?pc[7]:54,cp4=plen>=4?pc[8]:54,capp=pp<plen?pc[5+pp]:54;int h0,h1,h2;const i8*wm,*uq;int nd,ui,s1,s2,s3,u;NN(ce1,capp,cp3,ppb,plb,psb,ppb<<9,(ppb<<12)^(plb<<6)^capp^0x1111u,(cp1<<24)^(cp2<<18)^(cp3<<12)^(cp4<<6)^ppb^0x2222u)u=0;while(u<19){{u16 cl=lp[u],cr=rp[u];if(!(lmp[cl]&vpm)){{u=cr;continue;}}if(!(lmp[cr]&vpm)){{u=cl;continue;}}ST(u,cl,cr)}}int p=vp[u-19],lo=p==plen?0:pc[5+p]+1,loc=p==plen?54:pc[5+p],x1c=pc[4+p],x2c=pc[3+p],pb19=X(p,19),lp19=X(plen-p,19);NN(x1c,x2c,loc,20+pb19,20+lp19,16+(p==plen)+(p>=pp)*2,x1c<<9,(loc<<12)^(x1c<<6)^pb19^0x4444u,(x2c<<18)^(x1c<<12)^(loc<<6)^pb19^0x5555u)u=0;while(u<52){{if(lo>(int)mc0[u]){{u=rc0[u];continue;}}ST(19+u,lc0[u],rc0[u])}}int c0i=vc0[u-52];F(i,5+p)wc[i]=pc[i];wc[5+p]=c0i;Z(Wb,PB,p);Wb[p]=CH[c0i];int up0=(wc[5]>=1&&wc[5]<=26),wlen=p+1;u32 c1=c0i,c2=wc[5+p-1],c3=wc[5+p-2];int cln=p+1,pi2=1;for(;;){{int cl19=X(cln,19),ps19=X(pi2,19);NN(c1,c2,c3,40+cl19,40+ps19,32+up0,c1<<9,(c2<<6)^c1^0x7777u,(c3<<12)^(c2<<6)^c1^0x8888u)u=0;while(u<53){{ST(71+u,lr[u],rr[u])}}int tok=vr[u-53];if(tok==53)break;wc[5+cln++]=tok;Wb[wlen++]=CH[tok];c3=c2;c2=c1;c1=tok;pi2++;}}int sl=pi2,pif=pi!=2,wl19=X(cln,19);NN(c1,c2,c3,60+wl19,60+pb19,48+up0*2+pif,c1<<9,(c2<<6)^c1^0xAAAAu,(c3<<12)^(c2<<6)^c1^0xBBBBu)int lm2=up0?3:199,bo=up0?323:124;const u32*lt=up0?liu:lil,*rt=up0?riu:ril;u=0;while(u<lm2){{ST(bo+u,lt[u],rt[u])}}int inf=up0?upi[viu[u-3]]:vil[u-199];G(Wb,wlen);const u8*mp=MP+mo[inf];int nk=mp[0];for(int k=1;k<=nk;k++)AP(Wb,wlen,mp[k]);F(i,5+wlen)pc[i]=wc[i];Z(PB,Wb,wlen);plen=wlen;pp=p;pi=inf;psl=sl;}}
qs(0,(int)nw-1);int oi=0;F(i,(int)nw){{u32 id=WI[i];int l=WL[id];Z(OT+oi,W+WO[id],l);oi+=l;OT[oi++]='\\n';}}fwrite(OT,1,oi,stdout);
}}
"""

with open(OUT_CPP, "wb") as f:
    f.write(cpp_src.encode("utf-8"))

cpp_sz = os.path.getsize(OUT_CPP)
print(f"Wrote {OUT_CPP}: {cpp_sz:,} B ({cpp_sz/1024:.2f} KiB) | raw={len(full_bin):,}B (w_b={len(w_b):,}B m_b={len(m_b):,}B ml_bin={len(ml_bytes):,}B)")

t_c0 = time.perf_counter()
res_c = subprocess.run(["g++", "-O2", "-std=c++20", "-Wall", "-Wextra", "-Wbidi-chars=any", OUT_CPP, "-o", OUT_EXE], capture_output=True, text=True)
dt_c = (time.perf_counter() - t_c0)*1000
print(f"g++ rc={res_c.returncode} compile={dt_c:.1f}ms | stderr:", res_c.stderr.strip() or "(none)")
if res_c.returncode == 0:
    exp = open(DICT_PATH, "rb").read().replace(b"\r\n", b"\n")
    ts = []
    for r in range(6):
        t0 = time.perf_counter()
        rr = subprocess.run([OUT_EXE], capture_output=True, timeout=5)
        dt = (time.perf_counter() - t0)*1000
        ts.append(dt)
        ok = (rr.returncode == 0 and rr.stdout == exp)
        print(f"  Run {r+1}: {dt:.1f} ms | rc={rr.returncode} | exact={ok}")
    st = sorted(ts)
    print(f"  -> min={min(ts):.1f} ms | median={(st[2]+st[3])/2:.1f} ms | max={max(ts):.1f} ms")
if os.path.exists(OUT_EXE): os.remove(OUT_EXE)
