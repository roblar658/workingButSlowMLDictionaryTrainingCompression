import gc, math, os, pickle, time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = HERE if os.path.exists(os.path.join(HERE, "dict.txt")) else r"C:\dev\ML"
DICT_PATH = os.path.join(BASE_DIR, "dict.txt")
OUT_PATH = os.path.join(BASE_DIR, "ml_dataset.pkl")

with open(DICT_PATH, "r", encoding="ascii") as f:
    words = f.read().splitlines()
ws = set(words)

def is_cy(w): return len(w) >= 2 and w[-1] == "y" and w[-2] not in "aeiou"
def plural(w):
    if not w[0].isupper():
        if w.endswith(("s", "x", "z", "ch", "sh")): return w + "es"
        if is_cy(w): return w[:-1] + "ies"
    return w + "s"
def ed(w):
    if w.endswith("e"): return w + "d"
    if is_cy(w): return w[:-1] + "ied"
    return w + "ed"
def ing(w):
    if w.endswith("e") and not w.endswith(("ee", "ye", "oe")): return w[:-1] + "ing"
    return w + "ing"
def er(w):
    if w.endswith("e"): return w + "r"
    if is_cy(w): return w[:-1] + "ier"
    return w + "er"
def est(w):
    if w.endswith("e"): return w + "st"
    if is_cy(w): return w[:-1] + "iest"
    return w + "est"
def ly(w):
    if is_cy(w): return w[:-1] + "ily"
    if w.endswith("le") and len(w) > 2 and w[-3] not in "aeiou": return w[:-1] + "y"
    if w.endswith("ic"): return w + "ally"
    return w + "ly"
def ness(w): return (w[:-1] + "iness") if is_cy(w) else (w + "ness")
def ion(w): return (w[:-1] + "ion") if w.endswith("e") else (w + "ion")
def able(w): return (w[:-1] + "able") if (w.endswith("e") and not w.endswith(("ee", "ce", "ge"))) else (w + "able")
def ity(w): return (w[:-1] + "ity") if (is_cy(w) or w.endswith("e")) else (w + "ity")
def ize(w): return (w[:-1] + "ize") if w.endswith("e") else (w + "ize")
def al_fn(w):
    if w.endswith("e"): return w[:-1] + "al"
    if w.endswith("y"): return w[:-1] + "ial"
    return w + "al"
def ic_fn(w): return (w[:-1] + "ic") if w.endswith(("e", "y")) else (w + "ic")
def y_fn(w): return (w[:-1] + "y") if w.endswith("e") else (w + "y")
def ive_fn(w): return (w[:-1] + "ive") if w.endswith("e") else (w + "ive")
def ous_fn(w): return (w[:-1] + "ous") if w.endswith("e") else (w + "ous")

PRE_WORDS = "un re dis in de over mis non pre out under sub im en be".split()
def apply_op(w, op):
    if op >= 37: return (PRE_WORDS[op - 37] + w,)
    if op == 0: return (w + "'s",)
    if op == 1: return (plural(w),)
    if op == 2: return (ed(w),)
    if op == 3: return (ing(w),)
    if op == 4: e = er(w); return (e, e + "'s", plural(e))
    if op == 5: return (er(w),)
    if op == 6: return (est(w),)
    if op == 7: return (ly(w),)
    if op == 8: n = ness(w); return (n, n + "'s")
    if op == 9: j = ion(w); return (j, j + "'s", plural(j))
    if op == 10: t = w + "ment"; return (t, t + "'s", plural(t))
    if op == 11: return (able(w),)
    if op == 12: return ("un" + plural(w),)
    if op == 13: return ("un" + ed(w),)
    if op == 14: return ("un" + ing(w),)
    if op == 15: return ("re" + plural(w),)
    if op == 16: return ("re" + ed(w),)
    if op == 17: return ("re" + ing(w),)
    if op == 18: return ("dis" + plural(w),)
    if op == 19: it = ity(w); return (it, it + "'s", plural(it))
    if op == 20: iz = ize(w); return (iz, plural(iz), ed(iz), ing(iz))
    if op == 21: return (w + w[-1] + "ed",)
    if op == 22: return (w + w[-1] + "ing",)
    if op == 23: e = w + w[-1] + "er"; return (e, e + "'s", plural(e))
    if op == 24: return (w + w[-1] + "er",)
    if op == 25: return (w + w[-1] + "est",)
    if op == 26: return (w + w[-1] + "y",)
    if op == 27: return (y_fn(w),)
    if op == 28: return (al_fn(w),)
    if op == 29: a = al_fn(w); return (a, ly(a))
    if op == 30: return (ic_fn(w),)
    if op == 31: icw = ic_fn(w); return (icw, icw + "al", icw + "ally")
    if op == 32: iv = ive_fn(w); return (iv, ly(iv))
    if op == 33: ou = ous_fn(w); return (ou, ly(ou))
    if op == 34: return (ing(w) + "ly",)
    if op == 35: return (able(w)[:-1] + "y",)
    if op == 36: return ("un" + ly(w),)
    return ()

N_OPS = 52
def gen_derivs(w, mask):
    out = []; m = mask
    while m: lsb = m & -m; out.extend(apply_op(w, lsb.bit_length() - 1)); m ^= lsb
    return out

# Phase 1: Discover top-200 morphological operation masks + greedy initialization
t0 = time.perf_counter()
full_masks = {}
for w in words:
    if w[0].isupper():
        full_masks[w] = (1 if (w + "'s") in ws else 0) | (2 if (w + "s") in ws else 0)
        continue
    m = 0
    for op in range(N_OPS):
        if op == 5 and (m & 16): continue
        if op == 24 and (m & (1 << 23)): continue
        ds = apply_op(w, op)
        if ds and len(set(ds)) == len(ds) and all(d in ws and d != w for d in ds):
            if op == 29: m &= ~(1 << 28)
            if op == 31: m &= ~(1 << 30)
            m |= (1 << op)
    full_masks[w] = m

word_power = {w: len(gen_derivs(w, full_masks[w])) for w in words}
c_masks = Counter(full_masks.values())
scored = sorted([(cnt * len(gen_derivs("testate", m)), m) for m, cnt in c_masks.items() if m], reverse=True)
pool = [0] + [m for _, m in scored[:1500]]
for op in range(N_OPS):
    if (1 << op) not in pool: pool.append(1 << op)

def solve_with(masks_list, power_map, pen_thresh=2, pen_w=1.0):
    cand_order = sorted([(i, m) for i, m in enumerate(masks_list) if m], key=lambda x: (len(gen_derivs("testate", x[1])), -x[0]), reverse=True)
    claimed = set(); inf_of = {}
    for w in sorted(words, key=len):
        if w in claimed: continue
        fm = full_masks[w]
        best_inf = 0; best_ds = None; best_sc = -1e9
        if fm:
            for inf_id, m in cand_order:
                if (m & fm) != m: continue
                if w[0].isupper() and (m & ~3): continue
                ds = gen_derivs(w, m)
                ds_set = set(ds)
                if len(ds_set) != len(ds): continue
                ok = True
                for d in ds:
                    if d in claimed or ((d + "'s") in ws and (d + "'s") not in claimed and (d + "'s") not in ds_set):
                        ok = False; break
                if not ok: continue
                sc = len(ds) - sum(pen_w * power_map[d] for d in ds if power_map[d] >= pen_thresh)
                if sc > best_sc: best_sc = sc; best_inf = inf_id; best_ds = ds
                if best_sc >= len(ds) - 0.1: break
        if best_ds and best_sc > 0: claimed.update(best_ds); inf_of[w] = best_inf
        else: inf_of[w] = 0
        claimed.add(w)
    return inf_of

inf_p = solve_with(pool, word_power, pen_thresh=2, pen_w=1.0)
pwr = {w: (len(gen_derivs(w, pool[inf_p[w]])) if w in inf_p else 0) for w in words}
del inf_p, word_power; gc.collect()
inf_p2 = solve_with(pool, pwr, pen_thresh=1, pen_w=0.95)
used_m = Counter(pool[i] for i in inf_p2.values() if i)
del inf_p2, pool, scored, c_masks; gc.collect()

NM = 200
tm = [0] + [m for m, _ in used_m.most_common(NM - 1)]
del used_m; gc.collect()
inf_o1 = solve_with(tm, pwr, pen_thresh=2, pen_w=1.0)
pwr2 = {w: (len(gen_derivs(w, tm[inf_o1[w]])) if w in inf_o1 else 0) for w in words}
del inf_o1, pwr; gc.collect()
inf_o2 = solve_with(tm, pwr2, pen_thresh=1, pen_w=0.95)
u2 = Counter(tm[i] for i in inf_o2.values() if i)
del inf_o2; gc.collect()
tm = [0] + [m for m, _ in u2.most_common()]
del u2; gc.collect()
inf_o = solve_with(tm, pwr2, pen_thresh=1, pen_w=0.95)
del pwr2, full_masks; gc.collect()

base_greedy = sorted(inf_o.keys())
i_s_greedy = [inf_o[w] for w in base_greedy]
freq_g = [k for k, _ in Counter(i_s_greedy).most_common()]
remap_g = {old: new for new, old in enumerate(freq_g)}
MASKS = [tm[k] for k in freq_g]
RAW_OLD = [(0, w, remap_g[x]) for w, x in zip(base_greedy, i_s_greedy)]

# Phase 2: Exact Tree DP + Bitmask DAG DP + local neighborhood Bitmask DP optimization
full_m = {}
for w in words:
    if w[0].isupper():
        full_m[w] = (1 if (w + "'s") in ws else 0) | (2 if (w + "s") in ws else 0)
    else:
        m = 0
        for op in range(N_OPS):
            ds = apply_op(w, op)
            if ds and len(set(ds)) == len(ds) and all(d in ws and d != w for d in ds):
                m |= (1 << op)
        full_m[w] = m

ZERO_MID = MASKS.index(0)
cands = defaultdict(list)
for w in words:
    fm = full_m[w]
    cands[w].append((ZERO_MID, ()))
    if not fm: continue
    up = w[0].isupper()
    for mid, m in enumerate(MASKS):
        if m == 0 or (m & fm) != m: continue
        if up and (m & ~3): continue
        ds = tuple(gen_derivs(w, m))
        if len(set(ds)) == len(ds):
            cands[w].append((mid, ds))

adj = defaultdict(set)
for w, opts in cands.items():
    for mid, ds in opts:
        for d in ds: adj[w].add(d); adj[d].add(w)

seen = set(); comps = []
for w in words:
    if w in seen: continue
    if not adj[w]: seen.add(w); continue
    q = [w]; seen.add(w); i = 0
    while i < len(q):
        u = q[i]; i += 1
        for v in adj[u]:
            if v not in seen: seen.add(v); q.append(v)
    comps.append(q)

mask_cnt = Counter(inf for _, _, inf in RAW_OLD)
tot_r = len(RAW_OLD)
mask_cost = [-math.log2((mask_cnt.get(i, 0) + 0.5) / (tot_r + 100)) for i in range(len(MASKS))]

def word_cost(w, mid):
    return 18.0 + 1.5 * len(w) + mask_cost[mid]

chosen_inf = {}
for w in words:
    if not adj[w]: chosen_inf[w] = ZERO_MID

greedy_inf = {w: inf for w, (_, _, inf) in zip(base_greedy, RAW_OLD)}

for comp in comps:
    edges = sum(len(adj[u]) for u in comp) // 2
    if edges == len(comp) - 1:
        root = min(comp, key=len)
        par = {root: None}; children = defaultdict(list); order = [root]
        for u in order:
            for v in adj[u]:
                if v not in par: par[v] = u; children[u].append(v); order.append(v)
        dp_cov_p, dp_cov_self, dp_cov_child, dp_up = {}, {}, {}, {}
        ch_self_m, ch_up_m, ch_cov_c = {}, {}, {}
        for u in reversed(order):
            p = par[u]
            dp_cov_p[u] = sum(min(dp_cov_self[v], dp_cov_child[v]) for v in children[u])
            b_self = 1e18; bm_self = 0; b_up = 1e18; bm_up = 0
            for mid, ds in cands[u]:
                ds_s = set(ds)
                c = word_cost(u, mid) + sum(dp_cov_p[v] if v in ds_s else min(dp_cov_self[v], dp_cov_child[v]) for v in children[u])
                if p is not None and p in ds_s:
                    if c < b_up: b_up = c; bm_up = mid
                else:
                    if c < b_self: b_self = c; bm_self = mid
            dp_cov_self[u] = b_self; ch_self_m[u] = bm_self
            dp_up[u] = b_up; ch_up_m[u] = bm_up
            base_sum = sum(min(dp_cov_self[v], dp_cov_child[v]) for v in children[u])
            b_cc = 1e18; best_c = None
            for vc in children[u]:
                if dp_up[vc] < 1e17:
                    c = base_sum - min(dp_cov_self[vc], dp_cov_child[vc]) + dp_up[vc]
                    if c < b_cc: b_cc = c; best_c = vc
            dp_cov_child[u] = b_cc; ch_cov_c[u] = best_c
        def recon(u, mode):
            if mode == 0:
                for v in children[u]: recon(v, 1)
            elif mode == 2:
                chosen_inf[u] = ch_up_m[u]
                ds_s = set(gen_derivs(u, MASKS[ch_up_m[u]]))
                for v in children[u]: recon(v, 0 if v in ds_s else 1)
            else:
                if dp_cov_self[u] <= dp_cov_child[u]:
                    chosen_inf[u] = ch_self_m[u]
                    ds_s = set(gen_derivs(u, MASKS[ch_self_m[u]]))
                    for v in children[u]: recon(v, 0 if v in ds_s else 1)
                else:
                    vc = ch_cov_c[u]
                    for v in children[u]: recon(v, 2 if v == vc else 1)
        recon(root, 1)
    elif len(comp) <= 21:
        idx_of = {u: i for i, u in enumerate(comp)}
        full_mask = (1 << len(comp)) - 1
        opts_comp = []
        for i, u in enumerate(comp):
            best_cm = {}
            for mid, ds in cands[u]:
                cm = (1 << i)
                for d in ds: cm |= (1 << idx_of[d])
                co = word_cost(u, mid)
                if cm not in best_cm or co < best_cm[cm][0]: best_cm[cm] = (co, mid)
            opts_comp.append([(cm, v[0], v[1]) for cm, v in best_cm.items()])
        memo = {full_mask: (0.0, None)}
        def solve_bm(bm):
            if bm in memo: return memo[bm]
            v = (full_mask ^ bm) & -(full_mask ^ bm)
            vi = v.bit_length() - 1
            best = (1e18, None)
            for u_i, o_u in enumerate(opts_comp):
                if (bm >> u_i) & 1: continue
                for cm, co, mid in o_u:
                    if ((cm >> vi) & 1) and ((cm & bm) == 0):
                        sub_c, _ = solve_bm(bm | cm)
                        if co + sub_c < best[0]: best = (co + sub_c, (u_i, cm, mid))
            memo[bm] = best; return best
        solve_bm(0)
        bm = 0
        while bm != full_mask:
            _, act = memo[bm]
            u_i, cm, mid = act
            chosen_inf[comp[u_i]] = mid
            bm |= cm
    else:
        cur = {}
        covered_by = {}
        for u in comp:
            if u in greedy_inf:
                mid = greedy_inf[u]; cur[u] = mid; covered_by[u] = u
                for d in gen_derivs(u, MASKS[mid]): covered_by[d] = u
        comp_s = sorted(comp, key=len)
        for center in comp_s:
            sub_nodes = []; sub_set = set()
            q_loc = [center]
            while q_loc and len(sub_nodes) < 13:
                x = q_loc.pop(0)
                owner = covered_by[x]
                group = [owner] + list(gen_derivs(owner, MASKS[cur[owner]]))
                if any(g not in sub_set for g in group):
                    if len(sub_nodes) + len(group) > 14 and len(sub_nodes) > 0: continue
                    for g in group:
                        if g not in sub_set: sub_set.add(g); sub_nodes.append(g)
                    for g in group:
                        for nb in adj[g]:
                            if nb not in sub_set: q_loc.append(nb)
            if len(sub_nodes) <= 1: continue
            idx_of = {u: i for i, u in enumerate(sub_nodes)}
            full_m_sub = (1 << len(sub_nodes)) - 1
            old_cost = sum(word_cost(u, cur[u]) for u in sub_nodes if u in cur)
            opts_sub = []
            for i, u in enumerate(sub_nodes):
                best_cm = {}
                for mid, ds in cands[u]:
                    if not all(d in idx_of for d in ds): continue
                    cm = (1 << i)
                    for d in ds: cm |= (1 << idx_of[d])
                    co = word_cost(u, mid)
                    if cm not in best_cm or co < best_cm[cm][0]: best_cm[cm] = (co, mid)
                opts_sub.append([(cm, v[0], v[1]) for cm, v in best_cm.items()])
            memo = {full_m_sub: (0.0, None)}
            def sbm(bm):
                if bm in memo: return memo[bm]
                v = (full_m_sub ^ bm) & -(full_m_sub ^ bm)
                vi = v.bit_length() - 1
                best = (1e18, None)
                for u_i, o_u in enumerate(opts_sub):
                    if (bm >> u_i) & 1: continue
                    for cm, co, mid in o_u:
                        if ((cm >> vi) & 1) and ((cm & bm) == 0):
                            sc, _ = sbm(bm | cm)
                            if co + sc < best[0]: best = (co + sc, (u_i, cm, mid))
                memo[bm] = best; return best
            new_c, _ = sbm(0)
            if new_c < old_cost - 1e-6:
                for u in sub_nodes: cur.pop(u, None)
                bm = 0
                while bm != full_m_sub:
                    _, act = memo[bm]; u_i, cm, mid = act
                    bu = sub_nodes[u_i]; cur[bu] = mid; covered_by[bu] = bu
                    for d in gen_derivs(bu, MASKS[mid]): covered_by[d] = bu
                    bm |= cm
        for u, mid in cur.items(): chosen_inf[u] = mid

base_list = sorted(chosen_inf.keys())
rebuilt = set()
for w in base_list:
    rebuilt.add(w)
    for d in gen_derivs(w, MASKS[chosen_inf[w]]): rebuilt.add(d)
assert rebuilt == ws, f"Mismatch! missing={len(ws-rebuilt)} extra={len(rebuilt-ws)}"
assert sum(1 + len(gen_derivs(w, MASKS[chosen_inf[w]])) for w in base_list) == len(ws), "Duplicate derived word!"

p_s, suf_list, i_s = [], [], []
prev = ""
for w in base_list:
    p = 0
    while p < len(prev) and p < len(w) and prev[p] == w[p]: p += 1
    p_s.append(p); i_s.append(chosen_inf[w]); suf_list.append(w[p:]); prev = w

freq_m = [k for k, _ in Counter(i_s).most_common()]
remap = {old: new for new, old in enumerate(freq_m)}
inf_rem = [remap[x] for x in i_s]
masks_rem = [MASKS[k] for k in freq_m]
while len(masks_rem) < 200:
    for m in MASKS:
        if m not in masks_rem: masks_rem.append(m); break
raw_items = [(p, suf, inf) for p, suf, inf in zip(p_s, suf_list, inf_rem)]

with open(OUT_PATH, "wb") as f:
    pickle.dump({"base_list": base_list, "raw_items": raw_items, "masks_rem": masks_rem}, f)
print(f"Saved {len(raw_items)} DP-optimal base items ({sum(len(s) for _,s,_ in raw_items)} suf chars) and {len(masks_rem)} masks to {OUT_PATH} in {(time.perf_counter()-t0):.2f}s")
