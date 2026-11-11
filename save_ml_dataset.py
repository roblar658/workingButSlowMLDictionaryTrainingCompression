import gc, pickle, time
from collections import Counter

DICT_PATH = r"C:\dev\ML\dict.txt"
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
del pwr2, full_masks, ws; gc.collect()

base_list = sorted(inf_o.keys())
p_s = []; i_s = []; suf_list = []; prev = ""
for w in base_list:
    p = 0
    while p < len(prev) and p < len(w) and prev[p] == w[p]: p += 1
    p_s.append(p); i_s.append(inf_o[w]); suf_list.append(w[p:]); prev = w

freq = [k for k, _ in Counter(i_s).most_common()]
remap = {old: new for new, old in enumerate(freq)}
inf_rem = [remap[x] for x in i_s]
masks_rem = [tm[k] for k in freq]

raw_items = [(p, suf, inf) for p, suf, inf in zip(p_s, suf_list, inf_rem)]
data = {"base_list": base_list, "raw_items": raw_items, "masks_rem": masks_rem}
out_p = r"C:\dev\ML\ml_dataset.pkl"
with open(out_p, "wb") as f:
    pickle.dump(data, f)
print(f"Saved {len(raw_items)} base items ({sum(len(s) for _,s,_ in raw_items)} suf chars) and {len(masks_rem)} masks to ml_dataset.pkl")
