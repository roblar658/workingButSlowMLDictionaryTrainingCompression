"""
Dataset Builder
===============
Builds the morphological exact cover dataset from dict.txt and saves ml_dataset.pkl.
"""

import os
import pickle
import time
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

from .operators import MorphologicalRules
from .cover_dp import ExactCoverDP

class DatasetBuilder:
    """Discovers rules, runs Exact Cover DP, and builds dataset structures."""

    def __init__(self, dict_path: str):
        self.dict_path = dict_path
        with open(dict_path, "r", encoding="utf-8-sig") as f:
            raw_words = [w for w in f.read().splitlines() if w]
        self.words: List[str] = list(dict.fromkeys(raw_words))
        self.word_set: Set[str] = set(self.words)

    @staticmethod
    def _build_components(
        words: List[str],
        candidates: Dict[str, List[Tuple[int, Set[str]]]]
    ) -> Tuple[Dict[str, Set[str]], List[Tuple[List[str], int]]]:
        """Partition words into connected components of the derivation graph."""
        adj: Dict[str, Set[str]] = defaultdict(set)
        for w, cands in candidates.items():
            for _, cov in cands:
                for v in cov:
                    adj[w].add(v)
                    adj[v].add(w)

        visited: Set[str] = set()
        comps: List[Tuple[List[str], int]] = []
        for w in words:
            if w in visited:
                continue
            comp = [w]
            visited.add(w)
            edges = 0
            for u in comp:
                for v in adj[u]:
                    edges += 1
                    if v not in visited:
                        visited.add(v)
                        comp.append(v)
            comps.append((comp, edges // 2))
        return adj, comps

    @staticmethod
    def _solve_components(
        comps: List[Tuple[List[str], int]],
        adj: Dict[str, Set[str]],
        candidates: Dict[str, List[Tuple[int, Set[str]]]],
        mask_probs: Dict[int, float],
        group_limit: int = 16,
        sweeps: int = 2
    ) -> List[Tuple[str, int]]:
        """Solve each connected component using Tree DP, exact Subset DP, or Closed-Group DP."""
        chosen: List[Tuple[str, int]] = []
        for comp, edges in comps:
            if len(comp) == 1:
                chosen.append((comp[0], 0))
            elif edges == len(comp) - 1:
                chosen.extend(ExactCoverDP.solve_tree(comp, adj, candidates, mask_probs))
            elif len(comp) <= 20:
                chosen.extend(ExactCoverDP.solve_subset_dp(comp, candidates, mask_probs))
            else:
                chosen.extend(ExactCoverDP.solve_large_dp(
                    comp, adj, candidates, mask_probs, group_limit=group_limit, sweeps=sweeps
                ))
        return chosen

    def discover_valid_masks(self) -> Tuple[Dict[str, Dict[int, Set[str]]], Dict[str, List[Tuple[int, Set[str]]]], List[int]]:
        """Find candidate valid bitmasks for all words and run Stage-1 cover to rank useful masks."""
        word_op_map: Dict[str, Dict[int, Set[str]]] = {}
        init_candidates: Dict[str, List[Tuple[int, Set[str]]]] = {}
        init_mask_hits: Counter = Counter()

        for w in self.words:
            # Proper nouns / capitalized words only use 0 ('s) and 1 (plural)
            max_op = 2 if w[0].isupper() else 52

            valid_ops = []
            op_map: Dict[int, Set[str]] = {}
            for op_idx in range(max_op):
                generated = MorphologicalRules.apply_rule(op_idx, w)
                if generated and all(g in self.word_set and g != w for g in generated):
                    gs = set(generated)
                    valid_ops.append((op_idx, gs))
                    op_map[op_idx] = gs

            word_op_map[w] = op_map
            n_ops = len(valid_ops)
            cands: List[Tuple[int, Set[str]]] = [(0, set())]
            seen_m = {0}

            if n_ops <= 12:
                for bm in range(1, 1 << n_ops):
                    cov_set: Set[str] = set()
                    mask_val = 0
                    collision = False
                    for i in range(n_ops):
                        if (bm >> i) & 1:
                            op_id, g_set = valid_ops[i]
                            if cov_set & g_set:
                                collision = True
                                break
                            cov_set |= g_set
                            mask_val |= (1 << op_id)
                    if not collision and mask_val not in seen_m:
                        seen_m.add(mask_val)
                        cands.append((mask_val, cov_set))
                        init_mask_hits[mask_val] += 1
            else:
                # Enumerate all subsets of first 10 ops, plus all singletons, plus greedy maximal
                for bm in range(1, 1 << 10):
                    cov_set = set()
                    mask_val = 0
                    collision = False
                    for i in range(10):
                        if (bm >> i) & 1:
                            op_id, g_set = valid_ops[i]
                            if cov_set & g_set:
                                collision = True
                                break
                            cov_set |= g_set
                            mask_val |= (1 << op_id)
                    if not collision and mask_val not in seen_m:
                        seen_m.add(mask_val)
                        cands.append((mask_val, cov_set))
                        init_mask_hits[mask_val] += 1
                greedy_cov: Set[str] = set()
                greedy_m = 0
                for op_id, g_set in valid_ops:
                    sm = 1 << op_id
                    if sm not in seen_m:
                        seen_m.add(sm)
                        cands.append((sm, set(g_set)))
                        init_mask_hits[sm] += 1
                    if not (greedy_cov & g_set):
                        greedy_cov |= g_set
                        greedy_m |= sm
                if greedy_m not in seen_m:
                    seen_m.add(greedy_m)
                    cands.append((greedy_m, greedy_cov))
                    init_mask_hits[greedy_m] += 1

            init_candidates[w] = cands

        # Stage-1 heuristic solve using rough prior probabilities to identify high-utility masks
        tot_hits = max(1, sum(init_mask_hits.values()))
        init_probs: Dict[int, float] = {0: 0.25}
        for m, cnt in init_mask_hits.items():
            init_probs[m] = max(1e-5, cnt / (tot_hits * 2.0))

        adj1, comps1 = self._build_components(self.words, init_candidates)
        sol1 = self._solve_components(comps1, adj1, init_candidates, init_probs, group_limit=14, sweeps=1)
        used_counts = Counter(m for _, m in sol1 if m != 0)

        # Select top up-to-200 masks (always preserving mask 0, plus any top unchosen candidates if <200)
        selected_nz = [m for m, _ in used_counts.most_common(199)]
        if len(selected_nz) < 199:
            seen_nz = set(selected_nz)
            for m, _ in init_mask_hits.most_common():
                if m != 0 and m not in seen_nz:
                    selected_nz.append(m)
                    seen_nz.add(m)
                    if len(selected_nz) == 199:
                        break
        top_masks = [0] + selected_nz
        top_mask_set = set(top_masks)

        # Build final candidate table, checking all top_masks against hub words (>12 ops)
        filtered_candidates: Dict[str, List[Tuple[int, Set[str]]]] = {}
        for w in self.words:
            cands = [(m, cov) for m, cov in init_candidates[w] if m in top_mask_set]
            op_map = word_op_map[w]
            if len(op_map) > 12:
                have = {m for m, _ in cands}
                avail_bits = 0
                for op_id in op_map:
                    avail_bits |= (1 << op_id)
                for m in top_masks:
                    if m != 0 and m not in have and (m & avail_bits) == m:
                        cov_set = set()
                        collision = False
                        mm = m
                        while mm:
                            lsb = mm & -mm
                            gs = op_map[lsb.bit_length() - 1]
                            if cov_set & gs:
                                collision = True
                                break
                            cov_set |= gs
                            mm ^= lsb
                        if not collision:
                            cands.append((m, cov_set))
                            have.add(m)
            if not any(m == 0 for m, _ in cands):
                cands.append((0, set()))
            filtered_candidates[w] = cands

        return word_op_map, filtered_candidates, top_masks

    def build(self) -> Tuple[List[Tuple[int, str, int]], List[int]]:
        """Execute EM-refined morphological graph partitioning and return raw dataset."""
        _, filtered_cands, top_masks = self.discover_valid_masks()
        adj, comps = self._build_components(self.words, filtered_cands)

        # Iterative EM estimation of mask probabilities
        mask_probs: Dict[int, float] = {0: 0.25}
        for m in top_masks:
            if m != 0:
                mask_probs[m] = 0.75 / max(1, len(top_masks) - 1)

        chosen_bases: List[Tuple[str, int]] = []
        for _ in range(2):
            chosen_bases = self._solve_components(comps, adj, filtered_cands, mask_probs, group_limit=16, sweeps=2)
            cnt = Counter(m for _, m in chosen_bases)
            tot = max(1, len(chosen_bases))
            denom = tot + 0.5 * len(top_masks)
            mask_probs = {m: (cnt[m] + 0.5) / denom for m in top_masks}

        # Prune unused non-zero masks and order masks by descending frequency
        final_cnt = Counter(m for _, m in chosen_bases)
        active_masks = sorted(
            {0} | {m for m in top_masks if final_cnt[m] > 0},
            key=lambda m: (-final_cnt[m], m)
        )

        # Sort bases in UTF-8 lexicographic order and compute prefix length & suffix strings
        chosen_bases.sort(key=lambda t: t[0])
        m2idx = {m: i for i, m in enumerate(active_masks)}

        raw_items: List[Tuple[int, str, int]] = []
        prev = ""
        for b, m in chosen_bases:
            p = 0
            limit = min(len(prev), len(b), 255)
            while p < limit and prev[p] == b[p]:
                p += 1
            raw_items.append((p, b[p:], m2idx.get(m, 0)))
            prev = b

        return raw_items, active_masks

    def save_pickle(self, out_path: str) -> None:
        """Run build and persist ml_dataset.pkl."""
        raw_items, masks_rem = self.build()
        data = {
            "raw_items": raw_items,
            "masks_rem": masks_rem,
            "n_bases": len(raw_items),
        }
        with open(out_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
