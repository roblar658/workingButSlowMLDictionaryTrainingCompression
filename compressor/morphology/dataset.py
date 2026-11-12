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
        with open(dict_path, "r", encoding="ascii") as f:
            self.words: List[str] = f.read().splitlines()
        self.word_set: Set[str] = set(self.words)

    def discover_valid_masks(self) -> Tuple[Dict[str, List[Tuple[int, Set[str]]]], List[int]]:
        """Find candidate valid bitmasks for all words in the dictionary."""
        word_candidates: Dict[str, List[Tuple[int, Set[str]]]] = defaultdict(list)
        mask_counts: Counter = Counter()

        for w in self.words:
            # Proper nouns / capitalized words only use 0 ('s) and 1 (plural)
            max_op = 2 if w[0].isupper() else 52
            
            valid_ops = []
            for op_idx in range(max_op):
                generated = MorphologicalRules.apply_rule(op_idx, w)
                if generated and all(g in self.word_set and g != w for g in generated):
                    valid_ops.append((op_idx, set(generated)))
            
            # Form compound bitmasks without internal collisions
            n_ops = len(valid_ops)
            for bm in range(1 << min(n_ops, 12)):
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
                if not collision:
                    word_candidates[w].append((mask_val, cov_set))
                    if mask_val:
                        mask_counts[mask_val] += 1

        # Select top 200 masks
        top_masks = [0] + [m for m, _ in mask_counts.most_common(199)]
        top_mask_set = set(top_masks)
        
        # Filter candidates to the top 200 vocabulary
        filtered_candidates: Dict[str, List[Tuple[int, Set[str]]]] = {}
        for w, cands in word_candidates.items():
            filtered = [(m, cov) for m, cov in cands if m in top_mask_set]
            if not any(m == 0 for m, _ in filtered):
                filtered.append((0, set()))
            filtered_candidates[w] = filtered

        return filtered_candidates, top_masks

    def build(self) -> Tuple[List[Tuple[int, str, int]], List[int]]:
        """Execute the full morphological graph partitioning and return raw dataset."""
        filtered_cands, top_masks = self.discover_valid_masks()
        mask_probs = {m: 1.0 / len(top_masks) for m in top_masks}

        # Build derivation graph components
        adj: Dict[str, Set[str]] = defaultdict(set)
        for w, cands in filtered_cands.items():
            for m, cov in cands:
                for v in cov:
                    adj[w].add(v)
                    adj[v].add(w)

        visited: Set[str] = set()
        chosen_bases: List[Tuple[str, int]] = []

        for w in self.words:
            if w in visited:
                continue
            # BFS component discovery
            comp = [w]
            visited.add(w)
            q = [w]
            edges = 0
            for u in q:
                for v in adj[u]:
                    edges += 1
                    if v not in visited:
                        visited.add(v)
                        comp.append(v)
                        q.append(v)
            edges //= 2

            if edges == len(comp) - 1:
                # Tree component: solve via exact bottom-up Tree DP
                sol = ExactCoverDP.solve_tree(comp, adj, filtered_cands, mask_probs)
            elif len(comp) <= 20:
                # Small component: solve via exact Bitmask Subset DP
                sol = ExactCoverDP.solve_subset_dp(comp, filtered_cands, mask_probs)
            else:
                # Fallback greedy initialization with tree approximation
                sol = [(u, 0) for u in comp]
            
            chosen_bases.extend(sol)

        # Sort bases and compute prefix length & suffix strings
        chosen_bases.sort(key=lambda t: t[0])
        m2idx = {m: i for i, m in enumerate(top_masks)}
        
        raw_items: List[Tuple[int, str, int]] = []
        prev = ""
        for b, m in chosen_bases:
            p = 0
            while p < len(prev) and p < len(b) and prev[p] == b[p]:
                p += 1
            raw_items.append((p, b[p:], m2idx.get(m, 0)))
            prev = b

        return raw_items, top_masks

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
