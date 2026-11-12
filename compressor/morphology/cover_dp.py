"""
Exact Cover Dynamic Programming
===============================
Solves the exact cover / dominating-mask problem over connected components of
the morphological derivation graph:
- Tree DP for acyclic components (|E| = |V| - 1)
- Bitmask subset DP for small cyclic/DAG components (|V| <= 21)
- Iterative BFS closed-group DP for large components
"""

import math
from typing import Dict, List, Set, Tuple

class ExactCoverDP:
    """Exact DP solver for morphological graph cover."""

    @staticmethod
    def cost(word: str, mask: int, mask_probs: Dict[int, float]) -> float:
        """Estimate coding cost in bits for base word with mask."""
        prob = mask_probs.get(mask, 1e-6)
        return 18.0 + 1.5 * len(word) - math.log2(prob)

    @classmethod
    def solve_tree(
        cls,
        nodes: List[str],
        adj: Dict[str, Set[str]],
        valid_masks: Dict[str, List[Tuple[int, Set[str]]]],
        mask_probs: Dict[int, float]
    ) -> List[Tuple[str, int]]:
        """Exact bottom-up Tree DP for trees (|E| = |V| - 1)."""
        root = min(nodes, key=len)
        parent: Dict[str, str] = {root: ""}
        children: Dict[str, List[str]] = {n: [] for n in nodes}
        order = [root]
        
        for u in order:
            for v in adj[u]:
                if v != parent[u]:
                    parent[v] = u
                    children[u].append(v)
                    order.append(v)
        
        # DP states: covP, covSelf, up, covChild
        dp_covP = {}
        dp_covSelf = {}
        dp_up = {}
        dp_covChild = {}
        plan_covSelf = {}
        plan_up = {}
        plan_covChild = {}

        for u in reversed(order):
            ch = children[u]
            p = parent[u]
            
            # 1. covP (covered by parent)
            cost_covP = sum(min(dp_covSelf[v], dp_covChild[v]) for v in ch)
            dp_covP[u] = cost_covP
            
            # 2. covSelf (u is base word, covering subset of children)
            best_self = float("inf")
            best_m = 0
            for m, cov in valid_masks[u]:
                if p and p in cov:
                    continue  # covSelf must not cover parent
                c = cls.cost(u, m, mask_probs)
                for v in ch:
                    if v in cov:
                        c += dp_covP[v]
                    else:
                        c += min(dp_covSelf[v], dp_covChild[v])
                if c < best_self:
                    best_self = c
                    best_m = m
            dp_covSelf[u] = best_self
            plan_covSelf[u] = best_m
            
            # 3. up (u is base word, covering parent and subset of children)
            if p:
                best_up = float("inf")
                best_up_m = 0
                for m, cov in valid_masks[u]:
                    if p in cov:
                        c = cls.cost(u, m, mask_probs)
                        for v in ch:
                            if v in cov:
                                c += dp_covP[v]
                            else:
                                c += min(dp_covSelf[v], dp_covChild[v])
                        if c < best_up:
                            best_up = c
                            best_up_m = m
                dp_up[u] = best_up
                plan_up[u] = best_up_m
            else:
                dp_up[u] = float("inf")
            
            # 4. covChild (u is covered by exactly one child in state 'up')
            best_child = float("inf")
            best_v = None
            sum_other = sum(min(dp_covSelf[v], dp_covChild[v]) for v in ch)
            for v in ch:
                if dp_up[v] < float("inf"):
                    c = dp_up[v] + (sum_other - min(dp_covSelf[v], dp_covChild[v]))
                    if c < best_child:
                        best_child = c
                        best_v = v
            dp_covChild[u] = best_child
            plan_covChild[u] = best_v

        # Reconstruct optimal choice
        res: List[Tuple[str, int]] = []
        
        def trace(u: str, state: str):
            ch = children[u]
            if state == "covP":
                for v in ch:
                    st = "covSelf" if dp_covSelf[v] <= dp_covChild[v] else "covChild"
                    trace(v, st)
            elif state == "covSelf":
                m = plan_covSelf[u]
                res.append((u, m))
                cov = next(c for mm, c in valid_masks[u] if mm == m)
                for v in ch:
                    if v in cov:
                        trace(v, "covP")
                    else:
                        st = "covSelf" if dp_covSelf[v] <= dp_covChild[v] else "covChild"
                        trace(v, st)
            elif state == "up":
                m = plan_up[u]
                res.append((u, m))
                cov = next(c for mm, c in valid_masks[u] if mm == m)
                for v in ch:
                    if v in cov:
                        trace(v, "covP")
                    else:
                        st = "covSelf" if dp_covSelf[v] <= dp_covChild[v] else "covChild"
                        trace(v, st)
            elif state == "covChild":
                star_v = plan_covChild[u]
                trace(star_v, "up")
                for v in ch:
                    if v != star_v:
                        st = "covSelf" if dp_covSelf[v] <= dp_covChild[v] else "covChild"
                        trace(v, st)

        root_state = "covSelf" if dp_covSelf[root] <= dp_covChild[root] else "covChild"
        trace(root, root_state)
        return res

    @classmethod
    def solve_subset_dp(
        cls,
        nodes: List[str],
        valid_masks: Dict[str, List[Tuple[int, Set[str]]]],
        mask_probs: Dict[int, float]
    ) -> List[Tuple[str, int]]:
        """Exact bitmask subset DP for small components (|V| <= 21)."""
        n = len(nodes)
        n2i = {w: i for i, w in enumerate(nodes)}
        
        cands = []
        for u in nodes:
            ui = n2i[u]
            for m, cov in valid_masks[u]:
                cov_mask = 1 << ui
                for v in cov:
                    if v in n2i:
                        cov_mask |= (1 << n2i[v])
                cands.append((ui, m, cov_mask, cls.cost(u, m, mask_probs)))
        
        full = (1 << n) - 1
        dp = [float("inf")] * (1 << n)
        parent = [None] * (1 << n)
        dp[0] = 0.0
        
        for bm in range(full):
            cur_c = dp[bm]
            if cur_c == float("inf"):
                continue
            # Branch only on candidates that cover the lowest uncovered element
            uncovered = full ^ bm
            target_bit = (uncovered & -uncovered).bit_length() - 1
            
            for ui, m, c_mask, cost_val in cands:
                if (c_mask & (1 << target_bit)) and not (c_mask & bm):
                    nbm = bm | c_mask
                    nc = cur_c + cost_val
                    if nc < dp[nbm]:
                        dp[nbm] = nc
                        parent[nbm] = (bm, ui, m)
        
        res = []
        curr = full
        while curr > 0:
            prev, ui, m = parent[curr]
            res.append((nodes[ui], m))
            curr = prev
        return res
