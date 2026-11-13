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
from typing import Dict, FrozenSet, List, Set, Tuple

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
        """Exact bitmask subset DP for small components (|V| <= 21) via reachable-state memoization."""
        n = len(nodes)
        if n == 0:
            return []
        n2i = {w: i for i, w in enumerate(nodes)}

        buckets: List[List[Tuple[int, int, int, float]]] = [[] for _ in range(n)]
        for u in nodes:
            ui = n2i[u]
            for m, cov in valid_masks[u]:
                if not cov.issubset(n2i):
                    continue
                cov_mask = 1 << ui
                for v in cov:
                    cov_mask |= (1 << n2i[v])
                cost_val = cls.cost(u, m, mask_probs)
                mm = cov_mask
                while mm:
                    lsb = mm & -mm
                    buckets[lsb.bit_length() - 1].append((ui, m, cov_mask, cost_val))
                    mm ^= lsb

        full = (1 << n) - 1
        memo: Dict[int, float] = {full: 0.0}
        choice: Dict[int, Tuple[int, int, int]] = {}

        def dfs(bm: int) -> float:
            if bm in memo:
                return memo[bm]
            uncovered = full ^ bm
            t = (uncovered & -uncovered).bit_length() - 1
            best = float("inf")
            best_c = None
            for ui, m, cm, cval in buckets[t]:
                if not (cm & bm):
                    val = cval + dfs(bm | cm)
                    if val < best:
                        best = val
                        best_c = (ui, m, cm)
            memo[bm] = best
            if best_c is not None:
                choice[bm] = best_c
            return best

        dfs(0)
        res = []
        curr = 0
        while curr != full:
            ui, m, cm = choice[curr]
            res.append((nodes[ui], m))
            curr |= cm
        return res

    @classmethod
    def solve_large_dp(
        cls,
        nodes: List[str],
        adj: Dict[str, Set[str]],
        valid_masks: Dict[str, List[Tuple[int, Set[str]]]],
        mask_probs: Dict[int, float],
        group_limit: int = 16,
        sweeps: int = 2
    ) -> List[Tuple[str, int]]:
        """Greedy initialization followed by iterative closed-group exact subset DP."""
        comp_set = set(nodes)
        all_cands = []
        for u in nodes:
            for m, cov in valid_masks[u]:
                if cov.issubset(comp_set):
                    st = frozenset({u} | cov)
                    all_cands.append((cls.cost(u, m, mask_probs) / len(st), u, m, st))
        all_cands.sort(key=lambda x: x[0])

        covered: Set[str] = set()
        star_of: Dict[str, Tuple[str, int, FrozenSet[str]]] = {}
        stars: Dict[str, Tuple[str, int, FrozenSet[str]]] = {}

        for _, u, m, st in all_cands:
            if not (st & covered):
                covered |= st
                entry = (u, m, st)
                stars[u] = entry
                for x in st:
                    star_of[x] = entry
        for u in nodes:
            if u not in covered:
                st = frozenset((u,))
                entry = (u, 0, st)
                stars[u] = entry
                star_of[u] = entry

        checked_groups: Set[FrozenSet[str]] = set()
        for _ in range(sweeps):
            improved = False
            for seed_root in list(stars.keys()):
                if seed_root not in stars:
                    continue
                seed_star = stars[seed_root]
                grp_nodes: Set[str] = set(seed_star[2])
                grp_stars: Dict[str, Tuple[str, int, FrozenSet[str]]] = {seed_root: seed_star}

                while len(grp_nodes) < group_limit:
                    best_cand = None
                    best_score = -1
                    cand_roots: Set[str] = set()
                    for v in grp_nodes:
                        for nb in adj[v]:
                            if nb in comp_set and nb not in grp_nodes:
                                cand_roots.add(star_of[nb][0])
                    if not cand_roots:
                        break
                    for cr in cand_roots:
                        st = stars[cr][2]
                        if len(grp_nodes) + len(st) <= group_limit:
                            # Prefer adding neighbor stars strongly connected to current group
                            links = sum(1 for x in st for nb in adj[x] if nb in grp_nodes)
                            sc = links * 10 - len(st)
                            if sc > best_score:
                                best_score = sc
                                best_cand = cr
                    if best_cand is None:
                        break
                    st_entry = stars[best_cand]
                    grp_stars[best_cand] = st_entry
                    grp_nodes |= st_entry[2]

                fg = frozenset(grp_nodes)
                if fg in checked_groups:
                    continue
                checked_groups.add(fg)

                old_cost = sum(cls.cost(su, sm, mask_probs) for su, sm, _ in grp_stars.values())
                sub_valid = {
                    x: [(m, cov) for m, cov in valid_masks[x] if cov.issubset(grp_nodes)]
                    for x in grp_nodes
                }
                sub_sol = cls.solve_subset_dp(list(grp_nodes), sub_valid, mask_probs)
                new_cost = sum(cls.cost(nu, nm, mask_probs) for nu, nm in sub_sol)
                if new_cost < old_cost - 1e-9:
                    improved = True
                    checked_groups.clear()
                    for old_r in grp_stars:
                        del stars[old_r]
                    for nu, nm in sub_sol:
                        ncov = next(c for mm, c in sub_valid[nu] if mm == nm)
                        nst = frozenset({nu} | ncov)
                        n_entry = (nu, nm, nst)
                        stars[nu] = n_entry
                        for x in nst:
                            star_of[x] = n_entry
            if not improved:
                break

        return [(u, m) for u, m, _ in stars.values()]
