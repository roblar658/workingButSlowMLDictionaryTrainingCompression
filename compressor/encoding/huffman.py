"""
Huffman Tree Module
===================
Canonical Huffman code length computation and binary prefix tree generation.
"""

from typing import List, Dict, Any, Tuple
import heapq

class CanonicalHuffman:
    """Computes canonical Huffman code lengths and binary prefix trees."""

    @staticmethod
    def compute_lengths(freqs: List[int]) -> List[int]:
        """Compute minimum-redundancy code lengths given symbol frequencies."""
        K = len(freqs)
        pq = []
        parent = [-1] * (2 * K - 1)
        for s in range(K):
            heapq.heappush(pq, (max(1, freqs[s]), s, s))
        uid = K
        while len(pq) > 1:
            f1, _, i1 = heapq.heappop(pq)
            f2, _, i2 = heapq.heappop(pq)
            parent[i1] = uid
            parent[i2] = uid
            heapq.heappush(pq, (f1 + f2, uid, uid))
            uid += 1
        lens = [0] * K
        for s in range(K):
            u = s
            d = 0
            while parent[u] != -1:
                u = parent[u]
                d += 1
            lens[s] = d
        return lens

    @staticmethod
    def build_canonical_tree(lens: List[int]) -> Dict[str, Any]:
        """Construct canonical binary decoding tree and bit-encoding paths from code lengths."""
        K = len(lens)
        L = [0] * (K - 1)
        R = [0] * (K - 1)
        V = [0] * K
        nxt = 1
        li = 0
        c = 0
        pl = 0
        codes = {}
        for l, s in sorted((lens[i], i) for i in range(K)):
            c <<= (l - pl)
            pl = l
            u = 0
            path = []
            for bp in range(l - 1, 0, -1):
                bit = (c >> bp) & 1
                path.append((u, bit))
                if bit:
                    if R[u] == 0:
                        R[u] = nxt
                        nxt += 1
                    u = R[u]
                else:
                    if L[u] == 0:
                        L[u] = nxt
                        nxt += 1
                    u = L[u]
            bit = c & 1
            path.append((u, bit))
            leaf = K - 1 + li
            if bit:
                R[u] = leaf
            else:
                L[u] = leaf
            V[li] = s
            codes[s] = path
            li += 1
            c += 1
        
        lm = [0] * (2 * K - 1)
        for i in range(K):
            lm[K - 1 + i] = 1 << V[i]
        for i in range(K - 2, -1, -1):
            lm[i] = lm[L[i]] | lm[R[i]]
        
        return {
            "n_int": K - 1,
            "left": L,
            "right": R,
            "leaf_val": V,
            "leaf_mask": lm,
            "codes": codes,
            "lens": lens,
        }
