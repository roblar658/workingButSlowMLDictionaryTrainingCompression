"""
PAQ Bit-FSM Module
==================
Defines the 3-order bit-level finite state machine tracking bit frequency histories.
"""

from typing import Tuple
import numpy as np

class PAQBitFSM:
    """Constructs and manages PAQ bit-history FSM state transitions."""

    RUNS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 26, 34, 44, 58, 76, 100, 140, 200]

    @classmethod
    def snap(cls, r: int) -> int:
        """Snap a run length to the nearest non-decreasing quantized boundary."""
        best = 0
        for x in cls.RUNS:
            if x <= r:
                best = x
        return best

    @classmethod
    def next_state(cls, c0: int, c1: int, bit: int) -> Tuple[int, int]:
        """Compute the next (c0, c1) count pair following an observed bit."""
        if bit == 0:
            if c1 == 0:
                return (cls.snap(c0 + 1), 0)
            if c1 == 1:
                return (min(c0 + 1, 16), 1)
            nc0 = c0 + 1
            nc1 = c1 if c1 < 4 else (c1 * 2 // 3 if c1 > 4 else 3)
        else:
            if c0 == 0:
                return (0, cls.snap(c1 + 1))
            if c0 == 1:
                return (1, min(c1 + 1, 16))
            nc1 = c1 + 1
            nc0 = c0 if c0 < 4 else (c0 * 2 // 3 if c0 > 4 else 3)
        
        if nc0 + nc1 > 40:
            nc0 = max(1, (nc0 * 3) // 4)
            nc1 = max(1, (nc1 * 3) // 4)
        return (nc0, nc1)

    @classmethod
    def build_fsm(cls, max_states: int = 80) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate transition arrays NX0 and NX1 for the FSM."""
        states = [(0, 0)]
        s2i = {(0, 0): 0}
        queue = [(0, 0)]

        idx = 0
        while idx < len(queue) and len(states) < max_states:
            st = queue[idx]
            idx += 1
            for b in (0, 1):
                t = cls.next_state(st[0], st[1], b)
                if t not in s2i and len(states) < max_states:
                    s2i[t] = len(states)
                    states.append(t)
                    queue.append(t)

        nx0 = [0] * len(states)
        nx1 = [0] * len(states)
        for i, st in enumerate(states):
            for b, nx in ((0, nx0), (1, nx1)):
                t = cls.next_state(st[0], st[1], b)
                while t not in s2i:
                    t = (max(0, t[0] - 1), max(0, t[1] - 1))
                nx[i] = s2i[t]

        return (
            np.array(states, dtype=np.float32),
            np.array(nx0, dtype=np.uint8),
            np.array(nx1, dtype=np.uint8)
        )
