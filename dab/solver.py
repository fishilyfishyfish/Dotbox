"""작은 판의 '정답'을 완전 탐색으로 구한다.

V[mask] = 지금 둘 차례인 사람이 최선으로 뒀을 때 얻게 되는
          (내가 앞으로 딸 칸 - 상대가 앞으로 딸 칸).
이 값은 mask에만 의존하므로 모든 mask에 대해 한 번에 구해 둘 수 있다.
변이 많아지는 판은 2^E 가 폭발하므로 E <= 26 정도까지만 쓴다.
"""
from __future__ import annotations
import numpy as np


def popcount_table(n_bits: int) -> np.ndarray:
    """0..2^n_bits-1 의 popcount 배열."""
    pc = np.zeros(1, dtype=np.int8)
    for _ in range(n_bits):
        pc = np.concatenate([pc, pc + 1]).astype(np.int8)
    return pc


def solve_table(game, verbose: bool = False) -> np.ndarray:
    E = game.E
    if E > 26:
        raise ValueError(f"변이 {E}개라 완전 탐색이 무리임 (26 이하만)")
    N = 1 << E
    V = np.zeros(N, dtype=np.int8)
    pc = popcount_table(E)

    # 변마다, 그 변에 맞닿은 칸들의 비트마스크
    ebm = [[game.box_mask[bi] for bi in game.edge_boxes[e]] for e in range(E)]

    for k in range(E - 1, -1, -1):
        idx = np.flatnonzero(pc == k).astype(np.int64)
        if idx.size == 0:
            continue
        best = np.full(idx.size, -127, dtype=np.int16)
        for e in range(E):
            bit = np.int64(1) << np.int64(e)
            free = (idx & bit) == 0
            if not free.any():
                continue
            sub = idx[free]
            nm = sub | bit
            comp = np.zeros(sub.size, dtype=np.int16)
            for bm in ebm[e]:
                comp += ((nm & bm) == bm).astype(np.int16)
            child = V[nm].astype(np.int16)
            val = np.where(comp > 0, comp + child, -child)
            cur = best[free]
            best[free] = np.maximum(cur, val)
        V[idx] = best.astype(np.int8)
        if verbose:
            print(f"  popcount {k}: {idx.size:,}개", flush=True)
    return V


def best_moves(game, V: np.ndarray, mask: int):
    """이 자리에서 '정답'인 변들의 집합과 그때의 값."""
    best, out = None, []
    for e in game.legal(mask):
        nm = mask | (1 << e)
        comp = game.completed(mask, e)
        v = comp + int(V[nm]) if comp else -int(V[nm])
        if best is None or v > best:
            best, out = v, [e]
        elif v == best:
            out.append(e)
    return set(out), best
