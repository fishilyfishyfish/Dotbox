"""알파제로식 몬테카를로 트리 탐색.

Dots & Boxes 특유의 점: 칸을 따면 같은 사람이 한 번 더 둔다.
그래서 자식 노드의 값을 가져올 때, 차례가 넘어갔으면 부호를 뒤집고
안 넘어갔으면 그대로 더한다.
"""
from __future__ import annotations
import math
import numpy as np


class Node:
    __slots__ = ("P", "N", "W", "acts")

    def __init__(self, P, acts):
        self.acts = acts
        self.P = P
        self.N = np.zeros(len(acts), dtype=np.int32)
        self.W = np.zeros(len(acts), dtype=np.float64)


class MCTS:
    def __init__(self, game, evaluator, sims=100, c_puct=1.5,
                 dirichlet_alpha=0.6, dirichlet_eps=0.25, rng=None):
        self.g, self.ev = game, evaluator
        self.sims, self.c = sims, c_puct
        self.d_alpha, self.d_eps = dirichlet_alpha, dirichlet_eps
        self.rng = rng or np.random.default_rng()
        self.tree = {}

    def reset(self):
        self.tree.clear()

    def _leaf(self, mask):
        p, v = self.ev(mask)
        acts = self.g.legal(mask)
        node = Node(np.array([p[a] for a in acts], dtype=np.float64), acts)
        s = node.P.sum()
        node.P = node.P / s if s > 0 else np.full(len(acts), 1.0 / max(len(acts), 1))
        self.tree[mask] = node
        return v

    def _search(self, mask, root_noise=False):
        if self.g.is_over(mask):
            return 0.0
        node = self.tree.get(mask)
        if node is None:
            return self._leaf(mask)

        P = node.P
        if root_noise and len(P) > 1:
            noise = self.rng.dirichlet([self.d_alpha] * len(P))
            P = (1 - self.d_eps) * P + self.d_eps * noise

        total = node.N.sum()
        q = np.where(node.N > 0, node.W / np.maximum(node.N, 1), 0.0)
        u = self.c * P * math.sqrt(total + 1e-8) / (1 + node.N)
        i = int(np.argmax(q + u))
        a = node.acts[i]

        child, comp, again = self.g.step(mask, a)
        vc = self._search(child)
        v = comp / self.g.total_boxes + vc if again else -vc
        v = max(-1.0, min(1.0, v))

        node.N[i] += 1
        node.W[i] += v
        return v

    def run(self, mask, add_noise=True):
        """방문 횟수 분포를 돌려준다 (길이 E, 불법 수는 0)."""
        for _ in range(self.sims):
            self._search(mask, root_noise=add_noise)
        node = self.tree[mask]
        pi = np.zeros(self.g.E, dtype=np.float64)
        for i, a in enumerate(node.acts):
            pi[a] = node.N[i]
        s = pi.sum()
        if s == 0:
            for a in node.acts:
                pi[a] = 1.0
            s = pi.sum()
        return pi / s

    def value_of(self, mask):
        node = self.tree.get(mask)
        if node is None or node.N.sum() == 0:
            return 0.0
        return float(node.W.sum() / node.N.sum())
