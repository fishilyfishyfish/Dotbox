"""정책망 + 가치망. 판 하나를 넣으면 (각 변을 둘 확률, 이 판의 유불리)를 돌려준다."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def features(game, masks):
    """mask 여러 개를 신경망 입력으로 바꾼다.
    [변이 그어졌는지 E개] + [칸마다 변이 몇 개 그어졌는지 원핫 5개씩]
    """
    masks = np.asarray(masks, dtype=np.int64).reshape(-1)
    n, E, B = masks.size, game.E, game.total_boxes
    x = np.zeros((n, E + 5 * B), dtype=np.float32)
    for e in range(E):
        x[:, e] = ((masks >> e) & 1).astype(np.float32)
    for bi in range(B):
        cnt = np.zeros(n, dtype=np.int64)
        for e in game.boxes[bi]:
            cnt += (masks >> e) & 1
        x[np.arange(n), E + bi * 5 + cnt] = 1.0
    return x


class Net(nn.Module):
    def __init__(self, game, hidden: int = 256, layers: int = 3):
        super().__init__()
        self.in_dim = game.E + 5 * game.total_boxes
        self.E = game.E
        body = [nn.Linear(self.in_dim, hidden), nn.ReLU()]
        for _ in range(layers - 1):
            body += [nn.Linear(hidden, hidden), nn.ReLU()]
        self.body = nn.Sequential(*body)
        self.policy = nn.Linear(hidden, game.E)
        self.value = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())

    def forward(self, x):
        h = self.body(x)
        return self.policy(h), self.value(h).squeeze(-1)


class Evaluator:
    """신경망 호출 + 같은 판을 또 묻지 않도록 캐시."""

    def __init__(self, game, net, device="cpu"):
        self.g, self.net, self.device = game, net, device
        self.cache = {}
        self.calls = 0

    def clear(self):
        self.cache.clear()

    def __call__(self, mask: int):
        hit = self.cache.get(mask)
        if hit is not None:
            return hit
        x = torch.from_numpy(features(self.g, [mask])).to(self.device)
        with torch.no_grad():
            logit, v = self.net(x)
        logit = logit[0].cpu().numpy()
        legal = np.zeros(self.g.E, dtype=bool)
        for e in self.g.legal(mask):
            legal[e] = True
        logit[~legal] = -1e9
        logit -= logit.max()
        p = np.exp(logit)
        s = p.sum()
        p = p / s if s > 0 else legal.astype(np.float32) / max(legal.sum(), 1)
        out = (p, float(v[0]))
        self.cache[mask] = out
        self.calls += 1
        return out
