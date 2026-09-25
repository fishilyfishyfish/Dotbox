"""자기대국과 대결을 여러 프로세스로 나눠 돌린다 (코어 수만큼 빨라짐)."""
from __future__ import annotations
import numpy as np
import torch

from .game import Game
from .net import Net
from .mcts import MCTS
from .net import Evaluator
from .arena import mcts_player, random_player, greedy_player, play_match

_CTX = {}


def init_worker(rows, cols, hidden, layers, state_dicts, cfg):
    torch.set_num_threads(1)
    g = Game(rows, cols)
    nets = []
    for sd in state_dicts:
        n = Net(g, hidden, layers)
        n.load_state_dict(sd)
        n.eval()
        nets.append(n)
    _CTX.clear()
    _CTX.update(g=g, nets=nets, cfg=cfg)


def job_selfplay(seed):
    from .train import self_play_game
    rng = np.random.default_rng(seed)
    data, score = self_play_game(_CTX["g"], _CTX["nets"][0], _CTX["cfg"], rng)
    return data


def _make(kind, idx, rng):
    g, cfg = _CTX["g"], _CTX["cfg"]
    if kind == "random":
        return random_player(g, rng)
    if kind == "greedy":
        return greedy_player(g, rng)
    return mcts_player(g, _CTX["nets"][idx], cfg["sims"], cfg["c_puct"], rng)


def job_duel(args):
    """(A가 선공인지, A종류, A신경망번호, B종류, B신경망번호, 난수씨앗) -> (A점수, B점수)"""
    a_first, ka, ia, kb, ib, seed = args
    rng = np.random.default_rng(seed)
    pa, pb = _make(ka, ia, rng), _make(kb, ib, rng)
    if a_first:
        sa, sb = play_match(_CTX["g"], pa, pb, rng)
    else:
        sb, sa = play_match(_CTX["g"], pb, pa, rng)
    return sa, sb


def summarize(results):
    wa = wb = dr = 0
    for sa, sb in results:
        if sa > sb:
            wa += 1
        elif sb > sa:
            wb += 1
        else:
            dr += 1
    return wa, wb, dr
