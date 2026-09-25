"""실력 재기: 두 선수를 붙이고 이긴 횟수를 센다."""
from __future__ import annotations
import numpy as np
from .mcts import MCTS
from .net import Evaluator


def mcts_player(game, net, sims, c_puct, rng, temp_moves=4):
    ev = Evaluator(game, net)

    def play(mask, ply):
        m = MCTS(game, ev, sims=sims, c_puct=c_puct, rng=rng)
        pi = m.run(mask, add_noise=False)
        if ply < temp_moves:                      # 초반엔 조금 흔들어서 매 판 다르게
            return int(rng.choice(len(pi), p=pi))
        return int(np.argmax(pi))
    return play


def random_player(game, rng):
    def play(mask, ply):
        legal = game.legal(mask)
        return int(rng.choice(legal))
    return play


def greedy_player(game, rng):
    """칸을 딸 수 있으면 딴다. 없으면 상대에게 칸을 안 내주는 수 중 무작위."""
    def play(mask, ply):
        legal = game.legal(mask)
        take = [e for e in legal if game.completed(mask, e)]
        if take:
            return int(rng.choice(take))
        safe = []
        for e in legal:
            nm = mask | (1 << e)
            if not any(game.completed(nm, f) for f in game.legal(nm)):
                safe.append(e)
        return int(rng.choice(safe if safe else legal))
    return play


def play_match(game, p0, p1, rng):
    """p0이 선공. (p0 점수, p1 점수)"""
    mask, turn, ply = 0, 0, 0
    score = [0, 0]
    players = [p0, p1]
    while not game.is_over(mask):
        a = players[turn](mask, ply)
        mask, comp, again = game.step(mask, a)
        score[turn] += comp
        ply += 1
        if not again:
            turn = 1 - turn
    return score[0], score[1]


def duel(game, pa_factory, pb_factory, n_games, rng):
    """A와 B를 선공 번갈아 가며 n_games 판. (A승, B승, 무)"""
    wa = wb = dr = 0
    for i in range(n_games):
        pa, pb = pa_factory(), pb_factory()
        if i % 2 == 0:
            sa, sb = play_match(game, pa, pb, rng)
        else:
            sb, sa = play_match(game, pb, pa, rng)
        if sa > sb:
            wa += 1
        elif sb > sa:
            wb += 1
        else:
            dr += 1
    return wa, wb, dr
