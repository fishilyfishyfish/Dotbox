"""자기대국 강화학습 루프 (알파제로 방식).

한 번 돌 때마다:
  1) 지금 가장 센 신경망끼리 자기대국 -> 학습 데이터
  2) 그 데이터로 새 신경망 학습
  3) 새 신경망 vs 지금 신경망 대결 -> 이기면 교체
  4) 성적 네 가지를 기록하고 저장

중간에 끊겨도 state/ 폴더에서 이어서 돈다.
"""
from __future__ import annotations
import argparse, json, os, time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from .game import Game
from .net import Net, Evaluator, features
from .mcts import MCTS
from .solver import solve_table, best_moves
from .arena import mcts_player, random_player, greedy_player, duel
from .report import write_report
from . import parallel as par
import multiprocessing as mp

STATE = "state"


# ---------------------------------------------------------------- 자기대국
def self_play_game(game, net, cfg, rng):
    ev = Evaluator(game, net)
    mcts = MCTS(game, ev, sims=cfg["sims"], c_puct=cfg["c_puct"],
                dirichlet_alpha=cfg["dirichlet_alpha"],
                dirichlet_eps=cfg["dirichlet_eps"], rng=rng)
    mask, turn, ply = 0, 0, 0
    score = [0, 0]
    trace = []
    while not game.is_over(mask):
        mcts.reset()
        pi = mcts.run(mask, add_noise=True)
        trace.append((mask, pi.copy(), turn, score[0], score[1]))
        if ply < cfg["temp_moves"]:
            a = int(rng.choice(len(pi), p=pi))
        else:
            a = int(np.argmax(pi))
        mask, comp, again = game.step(mask, a)
        score[turn] += comp
        ply += 1
        if not again:
            turn = 1 - turn

    T = game.total_boxes
    data = []
    for (m, pi, p, s0, s1) in trace:
        got = [score[0] - s0, score[1] - s1]          # 그 시점 이후로 각자 딴 칸
        z = (got[p] - got[1 - p]) / T                 # 둘 차례인 사람 기준
        data.append((m, pi, z))
    return data, score


# ---------------------------------------------------------------- 학습
def train_net(game, net, buffer, cfg, device):
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n = len(buffer)
    bs = min(cfg["batch_size"], n)
    steps = cfg["train_steps"]
    idx_all = np.arange(n)
    pl = vl = 0.0
    for _ in range(steps):
        idx = np.random.choice(idx_all, bs, replace=False)
        masks = [buffer[i][0] for i in idx]
        pi = np.stack([buffer[i][1] for i in idx])
        z = np.array([buffer[i][2] for i in idx], dtype=np.float32)

        x = torch.from_numpy(features(game, masks)).to(device)
        tpi = torch.from_numpy(pi.astype(np.float32)).to(device)
        tz = torch.from_numpy(z).to(device)

        logit, v = net(x)
        illegal = x[:, :game.E] > 0.5                 # 이미 그어진 변은 후보에서 제외
        logp = F.log_softmax(logit.masked_fill(illegal, -1e9), dim=1)
        loss_p = -(tpi * logp).sum(1).mean()
        loss_v = F.mse_loss(v, tz)
        loss = loss_p + loss_v
        opt.zero_grad()
        loss.backward()
        opt.step()
        pl += loss_p.detach().item(); vl += loss_v.detach().item()
    net.eval()
    return pl / steps, vl / steps


# ---------------------------------------------------------------- 성적표
def exact_match_rate(game, V, net, cfg, rng, n_pos=60):
    """무작위로 만든 자리에서, 신경망이 고른 수가 '정답'인 비율."""
    ev = Evaluator(game, net)
    hit_mcts = hit_raw = tot = 0
    tries = 0
    while tot < n_pos and tries < n_pos * 10:
        tries += 1
        k = int(rng.integers(0, game.E - 1))
        mask = 0
        for e in rng.permutation(game.E)[:k]:
            mask |= 1 << int(e)
        if game.is_over(mask):
            continue
        good, _ = best_moves(game, V, mask)
        legal = game.legal(mask)
        if len(good) == len(legal):       # 아무거나 둬도 정답이면 시험 가치가 없음
            continue
        m = MCTS(game, ev, sims=cfg["sims"], c_puct=cfg["c_puct"], rng=rng)
        pi = m.run(mask, add_noise=False)
        hit_mcts += int(np.argmax(pi) in good)
        p, _ = ev(mask)
        hit_raw += int(int(np.argmax(p)) in good)
        tot += 1
    if tot == 0:
        return None, None
    return hit_mcts / tot, hit_raw / tot


# ---------------------------------------------------------------- 체크포인트
def load_progress():
    p = os.path.join(STATE, "progress.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {"stage": 0, "iter": 0, "history": []}


def save_progress(prog):
    os.makedirs(STATE, exist_ok=True)
    tmp = os.path.join(STATE, "progress.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(prog, f, ensure_ascii=False, indent=1)
    os.replace(tmp, os.path.join(STATE, "progress.json"))


def net_path(stage_name):
    return os.path.join(STATE, f"best_{stage_name}.pt")


# ---------------------------------------------------------------- 본체
def run(cfg_path="config.json"):
    with open(cfg_path, encoding="utf-8") as f:
        cfg_all = json.load(f)
    device = "cuda" if torch.cuda.is_available() and cfg_all.get("use_gpu", True) else "cpu"
    deadline = time.time() + cfg_all["max_minutes"] * 60
    os.makedirs(STATE, exist_ok=True)
    workers = cfg_all.get("workers") or os.cpu_count() or 1
    ctxmp = mp.get_context("fork")
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    print(f"일꾼 {workers}개 · 장치 {device}", flush=True)
    prog = load_progress()
    rng = np.random.default_rng()

    while prog["stage"] < len(cfg_all["stages"]):
        cfg = dict(cfg_all["defaults"])
        cfg.update(cfg_all["stages"][prog["stage"]])
        name = cfg["name"]
        game = Game(cfg["rows"], cfg["cols"])

        V = None
        if cfg.get("exact_check") and game.E <= 26:
            print(f"[{name}] 정답표 계산 중 (변 {game.E}개)...", flush=True)
            V = solve_table(game)
            print(f"[{name}] 정답표 완료. 빈 판 값 = {int(V[0])}", flush=True)

        net = Net(game, cfg["hidden"], cfg["layers"]).to(device)
        if os.path.exists(net_path(name)):
            net.load_state_dict(torch.load(net_path(name), map_location=device))
            print(f"[{name}] 이어서 학습 (반복 {prog['iter']}회차부터)", flush=True)
        net.eval()

        buffer = deque(maxlen=cfg["buffer_size"])

        while prog["iter"] < cfg["iterations"]:
            if time.time() > deadline:
                print("시간 예산 종료. 다음 실행에서 이어감.", flush=True)
                save_progress(prog)
                write_report(prog, cfg_all)
                return
            t0 = time.time()
            cpu = lambda m: {k: v.cpu() for k, v in m.state_dict().items()}
            mkpool = lambda sds: ctxmp.Pool(
                workers, initializer=par.init_worker,
                initargs=(cfg["rows"], cfg["cols"], cfg["hidden"], cfg["layers"], sds, cfg))

            # 1) 자기대국 (여러 프로세스)
            with mkpool([cpu(net)]) as pool:
                seeds = rng.integers(0, 2**31 - 1, cfg["games_per_iter"]).tolist()
                for data in pool.imap_unordered(par.job_selfplay, seeds):
                    buffer.extend(data)

            # 2) 학습 (새 후보)
            cand = Net(game, cfg["hidden"], cfg["layers"]).to(device)
            cand.load_state_dict(net.state_dict())
            lp, lv = train_net(game, cand, buffer, cfg, device)

            # 3) 후보(0) vs 현재(1), 그리고 성적 측정까지 한 번에
            jobs = []
            n_arena, n_test = cfg["arena_games"], cfg["test_games"]
            for i in range(n_arena):
                jobs.append((i % 2 == 0, "net", 0, "net", 1, int(rng.integers(1 << 30))))
            for i in range(n_test):
                jobs.append((i % 2 == 0, "net", 1, "random", 0, int(rng.integers(1 << 30))))
            for i in range(n_test):
                jobs.append((i % 2 == 0, "net", 1, "greedy", 0, int(rng.integers(1 << 30))))
            with mkpool([cpu(cand), cpu(net)]) as pool:
                res = pool.map(par.job_duel, jobs)
            wa, wb, dr = par.summarize(res[:n_arena])
            rw, _, rd = par.summarize(res[n_arena:n_arena + n_test])
            gw, _, gd = par.summarize(res[n_arena + n_test:])

            wr = (wa + 0.5 * dr) / max(wa + wb + dr, 1)
            promoted = wr >= cfg["promote_winrate"]
            if promoted:
                net = cand
                torch.save(net.state_dict(), net_path(name))
            em = emr = None
            if V is not None:
                em, emr = exact_match_rate(game, V, net, cfg, rng, cfg["exact_positions"])

            rec = {
                "stage": name, "iter": prog["iter"] + 1,
                "loss_policy": round(lp, 4), "loss_value": round(lv, 4),
                "vs_prev_winrate": round(wr, 3), "promoted": promoted,
                "vs_random": round((rw + 0.5 * rd) / cfg["test_games"], 3),
                "vs_greedy": round((gw + 0.5 * gd) / cfg["test_games"], 3),
                "exact_match_mcts": None if em is None else round(em, 3),
                "exact_match_raw": None if emr is None else round(emr, 3),
                "buffer": len(buffer), "sec": round(time.time() - t0, 1),
            }
            prog["history"].append(rec)
            prog["iter"] += 1
            save_progress(prog)
            write_report(prog, cfg_all)
            print(
                f"[{name}] {rec['iter']:>3}회차 "
                f"손실 p={rec['loss_policy']:.3f} v={rec['loss_value']:.3f} | "
                f"이전판 대비 {rec['vs_prev_winrate']:.0%}{' 교체' if promoted else ''} | "
                f"무작위 {rec['vs_random']:.0%} 욕심쟁이 {rec['vs_greedy']:.0%}"
                + (f" | 정답일치 {em:.0%}(탐색) {emr:.0%}(직감)" if em is not None else "")
                + f" | {rec['sec']}초", flush=True)

        prog["stage"] += 1
        prog["iter"] = 0
        save_progress(prog)

    print("모든 단계 완료.", flush=True)
    write_report(prog, cfg_all)
    with open(os.path.join(STATE, "DONE"), "w") as f:
        f.write("done\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    a = ap.parse_args()
    run(a.config)
