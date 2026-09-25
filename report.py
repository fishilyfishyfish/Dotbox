"""성적표를 사람이 보기 좋은 형태로 남긴다 (추가 설치 없이 SVG + 마크다운)."""
from __future__ import annotations
import os, json

STATE = "state"
COLORS = {"exact_match_mcts": "#3b82f6", "vs_random": "#10b981",
          "vs_greedy": "#f59e0b", "vs_prev_winrate": "#ef4444"}
LABEL = {"exact_match_mcts": "정답 일치율", "vs_random": "무작위 상대 승률",
         "vs_greedy": "욕심쟁이 상대 승률", "vs_prev_winrate": "이전 버전 상대 승률"}


def _svg(series, title):
    W, H, PAD = 720, 300, 46
    n = max((len(v) for v in series.values()), default=0)
    if n < 2:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="60">' \
               f'<text x="10" y="35" fill="#888" font-family="monospace">{title}: 자료가 아직 적음</text></svg>'
    x = lambda i: PAD + i * (W - 2 * PAD) / (n - 1)
    y = lambda v: H - PAD - v * (H - 2 * PAD)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'font-family="ui-monospace,monospace" font-size="11">',
           f'<rect width="{W}" height="{H}" fill="#14161a"/>']
    for g in (0, .25, .5, .75, 1):
        out.append(f'<line x1="{PAD}" y1="{y(g):.1f}" x2="{W-PAD}" y2="{y(g):.1f}" stroke="#2a2e35"/>')
        out.append(f'<text x="8" y="{y(g)+4:.1f}" fill="#7a828e">{int(g*100)}%</text>')
    out.append(f'<line x1="{PAD}" y1="{y(0.5):.1f}" x2="{W-PAD}" y2="{y(0.5):.1f}" stroke="#444a55" stroke-dasharray="4 4"/>')
    for k, vals in series.items():
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals) if v is not None)
        if pts:
            out.append(f'<polyline points="{pts}" fill="none" stroke="{COLORS[k]}" stroke-width="2"/>')
    for j, k in enumerate(series):
        cx = PAD + j * 165
        out.append(f'<rect x="{cx}" y="8" width="9" height="9" fill="{COLORS[k]}"/>')
        out.append(f'<text x="{cx+14}" y="17" fill="#c7ccd4">{LABEL[k]}</text>')
    out.append(f'<text x="{W-PAD}" y="{H-14}" fill="#7a828e" text-anchor="end">{title} · 가로축 = 반복 회차</text>')
    out.append("</svg>")
    return "\n".join(out)


def write_report(prog, cfg_all):
    os.makedirs(STATE, exist_ok=True)
    hist = prog.get("history", [])
    stages = []
    for st in cfg_all["stages"]:
        rows = [h for h in hist if h["stage"] == st["name"]]
        if rows:
            stages.append((st["name"], rows))

    for name, rows in stages:
        series = {k: [r.get(k) for r in rows] for k in
                  ("exact_match_mcts", "vs_random", "vs_greedy", "vs_prev_winrate")
                  if any(r.get(k) is not None for r in rows)}
        with open(os.path.join(STATE, f"chart_{name}.svg"), "w", encoding="utf-8") as f:
            f.write(_svg(series, name))

    md = ["# 학습 성적표", ""]
    if not stages:
        md.append("아직 기록이 없음.")
    for name, rows in stages:
        last = rows[-1]
        md += [f"## {name}", "",
               f"![{name}](chart_{name}.svg)", "",
               f"- 반복 {len(rows)}회차까지 진행",
               f"- 정답 일치율: **{_pc(last.get('exact_match_mcts'))}** (탐색) / "
               f"{_pc(last.get('exact_match_raw'))} (신경망 직감)",
               f"- 무작위 상대 승률: **{_pc(last.get('vs_random'))}**",
               f"- 욕심쟁이 상대 승률: **{_pc(last.get('vs_greedy'))}**",
               f"- 마지막 손실: 정책 {last.get('loss_policy')} / 가치 {last.get('loss_value')}",
               f"- 신경망이 교체된 횟수: {sum(1 for r in rows if r.get('promoted'))}회", ""]
        md += ["<details><summary>회차별 기록</summary>", "",
               "| 회차 | 정답일치 | 무작위 | 욕심쟁이 | 이전판대비 | 교체 | 정책손실 | 가치손실 | 초 |",
               "|---|---|---|---|---|---|---|---|---|"]
        for r in rows[-40:]:
            md.append(f"| {r['iter']} | {_pc(r.get('exact_match_mcts'))} | {_pc(r.get('vs_random'))} | "
                      f"{_pc(r.get('vs_greedy'))} | {_pc(r.get('vs_prev_winrate'))} | "
                      f"{'O' if r.get('promoted') else ''} | {r.get('loss_policy')} | "
                      f"{r.get('loss_value')} | {r.get('sec')} |")
        md += ["", "</details>", ""]
    md += ["---", "",
           "**보는 법**", "",
           "- 정답 일치율: 완전 탐색으로 구한 '진짜 정답'과 얼마나 맞는지. 작은 판에서만 잴 수 있음. 이게 오르면 제대로 배우는 중.",
           "- 무작위 상대 승률: 금방 100%가 됨. 여기서 안 오르면 학습이 아예 안 도는 것.",
           "- 욕심쟁이 상대 승률: 칸을 딸 수 있으면 무조건 따는 단순한 상대. 이걸 넘기려면 '일부러 내주고 나중에 크게 먹기'를 배워야 함.",
           "- 이전 버전 상대 승률: 55%를 넘으면 새 신경망으로 교체됨. 계속 50% 근처면 더 이상 늘지 않는 중."]
    with open(os.path.join(STATE, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def _pc(v):
    return "-" if v is None else f"{v*100:.0f}%"
