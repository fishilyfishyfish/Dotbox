"""Dots & Boxes 규칙.

판은 rows x cols 개의 칸(box)으로 이루어지고, 선수는 아직 안 그어진 변(edge)을
하나씩 긋는다. 칸의 네 변이 모두 채워지면 그 칸을 딴 사람이 점수를 얻고
'한 번 더' 두게 된다. 모든 변이 그어지면 끝이고 칸을 많이 딴 쪽이 이긴다.

핵심 성질: 앞으로 남은 게임의 가치는 '지금까지 누가 몇 점인지'와 무관하고
오직 '어떤 변들이 그어져 있는지'(mask)에만 달려 있다. 그래서 상태를 정수
비트마스크 하나로 표현한다.

변 번호 매기기
  가로 변: (rows+1) x cols 개, 인덱스 r*cols + c
  세로 변: rows x (cols+1) 개, 인덱스 nH + r*(cols+1) + c
"""
from __future__ import annotations


class Game:
    def __init__(self, rows: int, cols: int):
        self.R, self.C = rows, cols
        self.nH = (rows + 1) * cols
        self.nV = rows * (cols + 1)
        self.E = self.nH + self.nV
        self.total_boxes = rows * cols
        self.full = (1 << self.E) - 1

        self.boxes = []      # 칸마다 변 4개의 인덱스
        self.box_mask = []   # 칸마다 그 4개 변의 비트마스크
        for r in range(rows):
            for c in range(cols):
                top = r * cols + c
                bot = (r + 1) * cols + c
                left = self.nH + r * (cols + 1) + c
                right = self.nH + r * (cols + 1) + c + 1
                es = (top, bot, left, right)
                self.boxes.append(es)
                m = 0
                for e in es:
                    m |= 1 << e
                self.box_mask.append(m)

        self.edge_boxes = [[] for _ in range(self.E)]   # 변 -> 맞닿은 칸들
        for bi, es in enumerate(self.boxes):
            for e in es:
                self.edge_boxes[e].append(bi)

    # ---------- 기본 동작 ----------
    def legal(self, mask: int):
        return [e for e in range(self.E) if not (mask >> e) & 1]

    def completed(self, mask: int, e: int) -> int:
        """변 e를 그었을 때 완성되는 칸의 개수 (0, 1, 2)."""
        nm = mask | (1 << e)
        n = 0
        for bi in self.edge_boxes[e]:
            bm = self.box_mask[bi]
            if nm & bm == bm:
                n += 1
        return n

    def step(self, mask: int, e: int):
        """(새 mask, 딴 칸 수, 같은 사람이 또 두는지)"""
        assert not (mask >> e) & 1, "이미 그어진 변"
        n = self.completed(mask, e)
        return mask | (1 << e), n, n > 0

    def is_over(self, mask: int) -> bool:
        return mask == self.full

    def box_counts(self, mask: int):
        """각 칸에 그어진 변의 수 (0~4)."""
        out = []
        for bi in range(self.total_boxes):
            bm = self.box_mask[bi]
            out.append(bin(mask & bm).count("1"))
        return out

    # ---------- 사람이 보기 위한 출력 ----------
    def render(self, mask: int, owner=None) -> str:
        on = lambda e: (mask >> e) & 1
        lines = []
        for r in range(self.R + 1):
            s = ""
            for c in range(self.C):
                s += "*" + ("---" if on(r * self.C + c) else "   ")
            lines.append(s + "*")
            if r < self.R:
                s = ""
                for c in range(self.C + 1):
                    s += "|" if on(self.nH + r * (self.C + 1) + c) else " "
                    if c < self.C:
                        bi = r * self.C + c
                        ch = " "
                        if owner is not None and owner[bi]:
                            ch = str(owner[bi])
                        s += f" {ch} "
                lines.append(s)
        return "\n".join(lines)
