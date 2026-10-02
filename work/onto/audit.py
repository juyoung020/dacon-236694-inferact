# -*- coding: utf-8 -*-
"""그룹 내부 라벨 vs 전 피처 조건부 분포·정보량 전수조사 → audit_report.md

목적: ① 하드 룰 후보(조건부 정밀도 99%+, 지지도 50+) 발굴 ② 필러 신호 검증 ③ L2 피처 우선순위.
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
def _p(rel):  # 저장소 루트 기준 경로
    return _os.path.join(_ROOT, *[x for x in rel.split('\\') if x])
import math, pickle, re, sys
from collections import Counter, defaultdict
from pathlib import Path
from build_graph import GROUPS, GROUP_LIST

OUT = Path(_p(r"work\onto"))

FILLERS = ["여기부터", "이번 것만", "천천히", "가볍게", "꼼꼼히", "간단히", "빨리", "시간 될 때", "시간 되실 때",
           "한 번만", "한번 더", "먼저", "우선", "지금", "asap", "please", "plz", "thx", "thanks", "cheers",
           "ㅠ", "ㅎㅎ", "ㅋㅋ", "no pressure", "when you can", "when free", "real quick", "sorry to bug"]

def bucket(x, edges):
    for i, e in enumerate(edges):
        if x <= e:
            return i
    return len(edges)

def feat_values(s, q):
    """스텝 → {피처명: 값} (audit용 범주형)"""
    h = s["hist"]
    acts = [a for t, a, _ in h if t == "A"]
    last = acts[-1] if acts else "NONE"
    last2 = "|".join(acts[-2:]) if len(acts) >= 2 else "NONE"
    lastg = GROUPS.get(last, "NONE")
    res = next((r for t, _, r in reversed(h) if t == "A"), "")
    rst = ("PASS" if res.startswith(("PASS", "ok")) else "FAIL" if res.startswith("FAIL") else
           "ERROR" if res.startswith("ERROR") else "NONE")
    p = s["prompt"]
    f = {
        "last_action": last, "last2_path": last2, "last_group": lastg, "last_result": rst,
        "hist_len": str(len(acts)), "step": str(min(s["step"], 12)), "turn": str(min(s["turn"], 12)),
        "tier": s["tier"], "lang": s["lang"], "dirty": str(s["dirty"]), "ci": s["ci"],
        "mix": s["mix"], "open_n": str(min(s["open_n"], 3)), "au": str(s["au"]),
        "loc_b": str(bucket(s["loc"], q["loc"])), "budget_b": str(bucket(s["budget"], q["budget"])),
        "elapsed_b": str(bucket(s["elapsed"], q["elapsed"])),
        "prompt_len": str(bucket(len(p), [30, 60, 90, 130])),
    }
    for m in FILLERS:
        f["filler:" + m] = "1" if m in p else "0"
    return f

def mi_norm(pairs):
    """[(값, 라벨)] → 정규화 상호정보량 MI/H(Y)"""
    n = len(pairs)
    cv, cy, cvy = Counter(), Counter(), Counter()
    for v, y in pairs:
        cv[v] += 1; cy[y] += 1; cvy[(v, y)] += 1
    mi = sum(c / n * math.log((c / n) / ((cv[v] / n) * (cy[y] / n))) for (v, y), c in cvy.items())
    hy = -sum(c / n * math.log(c / n) for c in cy.values())
    return mi / hy if hy else 0.0

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    g = pickle.load(open(OUT / "graph.pkl", "rb"))
    steps = g["steps"]
    import numpy as np
    q = {k: list(np.quantile([s[k] for s in steps], [i / 8 for i in range(1, 8)])) for k in ["loc", "budget", "elapsed"]}

    feats = [feat_values(s, q) for s in steps]
    fnames = sorted(feats[0])
    lines = ["# 그룹 내부 신호 전수조사\n"]
    rule_cands = []
    for grp in GROUP_LIST:
        idx = [i for i, s in enumerate(steps) if GROUPS[s["label"]] == grp]
        lines.append(f"\n## {grp} (n={len(idx)}, 클래스 {sorted({steps[i]['label'] for i in idx})})\n")
        base = Counter(steps[i]["label"] for i in idx)
        maj = base.most_common(1)[0][1] / len(idx)
        lines.append(f"최빈 찍기 정확도: {maj:.3f}\n")
        scored = []
        for fn in fnames:
            pairs = [(feats[i][fn], steps[i]["label"]) for i in idx]
            scored.append((mi_norm(pairs), fn))
        scored.sort(reverse=True)
        lines.append("| 피처 | 정규화 MI |\n|---|---|\n")
        for s_, fn in scored[:15]:
            lines.append(f"| {fn} | {s_:.4f} |\n")
        # 룰 후보: 피처값 하나가 라벨을 99%+로 확정 (지지도 50+)
        for fn in fnames:
            byv = defaultdict(Counter)
            for i in idx:
                byv[feats[i][fn]][steps[i]["label"]] += 1
            for v, c in byv.items():
                tot = sum(c.values()); top, cnt = c.most_common(1)[0]
                if tot >= 50 and cnt / tot >= 0.99:
                    rule_cands.append((grp, fn, v, top, cnt, tot))
    lines.append("\n## 하드 룰 후보 (조건부 정밀도 ≥99%, 지지도 ≥50)\n")
    if rule_cands:
        lines.append("| 그룹 | 피처 | 값 | 라벨 | 적중/전체 |\n|---|---|---|---|---|\n")
        for r in rule_cands:
            lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]}/{r[5]} |\n")
    else:
        lines.append("(없음 — 단일 범주형 피처로는 99% 확정 조합 없음)\n")
    (OUT / "audit_report.md").write_text("".join(lines), encoding="utf-8")
    print("피처 상위 MI (그룹별 1~3위):")
    for grp in GROUP_LIST:
        idx = [i for i, s in enumerate(steps) if GROUPS[s["label"]] == grp]
        scored = sorted(((mi_norm([(feats[i][fn], steps[i]["label"]) for i in idx]), fn) for fn in fnames), reverse=True)
        print(f"  {grp:7}", " / ".join(f"{fn}={v:.3f}" for v, fn in scored[:3]))
    print(f"하드 룰 후보: {len(rule_cands)}개 → audit_report.md")

if __name__ == "__main__":
    main()
