# -*- coding: utf-8 -*-
"""[0] 세션 매칭 + [1] 어휘 하드 룰.

세션 매칭: (세션, 정규화 프롬프트) → history 정답쌍 조회. train 실측 커버 86.5% / 정밀도 100%.
어휘 룰: 프롬프트 word n-gram(1~3) 중 정밀도 ≥99% & 지지도 ≥50 패턴만 채택.
중복제거는 같은 라벨이면 짧은 것 유지(커버 넓음), 적용은 긴(구체적) 패턴 우선.
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
def _p(rel):  # 저장소 루트 기준 경로
    return _os.path.join(_ROOT, *[x for x in rel.split('\\') if x])
import json, re, sys, pickle
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(_p(r"work\onto"))
_tok = re.compile(r"[A-Za-z가-힣0-9_./-]+")

def _ngrams(prompt, nmax=3):
    ws = _tok.findall(prompt.lower())
    for n in range(1, nmax + 1):
        for i in range(len(ws) - n + 1):
            yield " ".join(ws[i:i + n])

def mine_rules(steps, min_support=50, min_prec=0.99):
    """steps(라벨 포함) → [{pattern, label, prec, support}] (긴 패턴, 고정밀 순)"""
    cnt = defaultdict(Counter)
    for s in steps:
        for g in set(_ngrams(s["prompt"])):
            cnt[g][s["label"]] += 1
    rules = []
    for g, c in cnt.items():
        tot = sum(c.values())
        if tot < min_support:
            continue
        lab, k = c.most_common(1)[0]
        if k / tot >= min_prec:
            rules.append({"pattern": g, "label": lab, "prec": k / tot, "support": tot})
    # 중복 제거: 같은 라벨의 포함 패턴이 이미 있으면 더 짧은 것 우선(커버 넓음), 나머지 제거
    # 포함 판정은 토큰 경계 기준 (문자열 substring은 "가 나" ⊂ "가 나다" 같은 오판 가능)
    def _tok_contains(sub, seq):
        m = len(sub)
        return any(seq[i:i + m] == sub for i in range(len(seq) - m + 1))
    rules.sort(key=lambda r: (len(r["pattern"].split()), -r["support"]))
    kept = []
    for r in rules:
        rt = tuple(r["pattern"].split())
        if not any(k["label"] == r["label"] and _tok_contains(tuple(k["pattern"].split()), rt) for k in kept):
            kept.append(r)
    return kept

def compile_rules(rules):
    # 적용은 긴(=구체적) 패턴 우선: 다른 라벨의 룰이 겹치면 더 구체적인 매치가 이긴다.
    # 동률은 지지도 높은 순. (점수 영향 0 — 겹침 희소, 07-04 측정)
    return [(r["pattern"], r["label"])
            for r in sorted(rules, key=lambda r: (-len(r["pattern"].split()), -r["support"]))]

def apply_rules(prompt, compiled):
    toks = " " + " ".join(_tok.findall(prompt.lower())) + " "
    for pat, lab in compiled:
        if " " + pat + " " in toks:
            return lab
    return None

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    steps = pickle.load(open(OUT / "graph.pkl", "rb"))["steps"]
    rules = mine_rules(steps)
    comp = compile_rules(rules)
    hit = ok = 0
    for s in steps:
        p = apply_rules(s["prompt"], comp)
        if p is not None:
            hit += 1; ok += (p == s["label"])
    print(f"어휘 룰 {len(rules)}개: 커버 {hit}/{len(steps)} ({hit/len(steps):.1%}), 정밀도 {ok/max(hit,1):.2%}")
    json.dump(rules, open(OUT / "rules.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

if __name__ == "__main__":
    main()
