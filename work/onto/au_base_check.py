# -*- coding: utf-8 -*-
"""au BASE-variant 신호 검사. au id=sess_au_{BASE}_{V}-step_{KK}. 같은 BASE의 변형들이 라벨 상관?
같은 (BASE,step) 변형 간 라벨 일치도 vs au 전체 prior. 강하면 au 부스트 가능(단 test-base 겹침 필요=전이불확실)."""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
def _p(rel):  # 저장소 루트 기준 경로
    return _os.path.join(_ROOT, *[x for x in rel.split('\\') if x])
import re, pickle, sys
import numpy as np
from collections import defaultdict, Counter
sys.stdout.reconfigure(encoding="utf-8")
OUT = _p(r"work\onto")
oof = pickle.load(open(OUT + r"\oof.pkl", "rb")); ids = oof["ids"]; y = np.array(oof["y"])
au = np.array([i.startswith("sess_au") for i in ids])
auids = [ids[i] for i in range(len(ids)) if au[i]]
auy = [y[i] for i in range(len(ids)) if au[i]]
print(f"au {len(auids)}행", flush=True)

# 파싱: sess_au_{BASE}_{V}-step_{KK}
def parse(i):
    m = re.match(r"sess_au_(\d+)_(\d+)-step_(\d+)", i)
    return (m.group(1), int(m.group(2)), int(m.group(3))) if m else (None, None, None)

base_step = defaultdict(list)   # (BASE, step) -> [labels]
base_all = defaultdict(list)    # BASE -> [labels]
for i, lab in zip(auids, auy):
    b, v, k = parse(i)
    if b:
        base_step[(b, k)].append(lab); base_all[b].append(lab)

# au 전체 prior 최빈
prior = Counter(auy).most_common(1)[0]
print(f"au prior 최빈 = {prior[0]} {prior[1]/len(auy):.3f}", flush=True)

# (BASE,step) 그룹: 2개+ 변형 있는 그룹의 라벨 순도
multi = {k: v for k, v in base_step.items() if len(v) >= 2}
print(f"(BASE,step) 그룹 {len(base_step)}개, 변형2+ 그룹 {len(multi)}개", flush=True)
if multi:
    purities = [Counter(v).most_common(1)[0][1] / len(v) for v in multi.values()]
    covered = sum(len(v) for v in multi.values())
    print(f"  변형2+ 그룹 평균 최빈순도 = {np.mean(purities):.3f} (au prior 대비 신호)", flush=True)
    print(f"  변형2+ 커버 행수 = {covered}/{len(auids)} ({covered/len(auids):.0%})", flush=True)
    # LOO 예측: 각 행을 같은(BASE,step) 다른변형 최빈으로 예측 → 정확도
    correct = tot = 0
    for (b, k), labs in multi.items():
        for j in range(len(labs)):
            others = labs[:j] + labs[j+1:]
            if others:
                pred = Counter(others).most_common(1)[0][0]
                correct += (pred == labs[j]); tot += 1
    print(f"  LOO (같은BASE,step 다른변형 최빈) 정확도 = {correct/max(tot,1):.3f}  (n={tot})", flush=True)

# BASE 단위(step 무관): 같은 BASE 다른변형 LOO
multiB = {k: v for k, v in base_all.items() if len(v) >= 2}
correct = tot = 0
for b, labs in multiB.items():
    for j in range(len(labs)):
        others = labs[:j] + labs[j+1:]
        pred = Counter(others).most_common(1)[0][0]
        correct += (pred == labs[j]); tot += 1
print(f"BASE단위 LOO 정확도 = {correct/max(tot,1):.3f} (n={tot})  vs au prior {prior[1]/len(auy):.3f}", flush=True)
