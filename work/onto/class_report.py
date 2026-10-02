# -*- coding: utf-8 -*-
"""클래스별 상세 리포트: F1(전체/au/sim) + 혼동 방향 + 그룹 내부 정확도 (최신 oof.pkl 기준)"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
def _p(rel):  # 저장소 루트 기준 경로
    return _os.path.join(_ROOT, *[x for x in rel.split('\\') if x])
import pickle, sys
import numpy as np
sys.path.insert(0, _p(r"work\onto"))
from build_graph import ACTIONS, GROUPS, GROUP_LIST
from sklearn.metrics import f1_score, confusion_matrix

sys.stdout.reconfigure(encoding="utf-8")
D = _p(r"work\onto")
d = pickle.load(open(D + r"\oof.pkl", "rb"))
ids, rl, P, y = d["ids"], d["rl"], np.asarray(d["P"]), np.array(d["y"])
off = np.load(D + r"\offsets.npy")
am = (np.log(P + 1e-9) + off).argmax(1)
pred = np.array([rl[i] if rl[i] is not None else ACTIONS[am[i]] for i in range(len(y))])
au = np.array([1 if i.startswith("sess_au") else 0 for i in ids], dtype=bool)

f_all = f1_score(y, pred, average=None, labels=ACTIONS)
f_au = f1_score(y[au], pred[au], average=None, labels=ACTIONS)
f_sim = f1_score(y[~au], pred[~au], average=None, labels=ACTIONS)
cm = confusion_matrix(y, pred, labels=ACTIONS)

print(f"전체 Macro-F1 = {f1_score(y, pred, average='macro'):.4f} (au {f1_score(y[au], pred[au], average='macro'):.4f} / sim {f1_score(y[~au], pred[~au], average='macro'):.4f})\n")
print(f"{'클래스':18} {'n':>6} {'F1':>6} {'au':>6} {'sim':>6}  주요 오분류 방향 (정답→예측, 건수)")
order = np.argsort(f_all)
for j in order:
    a = ACTIONS[j]
    n = cm[j].sum()
    row = cm[j].copy(); row[j] = 0
    top = np.argsort(-row)[:2]
    conf = ", ".join(f"→{ACTIONS[t]} {row[t]}" for t in top if row[t] > 0)
    print(f"{a:18} {n:6d} {f_all[j]:.3f} {f_au[j]:.3f} {f_sim[j]:.3f}  {conf}")

print("\n그룹 내부 정확도 (그룹은 맞았는데 클래스를 틀린 비율):")
for g in GROUP_LIST:
    m = np.array([GROUPS[t] == g for t in y])
    same_g = np.array([GROUPS[p] == g for p in pred])
    within = m & same_g
    acc = (pred[within] == y[within]).mean()
    print(f"  {g:8} 행 {m.sum():6d} | 그룹적중 {same_g[m].mean():.3f} | 그룹적중시 클래스 정확도 {acc:.3f}")
