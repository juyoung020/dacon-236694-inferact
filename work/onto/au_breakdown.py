# -*- coding: utf-8 -*-
"""au/sim 구간 클래스별 F1 분해 (최근 oof.pkl 기준)"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
def _p(rel):  # 저장소 루트 기준 경로
    return _os.path.join(_ROOT, *[x for x in rel.split('\\') if x])
import pickle, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, _p(r"work\onto"))
from build_graph import ACTIONS
from sklearn.metrics import f1_score

D = _p(r"work\onto")
d = pickle.load(open(D + r"\oof.pkl", "rb"))
ids, rl, P, y = d["ids"], d["rl"], np.asarray(d["P"]), np.array(d["y"])
off = np.load(D + r"\offsets.npy")
am = (np.log(P + 1e-9) + off).argmax(1)
pred = np.array([rl[i] if rl[i] is not None else ACTIONS[am[i]] for i in range(len(y))])
au = np.array([1 if i.startswith("sess_au") else 0 for i in ids], dtype=bool)

print(f"{'클래스':20} {'au-F1':>7} {'sim-F1':>7}")
fa = f1_score(y[au], pred[au], average=None, labels=ACTIONS)
fs = f1_score(y[~au], pred[~au], average=None, labels=ACTIONS)
for a, x, s in zip(ACTIONS, fa, fs):
    print(f"  {a:20} {x:.3f}  {s:.3f}")
print(f"\nau macro = {f1_score(y[au], pred[au], average='macro'):.4f} ({au.sum()}행) / "
      f"sim macro = {f1_score(y[~au], pred[~au], average='macro'):.4f} ({(~au).sum()}행)")
