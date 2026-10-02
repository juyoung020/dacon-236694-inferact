# -*- coding: utf-8 -*-
"""au-specialist e5(ft_oof.pkl, au-only 학습) 재블렌드 — 챔피언 하드라우팅엔 미사용.
au 행에만 champ ⊕ au-e5 블렌드. au는 역사적으로 LB 전이되는 레버. au/전체 macroF1 스윕."""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
def _p(rel):  # 저장소 루트 기준 경로
    return _os.path.join(_ROOT, *[x for x in rel.split('\\') if x])
import json, pickle, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
OUT = _p(r"work\onto")
DATA = _p(r"data\data\train.jsonl")
sys.path.insert(0, OUT)
import corrector as C
import meta_feats as MF
from train_corrector import newclf
from sklearn.metrics import f1_score
ACTIONS = C.ACTIONS; A_IDX = C.A_IDX
oof = pickle.load(open(OUT + r"\oof.pkl", "rb")); ids = oof["ids"]; P = np.asarray(oof["P"], float)
y = np.array(oof["y"]); yi = np.array([A_IDX[t] for t in y]); au = np.array([i.startswith("sess_au") for i in ids])
_ff = pickle.load(open(OUT + r"\ft_full_oof.pkl", "rb")); Ff = np.array([_ff[i] for i in ids], float)
folds = json.load(open(OUT + r"\folds.json", encoding="utf-8")); fold_arr = np.array([folds[i] for i in ids])
offs = np.array(json.load(open(OUT + r"\corrector_w.json"))["offsets"])
ftau = pickle.load(open(OUT + r"\ft_oof.pkl", "rb"))
rec = {}
for line in open(DATA, encoding="utf-8"):
    d = json.loads(line); rec[d["id"]] = d
hist = {i: C.parse_history(rec[i]["history"]) for i in ids}
metaX = np.hstack([C.build_seq_feats(hist, ids), MF.build_meta_feats(rec, ids)])
P1 = 0.55 * P + 0.45 * Ff
Q = np.zeros_like(P)
for k in range(3):
    tr = fold_arr != k; va = fold_arr == k
    clf = newclf().fit(np.hstack([np.log(P1[tr] + 1e-9), metaX[tr]]), yi[tr])
    pk = clf.predict_proba(np.hstack([np.log(P1[va] + 1e-9), metaX[va]]))
    for ci, cls in enumerate(clf.classes_):
        Q[va, int(cls)] = pk[:, ci]
P2 = 0.65 * P1 + 0.35 * Q
champ_m = C.group_mask(P, P2)
au_e5 = np.zeros_like(P)
for j, i in enumerate(ids):
    if i in ftau:
        au_e5[j] = np.asarray(ftau[i], float)


def rep(Pb, tag):
    pred = np.array([ACTIONS[k] for k in (np.log(C.group_mask(P, Pb) + 1e-9) + offs).argmax(1)], dtype=object)
    fa = f1_score(y[au], pred[au], labels=ACTIONS, average="macro", zero_division=0)
    fs = f1_score(y[~au], pred[~au], labels=ACTIONS, average="macro", zero_division=0)
    ft = f1_score(y, pred, labels=ACTIONS, average="macro", zero_division=0)
    print(f"  {tag:22} 전체 {ft:.4f}  au {fa:.4f}  sim {fs:.4f}", flush=True)
    return ft


print("=== au 행에만 champ ⊕ au-e5 블렌드 스윕 ===", flush=True)
rep(champ_m, "champion (w=0)")
for w in [0.2, 0.3, 0.4, 0.5, 0.6]:
    Pb = champ_m.copy()
    m = au & (au_e5.sum(1) > 0)
    Pb[m] = (1 - w) * champ_m[m] + w * au_e5[m]
    rep(Pb, f"au-e5 w={w}")
print("기준 챔피언 0.7714 (au 0.8643)", flush=True)
