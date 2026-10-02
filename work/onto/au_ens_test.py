# -*- coding: utf-8 -*-
"""au 앙상블 재블렌드 평가 — ft_oof*.pkl(여러 au seed) softmax 평균 → au 행 재블렌드.
단일 au-e5(+0.0051) 대비 au 앙상블이 au를 더 올리나. 존재하는 au OOF 자동 감지."""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
def _p(rel):  # 저장소 루트 기준 경로
    return _os.path.join(_ROOT, *[x for x in rel.split('\\') if x])
import json, pickle, sys, glob, os
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
champ_m = C.group_mask(P, 0.65 * P1 + 0.35 * Q)

# au OOF들 로드 (ft_oof.pkl=seed42 + ft_oof_N.pkl)
aufiles = [OUT + r"\ft_oof.pkl"] + sorted(glob.glob(OUT + r"\ft_oof_*.pkl"))
aufiles = [f for f in aufiles if os.path.exists(f)]
AUS = {}
for f in aufiles:
    d = pickle.load(open(f, "rb"))
    arr = np.zeros_like(P)
    for j, i in enumerate(ids):
        if i in d:
            arr[j] = np.asarray(d[i], float)
    AUS[os.path.basename(f)] = arr
print(f"au OOF: {list(AUS.keys())}", flush=True)


def rep(au_e5, tag, w=0.3):
    Pb = champ_m.copy()
    m = au & (au_e5.sum(1) > 0)
    Pb[m] = (1 - w) * champ_m[m] + w * au_e5[m]
    pred = np.array([ACTIONS[k] for k in (np.log(C.group_mask(P, Pb) + 1e-9) + offs).argmax(1)], dtype=object)
    fa = f1_score(y[au], pred[au], labels=ACTIONS, average="macro", zero_division=0)
    ft = f1_score(y, pred, labels=ACTIONS, average="macro", zero_division=0)
    print(f"  {tag:32} au {fa:.4f}  전체 {ft:.4f}", flush=True)


print("기준 챔피언 au 0.8643 / 전체 0.7714", flush=True)
keys = list(AUS.keys())
print("=== 단일 au seed ===", flush=True)
for k in keys:
    rep(AUS[k], k)
if len(keys) >= 2:
    print("=== au 앙상블 (softmax 평균) ===", flush=True)
    for n in range(2, len(keys) + 1):
        ens = np.mean([AUS[k] for k in keys[:n]], axis=0)
        rep(ens, f"{n}-seed 평균")
    # 최적 w 스윕 (전체 앙상블)
    ens = np.mean(list(AUS.values()), axis=0)
    print("=== 전체 au앙상블 w 스윕 ===", flush=True)
    for w in [0.2, 0.3, 0.4, 0.5, 0.6]:
        rep(ens, f"w={w}", w)
