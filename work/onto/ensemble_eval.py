# -*- coding: utf-8 -*-
"""e5 시드 앙상블 CV 평가 — 여러 ft_full_oof*.pkl softmax 평균 → 코렉터 재학습 → 챔피언 대비.
variance 감소(디코이 bias 아님)라 LB 전이 가능성 높음. 존재하는 시드 파일 자동 감지."""
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
folds = json.load(open(OUT + r"\folds.json", encoding="utf-8")); fold_arr = np.array([folds[i] for i in ids])
offs = np.array(json.load(open(OUT + r"\corrector_w.json"))["offsets"])
rec = {}
for line in open(DATA, encoding="utf-8"):
    d = json.loads(line); rec[d["id"]] = d
hist = {i: C.parse_history(rec[i]["history"]) for i in ids}
metaX = np.hstack([C.build_seq_feats(hist, ids), MF.build_meta_feats(rec, ids)])

# 시드 OOF 파일들 (챔피언=ft_full_oof.pkl + 추가 시드)
files = [OUT + r"\ft_full_oof.pkl"] + sorted(glob.glob(OUT + r"\ft_full_oof_*.pkl"))
files = [f for f in files if os.path.exists(f)]
FFs = {}
for f in files:
    d = pickle.load(open(f, "rb"))
    if all(i in d for i in ids):
        FFs[os.path.basename(f)] = np.array([d[i] for i in ids], float)
    else:
        print(f"  (건너뜀: {os.path.basename(f)} — {sum(i in d for i in ids)}/{len(ids)}행만)", flush=True)
print(f"e5 OOF 로드: {list(FFs.keys())}", flush=True)


def evalFf(Ff, tag):
    P1 = 0.55 * P + 0.45 * Ff
    Q = np.zeros_like(P)
    for k in range(3):
        tr = fold_arr != k; va = fold_arr == k
        clf = newclf().fit(np.hstack([np.log(P1[tr] + 1e-9), metaX[tr]]), yi[tr])
        pk = clf.predict_proba(np.hstack([np.log(P1[va] + 1e-9), metaX[va]]))
        for ci, cls in enumerate(clf.classes_):
            Q[va, int(cls)] = pk[:, ci]
    best = (0, 0)
    for w2 in [0.35, 0.5]:
        blend = C.group_mask(P, (1 - w2) * P1 + w2 * Q)
        pred = np.array([ACTIONS[k] for k in (np.log(blend + 1e-9) + offs).argmax(1)], dtype=object)
        f = f1_score(y, pred, labels=ACTIONS, average="macro", zero_division=0)
        if f > best[0]:
            best = (f, w2)
    fa = f1_score(y[au], np.array([ACTIONS[k] for k in (np.log(C.group_mask(P, (1 - best[1]) * P1 + best[1] * Q) + 1e-9) + offs).argmax(1)])[au], labels=ACTIONS, average="macro", zero_division=0)
    print(f"  {tag:28} CV {best[0]:.4f} @w2={best[1]}", flush=True)
    return best[0]


keys = list(FFs.keys())
print("\n=== 단일 시드 ===", flush=True)
for k in keys:
    evalFf(FFs[k], k)
if len(keys) >= 2:
    print("\n=== 앙상블 (softmax 평균) ===", flush=True)
    for n in range(2, len(keys) + 1):
        ens = np.mean([FFs[k] for k in keys[:n]], axis=0)
        evalFf(ens, f"{n}-seed 평균 {keys[:n]}")
print("\n기준: 챔피언(단일 seed42) = 0.7714", flush=True)
