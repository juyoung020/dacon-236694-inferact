# -*- coding: utf-8 -*-
"""다양성 하베스트 — 코렉터 입력에 순수 궤적모델 예측(log trajP)을 추가한 스태커가 챔피언을 이기나.
챔피언 코렉터 = HGB[log(P1) | seqX]. 스태커 = HGB[log(P1) | log(trajP) | seqX].
trajP = 프롬프트-blind 궤적 HGB(디코이 면역). 오라클 상한 +0.024를 얼마나 하베스트하나."""
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
from sklearn.ensemble import HistGradientBoostingClassifier
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
SEQ = C.build_seq_feats(hist, ids)
metaX = np.hstack([SEQ, MF.build_meta_feats(rec, ids)])
P1 = 0.55 * P + 0.45 * Ff
EPS = 1e-9


def hgb_oof(Xbuild):
    """Xbuild(tr_idx)->X. 3fold OOF 14-확률."""
    R = np.zeros_like(P)
    for k in range(3):
        tr = fold_arr != k; va = fold_arr == k
        m = newclf().fit(Xbuild(tr), yi[tr])
        pk = m.predict_proba(Xbuild(va))
        for ci, cls in enumerate(m.classes_):
            R[va, int(cls)] = pk[:, ci]
    return R


# 순수 궤적모델 (프롬프트-blind, SEQ 67차원)
trajP = np.zeros_like(P)
for k in range(3):
    tr = fold_arr != k; va = fold_arr == k
    m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
                                       min_samples_leaf=40, class_weight="balanced", random_state=0).fit(SEQ[tr], yi[tr])
    pk = m.predict_proba(SEQ[va])
    for ci, cls in enumerate(m.classes_):
        trajP[va, int(cls)] = pk[:, ci]
print("궤적모델 OOF 완료", flush=True)

# 코렉터(챔피언 재현) vs 스태커(+log trajP)
Qc = hgb_oof(lambda ix: np.hstack([np.log(P1[ix] + EPS), metaX[ix]]))               # 챔피언 코렉터
Qs = hgb_oof(lambda ix: np.hstack([np.log(P1[ix] + EPS), np.log(trajP[ix] + EPS), metaX[ix]]))  # +traj 스태커
print("코렉터/스태커 OOF 완료\n", flush=True)


def evalQ(Q, tag):
    print(f"  [{tag}]", flush=True)
    best = (0, 0)
    for w2 in [0.35, 0.5, 0.65]:
        blend = C.group_mask(P, (1 - w2) * P1 + w2 * Q)
        pred = np.array([ACTIONS[k] for k in (np.log(blend + EPS) + offs).argmax(1)], dtype=object)
        f = f1_score(y, pred, labels=ACTIONS, average="macro", zero_division=0)
        fa = f1_score(y[au], pred[au], labels=ACTIONS, average="macro", zero_division=0)
        fs = f1_score(y[~au], pred[~au], labels=ACTIONS, average="macro", zero_division=0)
        print(f"    w2={w2}: 전체 {f:.4f}  (au {fa:.4f} / sim {fs:.4f})", flush=True)
        best = max(best, (f, w2))
    return best


print("=== 챔피언 코렉터 (기준) ===", flush=True)
bc = evalQ(Qc, "corrector [logP1|seqX]")
print("=== 스태커 (+log trajP) ===", flush=True)
bs = evalQ(Qs, "stacker [logP1|log trajP|seqX]")
print(f"\n>>> 챔피언 {bc[0]:.4f} → 스태커 {bs[0]:.4f}  (Δ {bs[0]-bc[0]:+.4f})", flush=True)
