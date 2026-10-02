# -*- coding: utf-8 -*-
"""오라클 다양성 하베스트 재시도 — 다중뷰 '합의' 셀렉터. 궤적+kNN이 서로 일치 & 챔피언과 불일치일 때
그 합의가 챔피언보다 정확한가? 맞으면 오버라이드로 하베스트 가능(디코이 잠금 우회 시도)."""
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
from sklearn.metrics import f1_score, accuracy_score
ACTIONS = C.ACTIONS; A_IDX = C.A_IDX
oof = pickle.load(open(OUT + r"\oof.pkl", "rb")); ids = oof["ids"]; P = np.asarray(oof["P"], float)
y = np.array(oof["y"]); yi = np.array([A_IDX[t] for t in y])
_ff = pickle.load(open(OUT + r"\ft_full_oof.pkl", "rb")); Ff = np.array([_ff[i] for i in ids], float)
folds = json.load(open(OUT + r"\folds.json", encoding="utf-8")); fold_arr = np.array([folds[i] for i in ids])
offs = np.array(json.load(open(OUT + r"\corrector_w.json"))["offsets"])
knnP = np.load(OUT + r"\_knnP_oof.npy")
rec = {}
for line in open(DATA, encoding="utf-8"):
    d = json.loads(line); rec[d["id"]] = d
hist = {i: C.parse_history(rec[i]["history"]) for i in ids}
SEQ = C.build_seq_feats(hist, ids); metaX = np.hstack([SEQ, MF.build_meta_feats(rec, ids)])
P1 = 0.55 * P + 0.45 * Ff
Q = np.zeros_like(P); trajP = np.zeros_like(P)
for k in range(3):
    tr = fold_arr != k; va = fold_arr == k
    m = newclf().fit(np.hstack([np.log(P1[tr] + 1e-9), metaX[tr]]), yi[tr])
    pk = m.predict_proba(np.hstack([np.log(P1[va] + 1e-9), metaX[va]]))
    for ci, cls in enumerate(m.classes_):
        Q[va, int(cls)] = pk[:, ci]
    mt = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=40,
                                        class_weight="balanced", random_state=0).fit(SEQ[tr], yi[tr])
    pk = mt.predict_proba(SEQ[va])
    for ci, cls in enumerate(mt.classes_):
        trajP[va, int(cls)] = pk[:, ci]
champ_m = C.group_mask(P, 0.65 * P1 + 0.35 * Q)
champ_pred = np.array([ACTIONS[k] for k in (np.log(champ_m + 1e-9) + offs).argmax(1)], dtype=object)
traj_pred = np.array([ACTIONS[k] for k in C.group_mask(P, trajP).argmax(1)], dtype=object)   # 챔피언 그룹으로 라우팅
knn_pred = np.array([ACTIONS[k] for k in C.group_mask(P, knnP).argmax(1)], dtype=object)
print("OOF 준비완료", flush=True)


def mf1(pred): return f1_score(y, pred, labels=ACTIONS, average="macro", zero_division=0)


cc = champ_pred == y
# 합의: traj==knn & != champ
cons = (traj_pred == knn_pred) & (traj_pred != champ_pred)
n = cons.sum()
cons_acc = accuracy_score(y[cons], traj_pred[cons]) if n else 0
champ_on_cons = accuracy_score(y[cons], champ_pred[cons]) if n else 0
print(f"\n합의(traj==kNN & !=champ) {n}행: 합의정답률 {cons_acc:.3f} vs 챔피언 {champ_on_cons:.3f}", flush=True)
print(f"챔피언 macroF1 {mf1(champ_pred):.4f}", flush=True)
# 오버라이드: 합의행을 합의로 교체
for tag, mask in [("합의 전체", cons),
                  ("합의 & 궤적confident", cons & (trajP.max(1) > 0.5)),
                  ("합의 & kNNconfident", cons & (knnP.max(1) > 0.5)),
                  ("합의 & 둘다conf", cons & (trajP.max(1) > 0.4) & (knnP.max(1) > 0.4))]:
    pred = champ_pred.copy(); pred[mask] = traj_pred[mask]
    print(f"  오버라이드[{tag}] ({mask.sum()}행): macroF1 {mf1(pred):.4f}  (Δ {mf1(pred)-mf1(champ_pred):+.4f})", flush=True)
