# -*- coding: utf-8 -*-
"""오라클 +0.024를 하베스트할 수 있나 — 마지막 검증.
(A) 챔피언이 '틀릴 때 확신하나'(디코이 서명)? 그러면 어떤 셀렉터도 구제케이스를 못 찾음.
(B) confidence/regime 게이트 오버라이드: traj가 신뢰구간에서 champ와 불일치할 때 traj로 교체. 챔피언 이기나."""
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
y = np.array(oof["y"]); yi = np.array([A_IDX[t] for t in y])
_ff = pickle.load(open(OUT + r"\ft_full_oof.pkl", "rb")); Ff = np.array([_ff[i] for i in ids], float)
folds = json.load(open(OUT + r"\folds.json", encoding="utf-8")); fold_arr = np.array([folds[i] for i in ids])
offs = np.array(json.load(open(OUT + r"\corrector_w.json"))["offsets"])
rec = {}
for line in open(DATA, encoding="utf-8"):
    d = json.loads(line); rec[d["id"]] = d
hist = {i: C.parse_history(rec[i]["history"]) for i in ids}
SEQ = C.build_seq_feats(hist, ids); metaX = np.hstack([SEQ, MF.build_meta_feats(rec, ids)])
P1 = 0.55 * P + 0.45 * Ff; EPS = 1e-9
Qc = np.zeros_like(P); trajP = np.zeros_like(P)
for k in range(3):
    tr = fold_arr != k; va = fold_arr == k
    m = newclf().fit(np.hstack([np.log(P1[tr] + EPS), metaX[tr]]), yi[tr])
    pk = m.predict_proba(np.hstack([np.log(P1[va] + EPS), metaX[va]]))
    for ci, cls in enumerate(m.classes_):
        Qc[va, int(cls)] = pk[:, ci]
    mt = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=40,
                                        class_weight="balanced", random_state=0).fit(SEQ[tr], yi[tr])
    pk = mt.predict_proba(SEQ[va])
    for ci, cls in enumerate(mt.classes_):
        trajP[va, int(cls)] = pk[:, ci]
champ_m = C.group_mask(P, 0.65 * P1 + 0.35 * Qc)
champ_logit = np.log(champ_m + EPS) + offs
champ_pred = np.array([ACTIONS[k] for k in champ_logit.argmax(1)], dtype=object)
traj_m = C.group_mask(P, trajP)
traj_pred = np.array([ACTIONS[k] for k in traj_m.argmax(1)], dtype=object)
champ_conf = (champ_m / np.maximum(champ_m.sum(1, keepdims=True), EPS)).max(1)
traj_conf = (traj_m / np.maximum(traj_m.sum(1, keepdims=True), EPS)).max(1)
cc = champ_pred == y


def mf1(pred): return f1_score(y, pred, labels=ACTIONS, average="macro", zero_division=0)


print(f"챔피언 macroF1 = {mf1(champ_pred):.4f}\n", flush=True)
# (A) 디코이 서명: 틀릴 때 확신하나?
print("=== (A) 챔피언 confidence: 맞을 때 vs 틀릴 때 ===", flush=True)
print(f"  정답행 평균확신 {champ_conf[cc].mean():.3f}  |  오답행 평균확신 {champ_conf[~cc].mean():.3f}", flush=True)
print(f"  오답인데 확신>0.7 비율 {(champ_conf[~cc] > 0.7).mean():.1%}  (높으면=confident-wrong=디코이, 셀렉터 불가)", flush=True)
# 구제케이스(champ오답&traj정답)에서 champ 확신
resc = (~cc) & (traj_pred == y)
print(f"  구제케이스({resc.sum()})에서 champ 평균확신 {champ_conf[resc].mean():.3f}, traj 평균확신 {traj_conf[resc].mean():.3f}", flush=True)

# (B) confidence-게이트 오버라이드 스윕
print("\n=== (B) 게이트 오버라이드: champ확신<c_lo & traj확신>t_hi & 불일치 → traj ===", flush=True)
dis = champ_pred != traj_pred
best = (mf1(champ_pred), None)
for c_lo in [0.5, 0.6, 0.7, 0.9]:
    for t_hi in [0.4, 0.5, 0.6, 0.7]:
        g = dis & (champ_conf < c_lo) & (traj_conf > t_hi)
        pred = champ_pred.copy(); pred[g] = traj_pred[g]
        f = mf1(pred)
        if f > best[0]: best = (f, (c_lo, t_hi, int(g.sum())))
print(f"  최고 게이트 {best[0]:.4f} @ {best[1]}  vs 챔피언 {mf1(champ_pred):.4f}  (Δ {best[0]-mf1(champ_pred):+.4f})", flush=True)
