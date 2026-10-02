# -*- coding: utf-8 -*-
"""kNN 리트리벌 프라이어 = 학습셋 최근접 이웃(ctx임베딩)의 라벨분포. 비모수적 새 뷰.
OOF(fold별, 같은세션 제외)로 계산 → 챔피언과 보완성(구제율)·오라클 측정. 셀렉션 신호로 쓸만한가."""
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
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score, accuracy_score
ACTIONS = C.ACTIONS; A_IDX = C.A_IDX
oof = pickle.load(open(OUT + r"\oof.pkl", "rb")); ids = oof["ids"]; P = np.asarray(oof["P"], float)
y = np.array(oof["y"]); yi = np.array([A_IDX[t] for t in y]); au = np.array([i.startswith("sess_au") for i in ids])
_ff = pickle.load(open(OUT + r"\ft_full_oof.pkl", "rb")); Ff = np.array([_ff[i] for i in ids], float)
folds = json.load(open(OUT + r"\folds.json", encoding="utf-8")); fold_arr = np.array([folds[i] for i in ids])
offs = np.array(json.load(open(OUT + r"\corrector_w.json"))["offsets"])
sess = np.array([i.rsplit("-step_", 1)[0] for i in ids])
z = np.load(OUT + r"\ctx_emb_frozen.npz", allow_pickle=True)
emap = {i: e for i, e in zip(z["ids"].tolist(), z["emb"])}
E = np.array([emap[i] for i in ids], np.float32)
print(f"emb {E.shape}", flush=True)

# 챔피언 OOF
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
champ = C.group_mask(P, 0.65 * P1 + 0.35 * Q)
champ_pred = np.array([ACTIONS[k] for k in (np.log(champ + 1e-9) + offs).argmax(1)], dtype=object)
print("챔피언 OOF 완료", flush=True)

# kNN 프라이어 OOF (fold별, 같은세션 제외)
K = 100
knnP = np.zeros((len(ids), 14))
for k in range(3):
    tri = np.where(fold_arr != k)[0]; vai = np.where(fold_arr == k)[0]
    nn = NearestNeighbors(n_neighbors=min(K, len(tri)), metric="cosine").fit(E[tri])
    dist, nbr = nn.kneighbors(E[vai])
    for r, vi in enumerate(vai):
        nb = tri[nbr[r]]
        keep = nb[sess[nb] != sess[vi]][:50]        # 같은세션 제외, 상위50
        w = 1.0
        for j in keep:
            knnP[vi, yi[j]] += w
    print(f"  fold{k} kNN done", flush=True)
knnP /= np.maximum(knnP.sum(1, keepdims=True), 1e-9)
knn_pred = np.array([ACTIONS[k] for k in knnP.argmax(1)], dtype=object)

cc = champ_pred == y; kc = knn_pred == y
print(f"\n챔피언 acc {cc.mean():.4f}  kNN acc {kc.mean():.4f}", flush=True)
print(f"구제(챔피언오답→kNN정답) {((~cc) & kc).sum()}/{(~cc).sum()} = {((~cc)&kc).sum()/max((~cc).sum(),1):.3f}", flush=True)
oracle = (cc | kc).mean()
print(f"오라클 acc(champ|kNN) {oracle:.4f}", flush=True)
# 그룹별 kNN acc (kNN이 어디서 강한가)
GROUPS = {"SEARCH": ["read_file","grep_search","list_directory","glob_pattern"], "MODIFY": ["edit_file","write_file","apply_patch"],
          "EXEC": ["run_bash","run_tests","lint_or_typecheck"], "TALK": ["ask_user","plan_task","web_search","respond_only"]}
for G, mem in GROUPS.items():
    m = np.isin(y, mem)
    print(f"  [{G}] champ {accuracy_score(y[m],champ_pred[m]):.3f}  kNN {accuracy_score(y[m],knn_pred[m]):.3f}  au{'' if G!='SEARCH' else ''}", flush=True)
# 스태커: 챔피언 ⊕ kNN prior
np.save(OUT + r"\_knnP_frozen.npy", knnP)
print("\n=== 스태커 [champ P2 | kNN prior | metaX] OOF ===", flush=True)
Qs = np.zeros_like(P)
for k in range(3):
    tr = fold_arr != k; va = fold_arr == k
    X = lambda ix: np.hstack([np.log(0.65 * P1[ix] + 0.35 * Q[ix] + 1e-9), np.log(knnP[ix] + 1e-9), metaX[ix]])
    clf = newclf().fit(X(tr), yi[tr]); pk = clf.predict_proba(X(va))
    for ci, cls in enumerate(clf.classes_):
        Qs[va, int(cls)] = pk[:, ci]
for w2 in [0.35, 0.5]:
    blend = C.group_mask(P, (1 - w2) * P1 + w2 * Qs)
    pred = np.array([ACTIONS[k] for k in (np.log(blend + 1e-9) + offs).argmax(1)], dtype=object)
    print(f"  w2={w2}: 전체 {f1_score(y,pred,labels=ACTIONS,average='macro',zero_division=0):.4f}", flush=True)
print("기준 챔피언 0.7714", flush=True)
