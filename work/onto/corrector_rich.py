# -*- coding: utf-8 -*-
"""코렉터 피처 강화 — 지금 코렉터는 last result를 5-cat로만 씀. result_summary의 '숫자'(비-디코이 사실)와
깊은 result/action 히스토리를 추가하면 CV 오르나. 챔피언 코렉터[logP1|metaX] vs 강화[logP1|metaX|extra]."""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
def _p(rel):  # 저장소 루트 기준 경로
    return _os.path.join(_ROOT, *[x for x in rel.split('\\') if x])
import json, pickle, sys, re
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


def num(pat, s):
    m = re.search(pat, s.lower()); return float(m.group(1)) if m else 0.0


def rich_one(i):
    names, argts, rcats = hist[i]
    d = rec[i]["history"]
    lastres = ""
    for e in reversed(d):
        if e.get("role") == "assistant_action":
            lastres = (e.get("result_summary") or ""); break
    # 숫자 추출 (비-디코이 사실)
    f = [num(r"(\d+) match", lastres), num(r"(\d+) files?", lastres), num(r"\((\d+)l\)", lastres),
         num(r"(\d+) tests? (?:failing|failed)", lastres), num(r"(\d+) tests? passed", lastres),
         num(r"(\d+) entries", lastres), num(r"\((\d+)\+/", lastres), num(r"(\d+) errors?", lastres),
         num(r"(\d+) steps?", lastres), num(r"exit\s*=\s*(\d+)", lastres)]
    # 깊은 rcat 히스토리
    rc = lambda k: (rcats[-k] if len(rcats) >= k else 4)
    rh = np.zeros(15)  # rcat[-2],[-3] onehot(5 each) + rcat 카운트(5)
    if len(rcats) >= 2: rh[rc(2)] = 1
    if len(rcats) >= 3: rh[5 + rc(3)] = 1
    for c in rcats: rh[10 + c] += 1
    # prev5,6 action onehot
    a5 = np.zeros(14); a6 = np.zeros(14)
    if len(names) >= 5: a5[A_IDX[names[-5]]] = 1
    if len(names) >= 6: a6[A_IDX[names[-6]]] = 1
    return np.concatenate([np.log1p(f), rh, a5, a6])


EXTRA = np.vstack([rich_one(i) for i in ids])
print(f"강화 extra {EXTRA.shape}", flush=True)
richX = np.hstack([metaX, EXTRA])
P1 = 0.55 * P + 0.45 * Ff


def evalX(X, tag):
    Q = np.zeros_like(P)
    for k in range(3):
        tr = fold_arr != k; va = fold_arr == k
        clf = newclf().fit(np.hstack([np.log(P1[tr] + 1e-9), X[tr]]), yi[tr])
        pk = clf.predict_proba(np.hstack([np.log(P1[va] + 1e-9), X[va]]))
        for ci, cls in enumerate(clf.classes_):
            Q[va, int(cls)] = pk[:, ci]
    best = 0
    for w2 in [0.35, 0.5]:
        blend = C.group_mask(P, (1 - w2) * P1 + w2 * Q)
        pred = np.array([ACTIONS[k] for k in (np.log(blend + 1e-9) + offs).argmax(1)], dtype=object)
        best = max(best, f1_score(y, pred, labels=ACTIONS, average="macro", zero_division=0))
    print(f"  {tag:28} CV {best:.4f}", flush=True)
    return best


print("=== 챔피언 코렉터 vs 강화 ===", flush=True)
evalX(metaX, "champion [logP1|metaX]")
evalX(richX, "강화 [logP1|metaX|extra]")
print("기준 챔피언 0.7714", flush=True)
