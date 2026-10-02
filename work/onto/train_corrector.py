# -*- coding: utf-8 -*-
"""[코렉터 학습] base(oof.pkl) + 풀ft(ft_full_oof.pkl) 위에 순서/args 코렉터 학습.
- 3-fold OOF로 w1/w2/offsets 튜닝 + 점수 검증(≈0.7538)
- 전체 70k로 최종 코렉터 학습 → corrector.pkl, corrector_w.json 저장 (추론이 읽음)
사용: python train_corrector.py
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
def _p(rel):  # 저장소 루트 기준 경로
    return _os.path.join(_ROOT, *[x for x in rel.split('\\') if x])
import json, pickle, sys, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
sys.path.insert(0, _p(r"work\onto"))
import corrector as C
ACTIONS = C.ACTIONS; A_IDX = C.A_IDX; eps = 1e-9
OUT = _p(r"work\onto")
DATA = _p(r"data\data\train.jsonl")

def final_preds(rlv, Pm, off):
    logP = np.log(Pm + eps) + off; am = logP.argmax(1)
    return np.array([rlv[i] if rlv[i] is not None else ACTIONS[am[i]] for i in range(len(Pm))])
def tune_offsets(rlv, Pm, yv):
    off = np.zeros(14); best = f1_score(yv, final_preds(rlv, Pm, off), average="macro")
    for _ in range(2):
        for c in range(14):
            for v in np.arange(-0.8, 0.81, 0.1):
                tr = off.copy(); tr[c] = v; f = f1_score(yv, final_preds(rlv, Pm, tr), average="macro")
                if f > best: best, off = f, tr
    return off, best

def newclf():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
                                          early_stopping=True, random_state=0, class_weight="balanced")

def main():
    sys.stdout.reconfigure(encoding="utf-8"); t0 = time.time()
    oof = pickle.load(open(OUT + r"\oof.pkl", "rb"))
    ids = oof["ids"]; P = np.asarray(oof["P"], float); y = np.array(oof["y"]); rl = oof["rl"]
    yi = np.array([A_IDX[t] for t in y])
    ff = pickle.load(open(OUT + r"\ft_full_oof.pkl", "rb"))
    FF = np.array([ff[i] for i in ids], float)
    folds = json.load(open(OUT + r"\folds.json", encoding="utf-8")); fold_arr = np.array([folds[i] for i in ids])
    # 순서/args 피처 (raw jsonl 재파싱, corrector.py 공유 함수)
    hist_by_id = {}
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line); hist_by_id[d["id"]] = C.parse_history(d["history"])
    seqX = C.build_seq_feats(hist_by_id, ids)
    print(f"parsed+feats {seqX.shape} ({time.time()-t0:.0f}s)")

    _, baseB = tune_offsets(rl, P, y); print(f"[base] {baseB:.4f}")
    # w1: base+ftfull
    best1 = (baseB, 0.0)
    for w1 in [0.3, 0.45, 0.55]:
        _, B = tune_offsets(rl, (1 - w1) * P + w1 * FF, y)
        print(f"  base+ftfull w1={w1}: {B:.4f}")
        if B > best1[0]: best1 = (B, w1)
    w1 = best1[1]; P1 = (1 - w1) * P + w1 * FF
    print(f"  -> w1={w1} ({best1[0]:.4f})")
    # 3-fold OOF 코렉터로 w2/offsets 튜닝
    Q = np.zeros_like(P)
    for k in range(3):
        tr = fold_arr != k; va = fold_arr == k
        clf = newclf(); clf.fit(C.stack_input(P1[tr], seqX[tr]), yi[tr])
        Q[va] = C.corrector_proba(clf, P1[va], seqX[va])
    print(f"[corrector OOF] ({time.time()-t0:.0f}s)")
    best = (best1[0], 0.0, tune_offsets(rl, P1, y)[0])
    for w2 in [0.2, 0.35, 0.5, 0.65]:
        off, B = tune_offsets(rl, (1 - w2) * P1 + w2 * Q, y)
        print(f"  +corrector w2={w2}: {B:.4f} (Δbase {B-baseB:+.4f})")
        if B > best[0]: best = (B, w2, off)
    fB, w2, offB = best
    print(f"\n>>> OOF 검증 최고 CV = {fB:.4f} (base {baseB:.4f}, Δ{fB-baseB:+.4f}) [w1={w1}, w2={w2}]")
    # 전체 데이터로 최종 코렉터 학습 (배포용)
    final = newclf(); final.fit(C.stack_input(P1, seqX), yi)
    pickle.dump(final, open(OUT + r"\corrector.pkl", "wb"))
    json.dump({"w1": float(w1), "w2": float(w2), "offsets": offB.tolist()},
              open(OUT + r"\corrector_w.json", "w"))
    print(f"→ corrector.pkl + corrector_w.json 저장 ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
