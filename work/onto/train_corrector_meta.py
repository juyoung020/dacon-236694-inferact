# -*- coding: utf-8 -*-
"""[코렉터+메타] 기존 코렉터 seqX에 meta_feats(open_files 미개척신호) 스택 → 재학습.
w1=0.45 고정, w2는 LB예측(0.86·sim+0.14·au) 최대로 선택(raw CV가 aut을 깎는 w2 회피).
저장: corrector.pkl + corrector_w.json (pack.py가 읽음). CPU ~5분.
사용: python train_corrector_meta.py"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
def _p(rel):  # 저장소 루트 기준 경로
    return _os.path.join(_ROOT, *[x for x in rel.split('\\') if x])
import json, pickle, sys, time
import numpy as np
OUT = _p(r"work\onto")
DATA = _p(r"data\data\train.jsonl")
sys.path.insert(0, OUT)
import corrector as C
import meta_feats as MF
from train_corrector import tune_offsets, newclf, final_preds
from sklearn.metrics import f1_score
ACTIONS = C.ACTIONS; A_IDX = C.A_IDX
BASE_AU, BASE_SIM = 0.8472, 0.7505   # 메타없는 코렉터 기준(회귀 감지용)

def main():
    sys.stdout.reconfigure(encoding="utf-8"); t0 = time.time()
    oof = pickle.load(open(OUT + r"\oof.pkl", "rb"))
    ids = oof["ids"]; P = np.asarray(oof["P"], float); y = np.array(oof["y"]); rl = oof["rl"]
    yi = np.array([A_IDX[t] for t in y])
    ff = pickle.load(open(OUT + r"\ft_full_oof.pkl", "rb")); FF = np.array([ff[i] for i in ids], float)
    folds = json.load(open(OUT + r"\folds.json", encoding="utf-8")); fold_arr = np.array([folds[i] for i in ids])
    au = np.array([i.startswith("sess_au") for i in ids])
    rec = {}
    for line in open(DATA, encoding="utf-8"):
        d = json.loads(line); rec[d["id"]] = d
    hist_by_id = {i: C.parse_history(rec[i]["history"]) for i in ids}
    seqX = np.hstack([C.build_seq_feats(hist_by_id, ids), MF.build_meta_feats(rec, ids)])
    print(f"seqX(메타포함) {seqX.shape} ({time.time()-t0:.0f}s)")

    w1 = 0.45; P1 = (1 - w1) * P + w1 * FF
    Q = np.zeros_like(P)
    for k in range(3):
        tr = fold_arr != k; va = fold_arr == k
        clf = newclf(); clf.fit(C.stack_input(P1[tr], seqX[tr]), yi[tr]); Q[va] = C.corrector_proba(clf, P1[va], seqX[va])
    print(f"코렉터 OOF ({time.time()-t0:.0f}s)")

    none_rl = [None] * len(rl)          # 룰 오버라이드 제거(07-06): 오프셋을 룰 없이 튜닝(순수 argmax 기준)
    best = (-1, None, None)   # (LBpred, w2, off)
    for w2 in [0.35, 0.5, 0.65]:
        blend = C.group_mask(P, (1 - w2) * P1 + w2 * Q)     # 하드라우팅: base P의 g*로 최종 그룹 고정
        off, B = tune_offsets(none_rl, blend, y)
        pred = final_preds(none_rl, blend, off)
        fau = f1_score(y[au], pred[au], average="macro"); fsim = f1_score(y[~au], pred[~au], average="macro")
        lbp = 0.86 * (fsim - BASE_SIM) + 0.14 * (fau - BASE_AU)
        print(f"  w2={w2}: CV {B:.4f} au {fau:.4f} sim {fsim:.4f}  LB예측Δ {lbp:+.4f}")
        if lbp > best[0]: best = (lbp, w2, off)
    lbp, w2, off = best
    print(f"\n>>> 선택 w1={w1} w2={w2} (LB예측Δ {lbp:+.4f})")

    final = newclf(); final.fit(C.stack_input(P1, seqX), yi)
    import joblib; joblib.dump(final, OUT + r"\corrector.pkl")
    json.dump({"w1": float(w1), "w2": float(w2), "offsets": off.tolist()},
              open(OUT + r"\corrector_w.json", "w"))
    print(f"→ corrector.pkl + corrector_w.json 저장 (메타포함) ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
