# -*- coding: utf-8 -*-
"""au 파인튜닝 블렌드 확정: oof.pkl(기본 확률) + ft_oof.pkl(au e5 파인튜닝 확률)에서
블렌드 강도 w와 블렌드 후 오프셋을 튜닝 → ft_w.json. 추론(pack.py)이 이걸 읽는다."""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
def _p(rel):  # 저장소 루트 기준 경로
    return _os.path.join(_ROOT, *[x for x in rel.split('\\') if x])
import json, pickle, sys
import numpy as np
sys.path.insert(0, _p(r"work\onto"))
from build_graph import ACTIONS
from train_hier import tune_offsets, final_preds
from sklearn.metrics import f1_score

OUT = _p(r"work\onto")

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    d = pickle.load(open(OUT + r"\oof.pkl", "rb"))
    ids, rl, P, y = d["ids"], d["rl"], np.asarray(d["P"]), np.array(d["y"])
    ft = pickle.load(open(OUT + r"\ft_oof.pkl", "rb"))
    au = np.array([i.startswith("sess_au") for i in ids])

    FT = np.zeros_like(P); has = np.zeros(len(ids), bool)
    for i, id_ in enumerate(ids):
        if id_ in ft:
            FT[i] = ft[id_]; has[i] = True

    best = None
    for w in (0.45, 0.5, 0.6, 0.7):
        Q = P.copy(); Q[has] = (1 - w) * P[has] + w * FT[has]
        off, B = tune_offsets(rl, Q, y)
        if best is None or B > best[1]:
            best = (w, B, off, Q)
    w, B, off, Q = best
    pred = np.array(final_preds(rl, Q, off)); yA = np.array(y)
    print(f"ft 블렌드 확정: w={w}, Macro-F1={B:.4f} "
          f"(au {f1_score(yA[au], pred[au], average='macro'):.4f} / sim {f1_score(yA[~au], pred[~au], average='macro'):.4f})")
    json.dump({"w": w, "offsets": off.tolist()}, open(OUT + r"\ft_w.json", "w"))

if __name__ == "__main__":
    main()
