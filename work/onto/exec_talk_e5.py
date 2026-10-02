# -*- coding: utf-8 -*-
"""EXEC/TALK regime-aware e5 down 테스트. 스크리닝서 EXEC는 +e5가 base보다 낮았음(e5=디코이 의심).
그룹별 e5 가중 w1을 낮춰 그 그룹 macroF1 + 전체 macroF1이 오르나. 코렉터 Q는 챔피언(고정)."""
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
from sklearn.metrics import f1_score, accuracy_score
ACTIONS = C.ACTIONS; A_IDX = C.A_IDX; GMASK = C.GMASK_IDX; GRPN = ["SEARCH", "MODIFY", "EXEC", "TALK"]
MEM = {"EXEC": ["run_bash", "run_tests", "lint_or_typecheck"],
       "TALK": ["ask_user", "plan_task", "web_search", "respond_only"],
       "MODIFY": ["edit_file", "write_file", "apply_patch"], "SEARCH": ["read_file", "grep_search", "list_directory", "glob_pattern"]}
oof = pickle.load(open(OUT + r"\oof.pkl", "rb")); ids = oof["ids"]; P = np.asarray(oof["P"], float)
y = np.array(oof["y"]); yi = np.array([A_IDX[t] for t in y])
_ff = pickle.load(open(OUT + r"\ft_full_oof.pkl", "rb")); Ff = np.array([_ff[i] for i in ids], float)
folds = json.load(open(OUT + r"\folds.json", encoding="utf-8")); fold_arr = np.array([folds[i] for i in ids])
offs = np.array(json.load(open(OUT + r"\corrector_w.json"))["offsets"])
rec = {}
for line in open(DATA, encoding="utf-8"):
    d = json.loads(line); rec[d["id"]] = d
hist = {i: C.parse_history(rec[i]["history"]) for i in ids}
seqX = np.hstack([C.build_seq_feats(hist, ids), MF.build_meta_feats(rec, ids)])
P1b = 0.55 * P + 0.45 * Ff; Q = np.zeros_like(P)
for k in range(3):
    tr = fold_arr != k; va = fold_arr == k
    clf = newclf(); clf.fit(C.stack_input(P1b[tr], seqX[tr]), yi[tr]); Q[va] = C.corrector_proba(clf, P1b[va], seqX[va])
gstar = np.array([GRPN[GMASK[P[i].argmax()]] for i in range(len(P))])
print("준비완료\n", flush=True)


def build(w1g, group, w2g=0.35):
    w1v = np.full(len(P), 0.45); w2v = np.full(len(P), 0.35)
    sel = gstar == group; w1v[sel] = w1g; w2v[sel] = w2g
    P1 = (1 - w1v)[:, None] * P + w1v[:, None] * Ff
    P2 = (1 - w2v)[:, None] * P1 + w2v[:, None] * Q
    logP = np.log(C.group_mask(P, P2) + 1e-9) + offs
    return np.array([ACTIONS[k] for k in logP.argmax(1)], dtype=object)


def rep(pred, group, tag):
    tG = np.isin(y, MEM[group])
    gf1 = f1_score(y[tG], pred[tG], labels=MEM[group], average="macro", zero_division=0)
    full = f1_score(y, pred, labels=ACTIONS, average="macro", zero_division=0)
    print(f"  {tag:22} {group}-macroF1 {gf1:.4f}   전체F1 {full:.4f}", flush=True)
    return full


for G in ["EXEC", "TALK"]:
    print(f"=== {G} e5 가중 w1 스윕 (w2=.35 고정) ===", flush=True)
    for w1g in [0.45, 0.3, 0.2, 0.1, 0.0]:
        rep(build(w1g, G), G, f"w1={w1g}")
    print(f"=== {G} e5=0 + corrector w2 스윕 ===", flush=True)
    for w2g in [0.35, 0.5, 0.2]:
        rep(build(0.0, G, w2g), G, f"w1=0 w2={w2g}")
    print()
