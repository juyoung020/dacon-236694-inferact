# -*- coding: utf-8 -*-
"""SEARCH regime-aware 블렌드 스윕 (사용자 스펙 4단계).
가설: 프롬프트=디코이(Detective B)인데 e5(내용기반)가 SEARCH 확률적74%에 w1=0.45로 노이즈 주입.
1) prev2 규칙으로 SEARCH를 cycle(26%)/stochastic(74%) 태깅 (test-time 가능, leak 없음)
2) stochastic 구간 e5 w1 스윕 {0,.1,.2,.3,.45} + prior 비교(핵심: e5=0가 최빈/prev2-prior 이기나?)
3) corrector w2도 regime별 스윕
4) SEARCH-group-only acc/macroF1로 먼저 검증 (전체 희석 방지). 챔피언 offsets 고정(효과 격리).
전부 onto 실제 OOF(oof.pkl champion + ft_full_oof + 코렉터 3fold 재현) 기반."""
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

ACTIONS = C.ACTIONS; A_IDX = C.A_IDX
SEARCH4 = ["read_file", "grep_search", "list_directory", "glob_pattern"]
GMASK = C.GMASK_IDX
GRPN = ["SEARCH", "MODIFY", "EXEC", "TALK"]

oof = pickle.load(open(OUT + r"\oof.pkl", "rb"))
ids = oof["ids"]; P = np.asarray(oof["P"], float); y = np.array(oof["y"]); yi = np.array([A_IDX[t] for t in y])
ff = pickle.load(open(OUT + r"\ft_full_oof.pkl", "rb")); Ff = np.array([ff[i] for i in ids], float)
folds = json.load(open(OUT + r"\folds.json", encoding="utf-8")); fold_arr = np.array([folds[i] for i in ids])
offs = np.array(json.load(open(OUT + r"\corrector_w.json"))["offsets"])
au = np.array([i.startswith("sess_au") for i in ids])
rec = {}
for line in open(DATA, encoding="utf-8"):
    d = json.loads(line); rec[d["id"]] = d
hist = {i: C.parse_history(rec[i]["history"]) for i in ids}
seqX = np.hstack([C.build_seq_feats(hist, ids), MF.build_meta_feats(rec, ids)])

# 코렉터 Q OOF (w1=0.45 base)
P1b = 0.55 * P + 0.45 * Ff
Q = np.zeros_like(P)
for k in range(3):
    tr = fold_arr != k; va = fold_arr == k
    clf = newclf(); clf.fit(C.stack_input(P1b[tr], seqX[tr]), yi[tr]); Q[va] = C.corrector_proba(clf, P1b[va], seqX[va])
print("코렉터 OOF 완료", flush=True)

gstar = np.array([GRPN[GMASK[P[i].argmax()]] for i in range(len(P))])
prev2 = np.array([hist[i][0][-2] if len(hist[i][0]) >= 2 else "<none>" for i in ids])
CYCLE_PREV2 = {"read_file", "grep_search", "list_directory", "edit_file", "apply_patch", "run_tests", "run_bash"}
is_cyc = np.array([p in CYCLE_PREV2 for p in prev2])
sr = gstar == "SEARCH"; tS = np.isin(y, SEARCH4)
print(f"SEARCH-routed {sr.sum()}  true-SEARCH {tS.sum()}", flush=True)
print(f"  routed&cycle {(sr & is_cyc).sum()}  routed&stoch {(sr & ~is_cyc).sum()} "
      f"({(sr & ~is_cyc).sum() / sr.sum():.0%} stoch)  ← 26/74 확인", flush=True)


def build_pred(w1s, w2s, w1c=0.45, w2c=0.35, prior=None):
    """prior=None: 블렌드. prior='grep'/'prev2': stochastic-routed 예측을 prior로 덮어씀."""
    w1v = np.full(len(P), 0.45); w2v = np.full(len(P), 0.35)
    w1v[sr & is_cyc] = w1c; w2v[sr & is_cyc] = w2c
    w1v[sr & ~is_cyc] = w1s; w2v[sr & ~is_cyc] = w2s
    P1 = (1 - w1v)[:, None] * P + w1v[:, None] * Ff
    P2 = (1 - w2v)[:, None] * P1 + w2v[:, None] * Q
    logP = np.log(C.group_mask(P, P2) + 1e-9) + offs
    pred = np.array([ACTIONS[k] for k in logP.argmax(1)], dtype=object)
    if prior is not None:
        st = sr & ~is_cyc
        if prior == "grep":
            pred[st] = "grep_search"
        elif prior == "prev2":                       # prev2-조건부 최빈 (OOF)
            for k in range(3):
                trm = (fold_arr != k) & tS & sr & ~is_cyc; vam = (fold_arr == k) & st
                maj = {}
                for pv in set(prev2[trm]):
                    yy = y[trm & (prev2 == pv)]
                    maj[pv] = max(set(yy), key=list(yy).count) if len(yy) else "grep_search"
                for j in np.where(vam)[0]:
                    pred[j] = maj.get(prev2[j], "grep_search")
    return pred


def rep(pred, tag):
    m = tS; ms = tS & sr & ~is_cyc; mc = tS & sr & is_cyc
    acc = accuracy_score(y[m], pred[m]); mf1 = f1_score(y[m], pred[m], labels=SEARCH4, average="macro", zero_division=0)
    full = f1_score(y, pred, labels=ACTIONS, average="macro", zero_division=0)
    accS = accuracy_score(y[ms], pred[ms]); accC = accuracy_score(y[mc], pred[mc])
    print(f"  {tag:26} SEARCH acc {acc:.4f} F1 {mf1:.4f} | cyc-acc {accC:.4f} stoch-acc {accS:.4f} | 전체F1 {full:.4f}", flush=True)
    return full, mf1


print("\n=== [baseline] 챔피언 전역 w1=.45 w2=.35 ===", flush=True)
rep(build_pred(0.45, 0.35), "champion")
print("\n=== [step2] stochastic e5 w1 스윕 (w2_sto=.35) ===", flush=True)
for w1s in [0.45, 0.3, 0.2, 0.1, 0.0]:
    rep(build_pred(w1s, 0.35), f"stoch w1={w1s}")
print("\n=== [step2 핵심] stochastic prior 비교 (e5=0 블렌드가 prior 이기나?) ===", flush=True)
rep(build_pred(0.0, 0.0), "stoch w1=0 w2=0(base만)")
rep(build_pred(0.0, 0.0, prior="grep"), "stoch=최빈(grep)")
rep(build_pred(0.0, 0.0, prior="prev2"), "stoch=prev2조건부최빈")
print("\n=== [step3] stochastic corrector w2 스윕 (w1_sto=0) ===", flush=True)
for w2s in [0.35, 0.15, 0.0]:
    rep(build_pred(0.0, w2s), f"stoch w1=0 w2={w2s}")
print("\n=== 조합: stoch(w1=0,w2=0) + cyc 유지 vs 챔피언 전체F1 ===", flush=True)
b = rep(build_pred(0.45, 0.35), "champion(전역)")
for w1s, w2s in [(0.0, 0.0), (0.1, 0.15), (0.2, 0.35)]:
    rep(build_pred(w1s, w2s), f"regime stoch({w1s},{w2s})")
