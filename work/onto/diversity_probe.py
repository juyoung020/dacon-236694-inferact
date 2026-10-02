# -*- coding: utf-8 -*-
"""다양성 프로브 — 프롬프트-blind 순수 궤적모델이 챔피언과 '다른 오차'를 내나?
궤적모델 = HGB on 67차원 액션시퀀스(prev1-4·argtype·result·counts)만. 프롬프트/e5 전무.
측정: (1) 궤적 단독 성능 (2) 챔피언 오답을 궤적이 구제하는 비율 (3) 오라클-블렌드 상한
      (4) 현실 블렌드가 챔피언 이기나. 오라클≈챔피언 → 다양성 없음(천장 모델독립). 오라클>>챔피언 → 다양성 존재."""
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
ACTIONS = C.ACTIONS; A_IDX = C.A_IDX; GMASK = C.GMASK_IDX; GRPN = ["SEARCH", "MODIFY", "EXEC", "TALK"]
GROUPS = {"SEARCH": ["read_file", "grep_search", "list_directory", "glob_pattern"],
          "MODIFY": ["edit_file", "write_file", "apply_patch"], "EXEC": ["run_bash", "run_tests", "lint_or_typecheck"],
          "TALK": ["ask_user", "plan_task", "web_search", "respond_only"]}
oof = pickle.load(open(OUT + r"\oof.pkl", "rb")); ids = oof["ids"]; P = np.asarray(oof["P"], float)
y = np.array(oof["y"]); yi = np.array([A_IDX[t] for t in y])
_ff = pickle.load(open(OUT + r"\ft_full_oof.pkl", "rb")); Ff = np.array([_ff[i] for i in ids], float)
folds = json.load(open(OUT + r"\folds.json", encoding="utf-8")); fold_arr = np.array([folds[i] for i in ids])
offs = np.array(json.load(open(OUT + r"\corrector_w.json"))["offsets"])
rec = {}
for line in open(DATA, encoding="utf-8"):
    d = json.loads(line); rec[d["id"]] = d
hist = {i: C.parse_history(rec[i]["history"]) for i in ids}
SEQ = C.build_seq_feats(hist, ids)                      # 67차원, 순수 궤적 (프롬프트 없음)
metaX = np.hstack([SEQ, MF.build_meta_feats(rec, ids)])
print(f"궤적피처 {SEQ.shape} (프롬프트/e5 전무)", flush=True)

# 챔피언 OOF
P1 = 0.55 * P + 0.45 * Ff; Q = np.zeros_like(P)
for k in range(3):
    tr = fold_arr != k; va = fold_arr == k
    clf = newclf(); clf.fit(C.stack_input(P1[tr], metaX[tr]), yi[tr]); Q[va] = C.corrector_proba(clf, P1[va], metaX[va])
P2 = 0.65 * P1 + 0.35 * Q
champ_masked = C.group_mask(P, P2)
champ_pred = np.array([ACTIONS[k] for k in (np.log(champ_masked + 1e-9) + offs).argmax(1)], dtype=object)

# 궤적 단독 모델 (HGB on SEQ 67차원만) OOF
trajP = np.zeros_like(P)
for k in range(3):
    tr = fold_arr != k; va = fold_arr == k
    m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
                                       min_samples_leaf=40, class_weight="balanced", random_state=0).fit(SEQ[tr], yi[tr])
    pk = m.predict_proba(SEQ[va])
    for ci, cls in enumerate(m.classes_):
        trajP[va, int(cls)] = pk[:, ci]
# 공정 비교: 궤적모델을 챔피언과 같은 그룹으로 라우팅(group_mask) → 그룹 내부 다양성만 측정
traj_masked = C.group_mask(P, trajP)
traj_pred = np.array([ACTIONS[k] for k in traj_masked.argmax(1)], dtype=object)
traj_pred_raw = np.array([ACTIONS[k] for k in trajP.argmax(1)], dtype=object)
print("궤적모델 OOF 완료 (챔피언 그룹으로 라우팅해 그룹내부 비교)\n", flush=True)

cc = champ_pred == y; tc = traj_pred == y


def mf1(pred, mask=None):
    m = np.ones(len(y), bool) if mask is None else mask
    return f1_score(y[m], pred[m], labels=ACTIONS, average="macro", zero_division=0)


print(f"{'':8} champ_acc  traj_acc  일치%   구제(champ오답→traj정답)  오라클_acc", flush=True)
for G in ["ALL"] + GRPN:
    m = np.ones(len(y), bool) if G == "ALL" else np.isin(y, GROUPS[G])
    ca = cc[m].mean(); ta = tc[m].mean(); agree = (champ_pred[m] == traj_pred[m]).mean()
    cw = m & ~cc; rescue = (cw & tc).sum() / max(cw.sum(), 1)     # 챔피언 오답 중 궤적이 맞춘 비율
    oracle = (m & (cc | tc)).sum() / m.sum()
    print(f"  {G:6} {ca:.4f}    {ta:.4f}   {agree:.2f}    {rescue:.3f} ({(cw & tc).sum()}/{cw.sum()})       {oracle:.4f}", flush=True)

# 오라클 macroF1 상한 (완벽 셀렉터: champ 맞으면 champ, 아니면 traj)
oracle_pred = np.where(cc, champ_pred, np.where(tc, traj_pred, champ_pred))
print(f"\n오라클-블렌드 macroF1 상한 = {mf1(oracle_pred):.4f}  (champ {mf1(champ_pred):.4f}, traj {mf1(traj_pred):.4f})", flush=True)

# 현실 블렌드 (champ_prob ⊕ traj_prob, 챔피언 그룹으로 마스킹) 스윕
cp_norm = champ_masked / np.maximum(champ_masked.sum(1, keepdims=True), 1e-9)
print("\n현실 블렌드 w·champ + (1-w)·traj (챔피언 그룹 마스킹, +offs) — 챔피언 이기나?", flush=True)
best = (mf1(champ_pred), 1.0)
for w in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
    Pb = C.group_mask(P, w * cp_norm + (1 - w) * trajP)
    pred = np.array([ACTIONS[k] for k in (np.log(Pb + 1e-9) + offs).argmax(1)], dtype=object)
    f = mf1(pred); best = max(best, (f, w))
    print(f"  w_champ={w:.1f}  macroF1={f:.4f}", flush=True)
print(f"\n>>> 최고 블렌드 {best[0]:.4f} @ w={best[1]}  vs 챔피언 {mf1(champ_pred):.4f}  (Δ {best[0]-mf1(champ_pred):+.4f})", flush=True)
