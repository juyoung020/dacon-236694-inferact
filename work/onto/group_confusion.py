# -*- coding: utf-8 -*-
"""MODIFY/EXEC/TALK confusion 심층 분석 (사용자 스펙 1·2·3·5 동시).
1) 그룹별 within-group confusion matrix + 비중/클래스수/acc/macroF1 (1차 스크리닝)
2) regime 분해: prev2/prev1/n_act/open_files별 서브셋 정확도 → 집중형 vs 분산형
3) base vs +e5 vs +e5+corrector 그룹별 기여도 (net-positive가 그룹마다 다른가)
5) offsets 그룹별 기여 (offs on/off) — 특정 그룹만 크게 작용하면 과적합 의심
+ 오분류 top pair + 예시 id (step4용). 전부 onto champion OOF 기반."""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
def _p(rel):  # 저장소 루트 기준 경로
    return _os.path.join(_ROOT, *[x for x in rel.split('\\') if x])
import json, pickle, sys
import numpy as np
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8")
OUT = _p(r"work\onto")
DATA = _p(r"data\data\train.jsonl")
sys.path.insert(0, OUT)
import corrector as C
import meta_feats as MF
from train_corrector import newclf
from sklearn.metrics import f1_score, accuracy_score

ACTIONS = C.ACTIONS; A_IDX = C.A_IDX
GROUPS = {"MODIFY": ["edit_file", "write_file", "apply_patch"],
          "EXEC": ["run_bash", "run_tests", "lint_or_typecheck"],
          "TALK": ["ask_user", "plan_task", "web_search", "respond_only"],
          "SEARCH": ["read_file", "grep_search", "list_directory", "glob_pattern"]}
GMASK = C.GMASK_IDX; GRPN = ["SEARCH", "MODIFY", "EXEC", "TALK"]

oof = pickle.load(open(OUT + r"\oof.pkl", "rb"))
ids = oof["ids"]; P = np.asarray(oof["P"], float); y = np.array(oof["y"]); yi = np.array([A_IDX[t] for t in y])
ff = pickle.load(open(OUT + r"\ft_full_oof.pkl", "rb")); Ff = np.array([ff[i] for i in ids], float)
folds = json.load(open(OUT + r"\folds.json", encoding="utf-8")); fold_arr = np.array([folds[i] for i in ids])
offs = np.array(json.load(open(OUT + r"\corrector_w.json"))["offsets"])
rec = {}
for line in open(DATA, encoding="utf-8"):
    d = json.loads(line); rec[d["id"]] = d
hist = {i: C.parse_history(rec[i]["history"]) for i in ids}
seqX = np.hstack([C.build_seq_feats(hist, ids), MF.build_meta_feats(rec, ids)])
P1 = 0.55 * P + 0.45 * Ff
Q = np.zeros_like(P)
for k in range(3):
    tr = fold_arr != k; va = fold_arr == k
    clf = newclf(); clf.fit(C.stack_input(P1[tr], seqX[tr]), yi[tr]); Q[va] = C.corrector_proba(clf, P1[va], seqX[va])
P2 = 0.65 * P1 + 0.35 * Q
print("OOF 준비완료\n", flush=True)

prev1 = np.array([hist[i][0][-1] if len(hist[i][0]) >= 1 else "<none>" for i in ids])
prev2 = np.array([hist[i][0][-2] if len(hist[i][0]) >= 2 else "<none>" for i in ids])
nact = np.array([len(hist[i][0]) for i in ids])
nopen = np.array([len(rec[i].get("session_meta", {}).get("workspace", {}).get("open_files", []) or []) for i in ids])


def final(Pb, use_off=True):
    logP = np.log(C.group_mask(P, Pb) + 1e-9) + (offs if use_off else 0.0)
    return np.array([ACTIONS[k] for k in logP.argmax(1)], dtype=object)


pred = {"base": final(P), "+e5": final(P1), "+e5+corr": final(P2), "champ_noff": final(P2, False)}
N = len(y)

for G, mem in [("MODIFY", GROUPS["MODIFY"]), ("EXEC", GROUPS["EXEC"]), ("TALK", GROUPS["TALK"])]:
    tG = np.isin(y, mem)
    print(f"\n{'='*70}\n[{G}]  클래스 {len(mem)}개, 비중 {tG.sum()}/{N}={tG.sum()/N:.1%}", flush=True)
    # step3 + step5: 기여도
    print("  ── base/e5/corrector/offset 기여 (true-G rows) ──", flush=True)
    for nm in ["base", "+e5", "+e5+corr", "champ_noff"]:
        p = pred[nm]
        acc = accuracy_score(y[tG], p[tG]); mf1 = f1_score(y[tG], p[tG], labels=mem, average="macro", zero_division=0)
        print(f"    {nm:11} acc {acc:.4f}  macroF1 {mf1:.4f}", flush=True)
    # step1: confusion matrix (within-group + escaped)
    pc = pred["+e5+corr"]
    print("  ── confusion (행=정답, 열=예측; ESC=그룹밖) ──", flush=True)
    hdr = mem + ["ESC"]
    print("    " + "true\\pred".ljust(16) + "".join(a[:9].rjust(10) for a in hdr), flush=True)
    for tc in mem:
        row = y == tc
        cnt = Counter(p if p in mem else "ESC" for p in pc[row])
        cells = "".join(str(cnt.get(a, 0)).rjust(10) for a in hdr)
        rc = f1_score(y[row], pc[row], labels=[tc], average="macro", zero_division=0)
        print(f"    {tc.ljust(16)}{cells}   (n={row.sum()}, F1={rc:.3f})", flush=True)
    # step2: regime 분해 — prev2/prev1/nact/nopen별 정확도
    print("  ── regime 스캔 (집중형이면 특정 조건서만 고정확) ──", flush=True)
    for axis, arr in [("prev2", prev2), ("prev1", prev1)]:
        states = [s for s, _ in Counter(arr[tG]).most_common(6)]
        segs = []
        for s in states:
            m = tG & (arr == s)
            if m.sum() >= 50:
                maj = Counter(y[m]).most_common(1)[0]
                segs.append(f"{s[:10]}={accuracy_score(y[m], pc[m]):.2f}(n{m.sum()},{maj[0][:4]}{maj[1]/m.sum():.0%})")
        print(f"    [{axis}] " + "  ".join(segs), flush=True)
    for axis, arr, bins in [("nact", nact, [(0, 0), (1, 1), (2, 3), (4, 6), (7, 99)]),
                            ("nopen", nopen, [(0, 0), (1, 1), (2, 9)])]:
        segs = []
        for lo, hi in bins:
            m = tG & (arr >= lo) & (arr <= hi)
            if m.sum() >= 50:
                segs.append(f"{lo}-{hi}={accuracy_score(y[m], pc[m]):.2f}(n{m.sum()})")
        print(f"    [{axis}] " + "  ".join(segs), flush=True)
    # step4 준비: top 오분류 쌍 + 예시 id
    pairs = Counter()
    for i in np.where(tG)[0]:
        pp = pc[i] if pc[i] in mem else "ESC"
        if pp != y[i]:
            pairs[(y[i], pp)] += 1
    print("  ── top 오분류 쌍 (정답→예측) + 예시 id ──", flush=True)
    for (tc, pp), c in pairs.most_common(5):
        exs = [ids[i] for i in np.where(tG & (y == tc))[0] if (pc[i] if pc[i] in mem else "ESC") == pp][:2]
        print(f"    {tc}→{pp}: {c}회   예: {exs}", flush=True)

# 참고: SEARCH 재출력(비교 기준)
tG = np.isin(y, GROUPS["SEARCH"]); pc = pred["+e5+corr"]
print(f"\n[참고 SEARCH] 비중 {tG.sum()/N:.1%}  acc {accuracy_score(y[tG], pc[tG]):.4f}  "
      f"macroF1 {f1_score(y[tG], pc[tG], labels=GROUPS['SEARCH'], average='macro', zero_division=0):.4f}", flush=True)
