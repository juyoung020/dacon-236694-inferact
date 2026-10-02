# -*- coding: utf-8 -*-
"""[코렉터] base(+풀ft) 확률 위에 '과거 액션 args + 순서'를 얹어 탐색 혼동을 교정.
학습(train_corrector.py)과 추론(pack.py의 script)이 공유. build_graph는 안 건드리고
raw jsonl history에서 직접 (액션명, arg타입, 결과범주)를 뽑는다.
"""
import numpy as np
from collections import Counter

ACTIONS = sorted(["read_file","grep_search","list_directory","glob_pattern","edit_file","write_file",
    "apply_patch","run_bash","run_tests","lint_or_typecheck","ask_user","plan_task","web_search","respond_only"])
A_IDX = {a: i for i, a in enumerate(ACTIONS)}
ARGT = ["rf_ext","rf_noext","grep","glob","list","edit","write","patch","bash","tests","lint","ask","plan","web","resp","other","NONE"]
AT_IDX = {a: i for i, a in enumerate(ARGT)}

# 하드라우팅 그룹마스크 (build_graph GROUPS와 동일 — 순환 import 방지 위해 로컬 정의)
GROUP_OF = {"read_file":"SEARCH","grep_search":"SEARCH","list_directory":"SEARCH","glob_pattern":"SEARCH",
            "edit_file":"MODIFY","write_file":"MODIFY","apply_patch":"MODIFY",
            "run_bash":"EXEC","run_tests":"EXEC","lint_or_typecheck":"EXEC",
            "ask_user":"TALK","plan_task":"TALK","web_search":"TALK","respond_only":"TALK"}
GMASK_IDX = np.array([["SEARCH","MODIFY","EXEC","TALK"].index(GROUP_OF[a]) for a in ACTIONS])

def group_mask(P_base, P_blend):
    """P_base(하드라우팅 base, non-g*=0)의 g*(=argmax 그룹)로 P_blend의 타그룹을 0으로.
    ft블렌드/코렉터가 g* 밖으로 확률을 새게 해도 최종 argmax를 g*에 고정."""
    gstar = GMASK_IDX[np.asarray(P_base).argmax(1)]
    return np.where(GMASK_IDX[None, :] == gstar[:, None], np.asarray(P_blend), 0.0)

def arg_type(name, args):
    args = args or {}
    if name == "read_file":
        p = str(args.get("path", "")); base = p.rsplit("/", 1)[-1]
        return "rf_ext" if "." in base else "rf_noext"
    if name == "grep_search": return "grep"
    if name == "glob_pattern": return "glob"
    if name == "list_directory": return "list"
    return {"edit_file":"edit","write_file":"write","apply_patch":"patch","run_bash":"bash","run_tests":"tests",
            "lint_or_typecheck":"lint","ask_user":"ask","plan_task":"plan","web_search":"web","respond_only":"resp"}.get(name, "other")

def result_cat(r):
    r = (r or "").lower()
    if "0 match" in r or "no match" in r or "0 occur" in r: return 0
    if "match" in r or "occur" in r or "found" in r: return 1
    if r.startswith(("ok","read","listed","defines")) or "defines" in r or "lines" in r: return 2
    if "error" in r or "fail" in r: return 3
    return 4

def parse_history(history):
    """raw jsonl의 d['history'] → (names, argts, rcats) — assistant_action만, 순서 보존."""
    names, argts, rcats = [], [], []
    for it in history:
        if it.get("role") == "assistant_action":
            nm = it["name"]
            names.append(nm)
            argts.append(arg_type(nm, it.get("args")))
            rcats.append(result_cat(it.get("result_summary", "")))
    return names, argts, rcats

def _oh(idx, n):
    v = np.zeros(n)
    if idx is not None: v[idx] = 1
    return v

def feats_one(names, argts, rcats):
    """한 스텝의 순서/args 피처 (67차원). train_corrector와 추론이 반드시 동일하게 사용."""
    def ai(k): return A_IDX.get(names[-k]) if len(names) >= k else None
    cnt = Counter(names)
    cv = np.array([cnt.get(a, 0) for a in ACTIONS], dtype=float)
    la = AT_IDX.get(argts[-1], AT_IDX["NONE"]) if argts else AT_IDX["NONE"]
    la2 = AT_IDX.get(argts[-2], AT_IDX["NONE"]) if len(argts) >= 2 else AT_IDX["NONE"]
    lr = rcats[-1] if rcats else 4
    return np.concatenate([_oh(ai(1), 14), _oh(ai(2), 14), _oh(ai(3), 14), _oh(ai(4), 14),
                           cv, [len(names)], _oh(la, 17), _oh(la2, 17), _oh(lr, 5)])

def build_seq_feats(hist_by_id, ids):
    """ids 순서대로 순서/args 피처 행렬 (n,67). hist_by_id[id] = (names,argts,rcats)."""
    return np.vstack([feats_one(*hist_by_id[i]) for i in ids])

def stack_input(P1, seq_feats, eps=1e-9):
    """코렉터 입력 = [log(base+ftft 확률) | 순서/args 피처]."""
    return np.hstack([np.log(P1 + eps), seq_feats])

def corrector_proba(clf, P1, seq_feats):
    """학습된 HGB → ACTIONS 순서의 (n,14) 확률."""
    Pk = clf.predict_proba(stack_input(P1, seq_feats))
    Q = np.zeros((len(P1), len(ACTIONS)))
    for ci, cls in enumerate(clf.classes_):
        Q[:, int(cls)] = Pk[:, ci]
    return Q
