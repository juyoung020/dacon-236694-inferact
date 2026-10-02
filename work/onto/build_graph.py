# -*- coding: utf-8 -*-
"""세션 타임라인 복원 → 검색용 (발화→액션) 쌍 추출.

겹치는 history 윈도우를 접미-접두 정렬로 병합해 세션 전체 사건열을 복원하고,
거기서 observed_event(발화→액션, 결과 포함) 쌍을 뽑아 검색 인덱스의 재료(events)로 쓴다.
산출: graph.pkl (steps, events), folds.json (GroupKFold 3, 세션 단위)
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
def _p(rel):  # 저장소 루트 기준 경로
    return _os.path.join(_ROOT, *[x for x in rel.split('\\') if x])
import json, re, csv, pickle, sys
from collections import defaultdict
from pathlib import Path

BASE = Path(_p(""))
DATA = BASE / "data" / "data"
OUT = BASE / "work" / "onto"
sys.path.insert(0, str(OUT))
import meta_feats as MF   # base에도 세션메타(open_files 등) 주입: step["meta_vec"] (features.py가 읽음)
import corrector as C     # L1이 쓸 순서/args 피처(67차원) 주입: step["l1_seq"] (features.py L1Group이 읽음)

GROUPS = {
    "read_file": "SEARCH", "grep_search": "SEARCH", "list_directory": "SEARCH", "glob_pattern": "SEARCH",
    "edit_file": "MODIFY", "write_file": "MODIFY", "apply_patch": "MODIFY",
    "run_bash": "EXEC", "run_tests": "EXEC", "lint_or_typecheck": "EXEC",
    "ask_user": "TALK", "plan_task": "TALK", "web_search": "TALK", "respond_only": "TALK",
}
ACTIONS = sorted(GROUPS)
GROUP_LIST = ["SEARCH", "MODIFY", "EXEC", "TALK"]

_ws = re.compile(r"\s+")

def norm(t):
    return _ws.sub(" ", t.strip())

def sess_of(rid):
    return rid.rsplit("-step_", 1)[0]

def step_of(rid):
    return int(rid.rsplit("-step_", 1)[1])

def parse_jsonl(path, labels=None):
    steps = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            m = d["session_meta"]; w = m["workspace"]
            hist = []
            for item in d["history"]:
                if item["role"] == "assistant_action":
                    hist.append(("A", item["name"], item.get("result_summary", "") or ""))
                else:
                    hist.append(("U", norm(item.get("content", "") or ""), ""))
            exts = sorted({p.rsplit(".", 1)[-1].lower() for p in w["open_files"] if "." in p})
            of_names = " ".join(p.split("/")[-1] for p in w["open_files"])[:100]   # ft용: 열린 파일명 텍스트
            _argtok = []                                                            # ft용: 히스토리 args 값 텍스트
            for it in d["history"]:
                if it.get("role") == "assistant_action":
                    a = it.get("args") or {}
                    if "cmd" in a: _argtok.append(str(a["cmd"]).split()[0].split("/")[-1][:20] if str(a["cmd"]).strip() else "")
                    elif "pattern" in a: _argtok.append(str(a["pattern"])[:20])
                    elif "scope" in a: _argtok.append(str(a["scope"])[:24])
                    elif "path" in a: _argtok.append(str(a["path"]).split("/")[-1][:24])
                    elif "target" in a: _argtok.append(str(a["target"])[:24])
            of_args = " ".join(t for t in _argtok[-10:] if t)[:140]
            steps.append({
                "id": d["id"], "sess": sess_of(d["id"]), "step": step_of(d["id"]),
                "prompt": norm(d["current_prompt"]), "hist": hist,
                "tier": m["user_tier"], "lang": m["language_pref"],
                "turn": m["turn_index"], "elapsed": m["elapsed_session_sec"],
                "budget": m["budget_tokens_remaining"],
                "loc": w["loc"], "dirty": bool(w["git_dirty"]), "ci": w["last_ci_status"],
                "open_n": len(w["open_files"]), "open_ext": exts,
                "mix": max(w["language_mix"], key=w["language_mix"].get) if w["language_mix"] else "",
                "meta_vec": MF.feats_one(d).tolist(),   # 미개척 세션메타 50차원 (base용)
                "l1_seq": C.feats_one(*C.parse_history(d["history"])).tolist(),   # L1용 순서/args 67차원 (하드라우팅 L1)
                "of_names": of_names, "of_args": of_args,   # ft용 텍스트(열린파일명·args)
                "au": 1 if d["id"].startswith("sess_au") else 0,
                "label": labels.get(d["id"]) if labels else None,
            })
    return steps

def _merge_window(T, W):
    """윈도우 W를 타임라인 T 꼬리에 정렬 병합. 동일성=(type, content), result는 보강.
       반환 겹침 길이 k (0=간극)."""
    max_k = min(len(T), len(W))
    for k in range(max_k, 0, -1):
        off = len(T) - k
        if all(T[off + i][0] == W[i][0] and T[off + i][1] == W[i][1] for i in range(k)):
            for i in range(k):
                if W[i][2] and not T[off + i][2]:
                    T[off + i][2] = W[i][2]
                T[off + i][3] = True
            for t, c, r in W[k:]:
                T.append([t, c, r, True])
            return k
    for t, c, r in W:
        T.append([t, c, r, True])
    return 0

def build_timelines(steps):
    """스텝들(세션 혼합 가능) → timelines, positions, events, stats
       events[sess] = [(발화, 액션, 결과, confirmed)] (타임라인 순서 보존)"""
    by_sess = defaultdict(list)
    for i, s in enumerate(steps):
        by_sess[s["sess"]].append(s)
    timelines, positions = {}, {}
    n_win = n_overlap = n_gap = 0
    for sess, ss in by_sess.items():
        ss.sort(key=lambda s: s["step"])
        T = []
        for s in ss:
            W = [[t, c, r] for t, c, r in s["hist"]]
            if W:
                n_win += 1
                k = _merge_window(T, W)
                if k > 0:
                    n_overlap += 1
                elif len(T) > len(W):
                    n_gap += 1
            T.append(["U", s["prompt"], "", True])
            positions[s["id"]] = len(T) - 1
            if s["label"] is not None:
                T.append(["A", s["label"], "", False])       # 자가 삽입(미확증)
        timelines[sess] = T
    events = {}
    for sess, T in timelines.items():
        ev = []
        for i in range(len(T) - 1):
            if T[i][0] == "U" and T[i + 1][0] == "A":
                ev.append((T[i][1], T[i + 1][1], T[i + 1][2], T[i + 1][3]))
        events[sess] = ev
    stats = {"windows": n_win, "overlap": n_overlap, "gap": n_gap}
    return timelines, positions, events, stats

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    labels = {}
    with open(DATA / "train_labels.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["id"]] = row["action"]
    steps = parse_jsonl(DATA / "train.jsonl", labels)

    _, _, events, st = build_timelines(steps)    # 윈도우 병합으로 세션 사건열 복원 → observed_event 쌍(events)
    print(f"윈도우 병합: {st['windows']}개 중 겹침 정렬 {st['overlap']} ({st['overlap']/st['windows']:.1%})")

    with open(OUT / "graph.pkl", "wb") as f:
        pickle.dump({"steps": steps, "events": events}, f, protocol=4)

    from sklearn.model_selection import GroupKFold
    groups = [s["sess"] for s in steps]
    folds = {}
    for k, (_, va) in enumerate(GroupKFold(n_splits=3).split(range(len(steps)), groups=groups)):
        for i in va:
            folds[steps[i]["id"]] = k
    with open(OUT / "folds.json", "w", encoding="utf-8") as f:
        json.dump(folds, f)
    print(f"folds(3): {[sum(1 for v in folds.values() if v==k) for k in range(3)]}")

if __name__ == "__main__":
    main()
