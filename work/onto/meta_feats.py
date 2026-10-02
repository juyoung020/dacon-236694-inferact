# -*- coding: utf-8 -*-
"""[미개척 메타신호 — comprehensive] session_meta의 구조적 신호 총동원:
open_files 확장자/종류개수/경로구조 + 프롬프트·히스토리 교차 + language_mix 전체.
챔피언(features.py)이 open_n만 쓰고 버린 신호. 코렉터 seqX에 스택 → held-out LB예측 +0.0055(au 유지·sim↑).
lexical(키워드/심볼)은 base TF-IDF와 겹쳐 과적합하므로 제외. 학습(train_corrector_meta.py)·추론(pack.py) 공유."""
import re
import numpy as np

LM_ALL = ["py","js","ts","tsx","java","go","rs","md","json","yaml","toml","html","css","sh","dockerfile"]
CFG = {"toml","yaml","yml","json","ini","cfg","conf","lock"}
SRC = {"py","js","ts","tsx","java","go","rs"}
_FT = re.compile(r"[\w./-]+\.\w{1,5}")   # 프롬프트 내 file.ext 토큰
META_DIM = 15 + 7 + 5 + 6 + 15 + 2       # 50

def feats_one(d):
    """원본 record(session_meta·current_prompt·history) → 50차원 구조적 메타피처."""
    m = d.get("session_meta", {}) or {}; prompt = d.get("current_prompt", "") or ""; pl = prompt.lower()
    w = m.get("workspace", {}) or {}; ofs = w.get("open_files", []) or []; lm = w.get("language_mix", {}) or {}
    # 열린파일 확장자 멀티핫
    exts = set(p.rsplit(".", 1)[-1].lower() for p in ofs if "." in p)
    ext_mh = [1.0 if e in exts else 0.0 for e in LM_ALL]                              # 15
    # 파일 종류 개수/플래그
    n_test = sum(1 for p in ofs if "test" in p.lower() or "spec" in p.lower())
    n_cfg = sum(1 for p in ofs if p.rsplit(".", 1)[-1].lower() in CFG)
    n_src = sum(1 for p in ofs if p.rsplit(".", 1)[-1].lower() in SRC)
    n_doc = sum(1 for p in ofs if p.lower().endswith((".md", ".rst", ".txt")))
    flags = [float(n_test > 0), float(n_cfg > 0), float(n_doc > 0),
             float(n_test), float(n_cfg), float(n_src), float(n_doc)]                 # 7
    # 경로 구조
    dirs = [p.rsplit("/", 1)[0] if "/" in p else "" for p in ofs]
    depths = [p.count("/") for p in ofs]
    pathfeat = [float(len(ofs)), float(len(set(dirs))), float(len(set(dirs)) <= 1 and len(ofs) > 1),
                float(max(depths) if depths else 0), float(np.mean(depths) if depths else 0)]  # 5
    # 프롬프트 × open_files / history 교차
    basenames = [p.split("/")[-1].lower() for p in ofs]
    stems = [b.rsplit(".", 1)[0] for b in basenames if len(b.rsplit(".", 1)[0]) > 3]
    ment_open = float(any(b in pl for b in basenames if len(b) > 4) or any(st in pl for st in stems))
    ptoks = [t.lower() for t in _FT.findall(prompt)]
    has_file = float(len(ptoks) > 0)
    ment_isopen = float(any(t.split("/")[-1] in basenames for t in ptoks))
    ment_notopen = float(has_file and not ment_isopen)
    histpaths = set()
    for it in d.get("history", []):
        if it.get("role") == "assistant_action":
            a = it.get("args") or {}; pth = a.get("path") or a.get("scope") or ""
            if pth: histpaths.add(pth.split("/")[-1].lower())
    ment_in_hist = float(any(t.split("/")[-1] in histpaths for t in ptoks))
    n_touched = float(sum(1 for b in basenames if b in histpaths))
    interact = [ment_open, has_file, ment_isopen, ment_notopen, ment_in_hist, n_touched]  # 6
    # language_mix 전체 + 통계
    lm_full = [float(lm.get(e, 0.0)) for e in LM_ALL]                                 # 15
    lm_stats = [float(len([v for v in lm.values() if v > 0.01])),
                float(max(lm.values()) if lm else 0.0)]                               # 2
    return np.array(ext_mh + flags + pathfeat + interact + lm_full + lm_stats)

def build_meta_feats(rec_by_id, ids):
    """ids 순서대로 (n, 50). rec_by_id[id] = 원본 record."""
    return np.vstack([feats_one(rec_by_id[i]) for i in ids])
