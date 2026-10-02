# -*- coding: utf-8 -*-
"""제출 패키징 + 모의채점.

submit.zip 구조: model/(bundle.joblib + 파이프라인 모듈들) + script.py + requirements.txt(빈 파일)
모의채점: train 앞 30,000행을 가짜 test.jsonl로 사용해 서버 규모의 실행시간·출력형식 검증.
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
def _p(rel):  # 저장소 루트 기준 경로
    return _os.path.join(_ROOT, *[x for x in rel.split('\\') if x])
import shutil, subprocess, sys, time, zipfile, json, csv
from pathlib import Path

OUT = Path(_p(r"work\onto"))
DATA = Path(_p(r"data\data"))
PY = Path(_p(r"work\venv311\Scripts\python.exe"))
MODULES = ["build_graph.py", "retrieval.py", "features.py", "rules.py", "train_hier.py", "corrector.py", "meta_feats.py"]

SCRIPT = '''# -*- coding: utf-8 -*-
import sys, os, json
sys.path.insert(0, "./model")
import numpy as np, joblib
# 주의: torch는 반드시 pandas보다 먼저 import (pandas 2.0.3의 DLL이 torch 초기화와 충돌 — Windows 실측)
_HAS_E5 = os.path.isdir("./model/e5")
_HAS_FT = os.path.isdir("./model/e5_ft")
_HAS_FULL = os.path.isdir("./model/e5_full")
_HAS_CORR = os.path.isfile("./model/corrector.pkl")
if _HAS_E5 or _HAS_FT or _HAS_FULL:
    import torch
    from transformers import AutoTokenizer, AutoModel
import pandas as pd
from build_graph import parse_jsonl, ACTIONS
from rules import apply_rules
from train_hier import predict_probs, final_preds

bundle = joblib.load("./model/bundle.joblib")
steps = parse_jsonl("./data/test.jsonl")

# 임베딩 사용 번들이면: 동봉된 얼린 e5로 테스트 문장 인코딩 (T4 GPU, 파인튜닝 아님)
fb = bundle["fb"]
if getattr(fb, "emb", None) is not None and _HAS_E5:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("./model/e5")
    enc_model = AutoModel.from_pretrained("./model/e5", torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev)
    enc_model.eval()
    texts = sorted({s["prompt"] for s in steps} - set(fb.emb))
    with torch.no_grad():
        for lo in range(0, len(texts), 256):
            chunk = ["query: " + t for t in texts[lo:lo + 256]]
            e = tok(chunk, padding=True, truncation=True, max_length=192, return_tensors="pt").to(dev)
            h = enc_model(**e).last_hidden_state
            m = e["attention_mask"].unsqueeze(-1)
            v = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=1).float().cpu().numpy().astype(np.float16)
            for t, vec in zip(texts[lo:lo + 256], v):
                fb.emb[t] = vec

rl = [apply_rules(s["prompt"], bundle["rules"]) for s in steps]   # [1] 어휘 룰
model = {"fb": bundle["fb"], "l1": bundle["l1"], "l2": bundle["l2"]}
P = predict_probs(model, steps)                          # [2]+[3] 하드라우팅 base (non-g*=0)
import corrector as _Cmask; P_hard = P.copy()            # 하드라우팅 g* 보존(최종 그룹마스크용, 폴백이 P 덮어써도)
# 스페셜리스트 제거(07-06): held-out -0.0005 브리틀 개입 → 코드 다이어트

# [3.8] 풀 파인튜닝 e5 raw 확률 Ff (full-context) — 코렉터/폴백 공용
Ff = None
if _HAS_FULL:
    import torch.nn as _nn2
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _dt = torch.float16 if dev == "cuda" else torch.float32
    _cf = json.load(open("./model/e5_full/config_ft.json")) if os.path.isfile("./model/e5_full/config_ft.json") else {}
    _ml = _cf.get("maxlen", 320); _hu = _cf.get("hu", 3); _ha = _cf.get("ha", 4); _hr = _cf.get("hr", 2)   # config 없으면 v7 기본
    tokf = AutoTokenizer.from_pretrained("./model/e5_full")
    encf = AutoModel.from_pretrained("./model/e5_full", torch_dtype=_dt).to(dev).eval()
    headf = _nn2.Linear(encf.config.hidden_size, len(ACTIONS))    # 모델 차원 자동 (e5-small 384 / e5-base 768)
    headf.load_state_dict(torch.load("./model/e5_full/head.pt", map_location="cpu"))
    headf = headf.to(dev).to(_dt).eval()
    def _ctx(s):
        us = [c for t, c, r in s["hist"] if t == "U"][-_hu:]; ac = [c for t, c, r in s["hist"] if t == "A"][-_ha:]
        rs = [r for t, c, r in s["hist"] if t == "A" and r][-_hr:]
        return (s["prompt"] + " || U " + " ".join(us) + " || A " + " ".join(ac) + " || R " + " ".join(rs)
                + f" || {s['tier']} {s['ci']} {s['lang']} loc{int(s['loc']>2000)} bud{int(s['budget']>100000)}")
    Ff = np.zeros((len(steps), len(ACTIONS)), np.float32)
    with torch.no_grad():
        for lo in range(0, len(steps), 128):
            js = list(range(lo, min(lo + 128, len(steps))))
            e = tokf(["query: " + _ctx(steps[i]) for i in js], padding=True, truncation=True,
                     max_length=_ml, return_tensors="pt").to(dev)
            h = encf(**e).last_hidden_state; m = e["attention_mask"].unsqueeze(-1)
            Ff[js] = torch.softmax(headf((h * m).sum(1) / m.sum(1)).float(), dim=1).cpu().numpy()

if _HAS_CORR and Ff is not None:                         # [코렉터] base+풀ft 위에 순서/args 교정 (챔피언 경로)
    import corrector as C
    cw = json.load(open("./model/corrector_w.json"))
    w1 = cw["w1"]; w2 = cw["w2"]; offs = np.array(cw["offsets"])
    clf = joblib.load("./model/corrector.pkl")
    P1 = (1 - w1) * P + w1 * Ff
    import meta_feats as MF                               # [미개척 메타] open_files 신호 스택
    hist_by_id = {}; rec_by_id = {}
    with open("./data/test.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line); rec_by_id[d["id"]] = d; hist_by_id[d["id"]] = C.parse_history(d["history"])
    _sids = [s["id"] for s in steps]
    seqX = np.hstack([C.build_seq_feats(hist_by_id, _sids), MF.build_meta_feats(rec_by_id, _sids)])
    Q = C.corrector_proba(clf, P1, seqX)
    P2 = (1 - w2) * P1 + w2 * Q
    if os.path.isdir("./model/e5_ft"):                       # [au 전용] au-specialist e5 재블렌드(au행만, +0.0051 au CV)
        import torch.nn as _nnA
        _dtA = torch.float16 if torch.cuda.is_available() else torch.float32
        _devA = "cuda" if torch.cuda.is_available() else "cpu"
        _AUW = float(json.load(open("./model/au_w.json"))["w"]) if os.path.isfile("./model/au_w.json") else 0.3
        tok_a = AutoTokenizer.from_pretrained("./model/e5_ft")
        enc_a = AutoModel.from_pretrained("./model/e5_ft", torch_dtype=_dtA).to(_devA).eval()
        head_a = _nnA.Linear(enc_a.config.hidden_size, len(ACTIONS))
        head_a.load_state_dict(torch.load("./model/e5_ft/head.pt", map_location="cpu"))
        head_a = head_a.to(_devA).to(_dtA).eval()
        _idm = torch.tensor(np.load("./model/e5_ft/id_map.npy")).to(_devA) if os.path.isfile("./model/e5_ft/id_map.npy") else None  # 어휘 프루닝(e5-base) 시 id 리맵
        au_idx = [i for i, s in enumerate(steps) if s["au"]]
        with torch.no_grad():
            for lo in range(0, len(au_idx), 128):
                ic = au_idx[lo:lo + 128]
                e = tok_a(["query: " + steps[i]["prompt"] for i in ic], padding=True, truncation=True, max_length=192, return_tensors="pt").to(_devA)
                if _idm is not None: e["input_ids"] = _idm[e["input_ids"]]
                h = enc_a(**e).last_hidden_state; m = e["attention_mask"].unsqueeze(-1)
                pa = torch.softmax(head_a((h * m).sum(1) / m.sum(1)).float(), 1).cpu().numpy()
                for r, i in enumerate(ic):
                    P2[i] = (1 - _AUW) * P2[i] + _AUW * pa[r]
        print(f"au-e5 reblend: {len(au_idx)} au rows, w={_AUW}")
    _am = (np.log(_Cmask.group_mask(P_hard, P2) + 1e-9) + offs).argmax(1)   # 하드라우팅 g* 고정
    print(f"corrector blend: w1={w1} w2={w2}")
else:                                                    # 폴백: 오프셋 → 결정확률 → 풀ft 블렌드 (구경로)
    offsets = bundle["offsets"]
    if os.path.exists("./model/ft_w.json"):
        offsets = np.array(json.load(open("./model/ft_w.json"))["offsets"])
    if _HAS_FT:                                          # [3.7] au 한정 파인튜닝 e5 블렌드 (e5_ft 있을 때만)
        import torch.nn as _nn
        w_ft = json.load(open("./model/ft_w.json"))["w"]
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _dt = torch.float16 if dev == "cuda" else torch.float32
        tok_ft = AutoTokenizer.from_pretrained("./model/e5_ft")
        enc_ft = AutoModel.from_pretrained("./model/e5_ft", torch_dtype=_dt).to(dev).eval()
        head = _nn.Linear(384, len(ACTIONS))
        head.load_state_dict(torch.load("./model/e5_ft/head.pt", map_location="cpu"))
        head = head.to(dev).to(_dt).eval()
        au_idx = [i for i, s in enumerate(steps) if s["au"]]
        with torch.no_grad():
            for lo in range(0, len(au_idx), 128):
                ids_c = au_idx[lo:lo + 128]
                e = tok_ft(["query: " + steps[i]["prompt"] for i in ids_c], padding=True,
                           truncation=True, max_length=192, return_tensors="pt").to(dev)
                h = enc_ft(**e).last_hidden_state
                m = e["attention_mask"].unsqueeze(-1)
                pf = torch.softmax(head((h * m).sum(1) / m.sum(1)).float(), dim=1).cpu().numpy()
                for r, i in enumerate(ids_c):
                    P[i] = (1 - w_ft) * P[i] + w_ft * pf[r]
    _lp = np.log(P + 1e-9) + offsets
    Dp = np.exp(_lp - _lp.max(1, keepdims=True)); Dp /= Dp.sum(1, keepdims=True)
    if Ff is not None:
        wf = json.load(open("./model/ft_full_w.json"))["w"]
        Dp = (1 - wf) * Dp + wf * Ff
        print(f"full-ft blend: {len(steps)} rows, w={wf}")
    _am = _Cmask.group_mask(P_hard, Dp).argmax(1)        # 하드라우팅 g* 고정

preds = [ACTIONS[_am[i]] for i in range(len(steps))]   # 룰 오버라이드 제거(07-06, 3fold +0.0005): rl 미적용, 순수 argmax

assert all(p in set(ACTIONS) for p in preds)
sub = pd.read_csv("./data/sample_submission.csv")
m = dict(zip([s["id"] for s in steps], preds))
sub["action"] = sub["id"].map(m).fillna("edit_file")
os.makedirs("./output", exist_ok=True)
sub.to_csv("./output/submission.csv", index=False)
print(f"done: {len(sub)} rows, rules={sum(1 for x in rl if x is not None)}")
'''

def assemble():
    sub = OUT / "submit"
    if sub.exists():
        shutil.rmtree(sub)
    (sub / "model").mkdir(parents=True)
    shutil.copy(OUT / "bundle.joblib", sub / "model" / "bundle.joblib")
    for m in MODULES:
        shutil.copy(OUT / m, sub / "model" / m)
    if (OUT / "emb_cache.npz").exists():                       # 임베딩 번들 → 얼린 e5(fp16) 동봉
        import os, torch
        os.environ["HF_HOME"] = str(OUT / "hf_cache"); os.environ["HF_HUB_OFFLINE"] = "1"
        from transformers import AutoTokenizer, AutoModel
        name = "dragonkue/multilingual-e5-small-ko-v2"
        AutoTokenizer.from_pretrained(name).save_pretrained(sub / "model" / "e5")
        AutoModel.from_pretrained(name, torch_dtype=torch.float16).save_pretrained(sub / "model" / "e5", safe_serialization=True)
        print("e5(fp16) 동봉 완료")
    if (OUT / "ft_w.json").exists():                                # 오프셋 설정 (e5_ft 모델은 용량 위해 미동봉 — e5_full이 au 커버)
        shutil.copy(OUT / "ft_w.json", sub / "model" / "ft_w.json")
    if (OUT / "e5_full").is_dir() and (OUT / "ft_full_w.json").exists():   # 풀 파인튜닝 e5 + 블렌드 설정
        shutil.copytree(OUT / "e5_full", sub / "model" / "e5_full")
        shutil.copy(OUT / "ft_full_w.json", sub / "model" / "ft_full_w.json")
        print("e5_full(풀 파인튜닝) 동봉 완료")
    if (OUT / "corrector.pkl").exists() and (OUT / "corrector_w.json").exists():   # 코렉터(순서/args 교정)
        shutil.copy(OUT / "corrector.pkl", sub / "model" / "corrector.pkl")
        shutil.copy(OUT / "corrector_w.json", sub / "model" / "corrector_w.json")
        print("corrector 동봉 완료")
    _ausrc = OUT / "e5_ft_pruned" if (os.environ.get("USE_PRUNED", "1") == "1" and (OUT / "e5_ft_pruned").is_dir()) else OUT / "e5_ft"  # 프루닝 e5-base au 우선(au 0.8953)
    _auw = float(os.environ.get("AU_W", "0.3"))
    if _ausrc.is_dir():                                            # au-specialist e5 (au 재블렌드용)
        shutil.copytree(_ausrc, sub / "model" / "e5_ft")
        json.dump({"w": _auw}, open(sub / "model" / "au_w.json", "w"))
        print(f"{_ausrc.name}(au specialist) 동봉 완료, w={_auw}")
    (sub / "script.py").write_text(SCRIPT, encoding="utf-8")
    (sub / "requirements.txt").write_text("", encoding="utf-8")
    zp = OUT / "submit.zip"
    if zp.exists():
        zp.unlink()
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sub.rglob("*"):
            z.write(p, p.relative_to(sub))
    print(f"submit.zip {zp.stat().st_size/1e6:.0f}MB (한도 1GB)")
    return sub

def mock(sub):
    import os
    mock_dir = OUT / "mock"
    if mock_dir.exists():
        try:
            shutil.rmtree(mock_dir)
        except PermissionError:                       # 다른 프로세스가 폴더 잠금 → 대체 경로
            mock_dir = OUT / f"mock_{os.getpid()}"
            if mock_dir.exists():
                shutil.rmtree(mock_dir)
    shutil.copytree(sub, mock_dir)
    (mock_dir / "data").mkdir()
    n = 30000
    with open(DATA / "train.jsonl", encoding="utf-8") as f, \
         open(mock_dir / "data" / "test.jsonl", "w", encoding="utf-8") as g:
        ids = []
        for i, line in enumerate(f):
            if i >= n:
                break
            g.write(line)
            ids.append(json.loads(line)["id"])
    with open(mock_dir / "data" / "sample_submission.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "action"])
        for i in ids:
            w.writerow([i, "read_file"])
    t0 = time.time()
    r = subprocess.run([str(PY), "script.py"], cwd=mock_dir, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=900)
    dt = time.time() - t0
    print(r.stdout.strip()); print(r.stderr[-2000:] if r.returncode else "", end="")
    outp = mock_dir / "output" / "submission.csv"
    ok = outp.exists()
    if ok:
        rows = list(csv.DictReader(open(outp, encoding="utf-8")))
        from build_graph import GROUPS
        ok = len(rows) == n and all(row["action"] in GROUPS for row in rows)
    print(f"모의채점: 실행 {dt:.0f}초 (서버 상한 600초), 형식 {'OK' if ok else 'FAIL'}, exit={r.returncode}")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    s = assemble()
    mock(s)
