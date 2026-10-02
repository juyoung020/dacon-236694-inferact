# -*- coding: utf-8 -*-
"""OOF 임베딩 추출 — fold별 e5(다른 fold로 학습)로 val fold 임베딩. leak 없음(일반화 표현).
honest kNN 리트리벌용. ft_full의 train/M 재사용. 저장: oof_ctx_emb.npz."""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
def _p(rel):  # 저장소 루트 기준 경로
    return _os.path.join(_ROOT, *[x for x in rel.split('\\') if x])
import os, sys, json, time
import numpy as np
OUT = _p(r"work\onto")
os.environ["HF_HOME"] = OUT + r"\hf_cache"; os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, OUT)
sys.stdout.reconfigure(encoding="utf-8")
import torch
import ft_full as F   # 모듈 로드: steps, TEXTS, Y, FK, CW, tok설정 등

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = F.AutoTokenizer.from_pretrained(F.NAME)
steps = F.steps; TEXTS = F.TEXTS; FK = F.FK; ML = F.MAXLEN
ids = [s["id"] for s in steps]
E = np.zeros((len(steps), 384), np.float32)


@torch.no_grad()
def embed(model, rows):
    model.eval()
    for lo in range(0, len(rows), 256):
        idx = rows[lo:lo + 256]
        e = tok(["query: " + TEXTS[i] for i in idx], padding=True, truncation=True, max_length=ML, return_tensors="pt").to(dev)
        h = model.enc(**e).last_hidden_state; m = e["attention_mask"].unsqueeze(-1)
        v = (h * m).sum(1) / m.sum(1)
        v = torch.nn.functional.normalize(v, dim=1).float().cpu().numpy()
        for r, i in enumerate(idx):
            E[i] = v[r]


for k in (0, 1, 2):
    t0 = time.time()
    tr = [i for i in range(len(steps)) if FK[i] != k]
    va = [i for i in range(len(steps)) if FK[i] == k]
    model = F.train(TEXTS, tr, F.CW, dev, tok)
    embed(model, va)
    print(f"fold{k}: train {len(tr)} → val {len(va)} 임베딩 추출 ({time.time()-t0:.0f}s)", flush=True)
    del model; torch.cuda.empty_cache()
np.savez_compressed(OUT + r"\oof_ctx_emb.npz", ids=np.array(ids), emb=E.astype(np.float16))
print(f"저장 oof_ctx_emb.npz {E.shape}", flush=True)
