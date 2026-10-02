# -*- coding: utf-8 -*-
"""e5_full 인코더로 전 행 '컨텍스트' 임베딩 추출(384d, mean-pooled). kNN 리트리벌용.
프롬프트-only(emb_cache)와 달리 풀컨텍스트(ctx)라 상태/궤적 신호 포함. 저장: ctx_emb_frozen.npz."""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
def _p(rel):  # 저장소 루트 기준 경로
    return _os.path.join(_ROOT, *[x for x in rel.split('\\') if x])
import json, os, sys, pickle, time
import numpy as np
OUT = _p(r"work\onto")
os.environ["HF_HOME"] = OUT + r"\hf_cache"; os.environ["HF_HUB_OFFLINE"] = "1"
import torch
from transformers import AutoTokenizer, AutoModel
sys.path.insert(0, OUT)
sys.stdout.reconfigure(encoding="utf-8")

cf = json.load(open(OUT + r"\e5_full\config_ft.json")) if os.path.isfile(OUT + r"\e5_full\config_ft.json") else {}
ML = cf.get("maxlen", 320); HU = cf.get("hu", 3); HA = cf.get("ha", 4); HR = cf.get("hr", 2)
dev = "cuda" if torch.cuda.is_available() else "cpu"
dt = torch.float16 if dev == "cuda" else torch.float32
_FROZEN = "dragonkue/multilingual-e5-small-ko-v2"
tok = AutoTokenizer.from_pretrained(_FROZEN)
enc = AutoModel.from_pretrained(_FROZEN, torch_dtype=dt).to(dev).eval()

g = pickle.load(open(OUT + r"\graph.pkl", "rb")); steps = g["steps"]
ids = [s["id"] for s in steps]


def ctx(s):
    us = [c for t, c, r in s["hist"] if t == "U"][-HU:]; ac = [c for t, c, r in s["hist"] if t == "A"][-HA:]
    rs = [r for t, c, r in s["hist"] if t == "A" and r][-HR:]
    return (s["prompt"] + " || U " + " ".join(us) + " || A " + " ".join(ac) + " || R " + " ".join(rs)
            + f" || {s['tier']} {s['ci']} {s['lang']} loc{int(s['loc']>2000)} bud{int(s['budget']>100000)}")


TEXTS = [ctx(s) for s in steps]
E = np.zeros((len(steps), enc.config.hidden_size), np.float32)
t0 = time.time()
with torch.no_grad():
    for lo in range(0, len(steps), 256):
        e = tok(["query: " + TEXTS[i] for i in range(lo, min(lo + 256, len(steps)))],
                padding=True, truncation=True, max_length=ML, return_tensors="pt").to(dev)
        h = enc(**e).last_hidden_state; m = e["attention_mask"].unsqueeze(-1)
        v = (h * m).sum(1) / m.sum(1)
        v = torch.nn.functional.normalize(v, dim=1).float().cpu().numpy()
        E[lo:lo + v.shape[0]] = v
        if lo % 12800 == 0:
            print(f"  {lo}/{len(steps)} ({time.time()-t0:.0f}s)", flush=True)
np.savez_compressed(OUT + r"\ctx_emb_frozen.npz", ids=np.array(ids), emb=E.astype(np.float16))
print(f"저장 ctx_emb_frozen.npz {E.shape} ({time.time()-t0:.0f}s)", flush=True)
