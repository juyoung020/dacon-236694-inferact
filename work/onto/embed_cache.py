# -*- coding: utf-8 -*-
"""얼린 e5 임베딩 캐시 구축 (파인튜닝 없음).

train 프롬프트 + observed_event 발화의 고유 문장 전체를 GPU로 1회 인코딩
→ emb_cache.npz (texts 순서 = hash 순서, float16). 이후 실험은 조회만.
모델: dragonkue/multilingual-e5-small-ko-v2 (MIT, 과거 캐시에서 복사) — e5 규약대로 'query: ' 프리픽스.
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
def _p(rel):  # 저장소 루트 기준 경로
    return _os.path.join(_ROOT, *[x for x in rel.split('\\') if x])
import os, pickle, sys, time
import numpy as np
from pathlib import Path

OUT = Path(_p(r"work\onto"))
os.environ["HF_HOME"] = str(OUT / "hf_cache")
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from transformers import AutoTokenizer, AutoModel

MODEL = "dragonkue/multilingual-e5-small-ko-v2"

def encode(texts, tok, model, device, batch=256, max_len=192):
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float16)
    model.eval()
    with torch.no_grad():
        for lo in range(0, len(texts), batch):
            chunk = ["query: " + t for t in texts[lo:lo + batch]]
            enc = tok(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            v = (h * mask).sum(1) / mask.sum(1)                     # mean pooling
            v = torch.nn.functional.normalize(v, dim=1)
            out[lo:lo + len(chunk)] = v.cpu().numpy().astype(np.float16)
    return out

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    g = pickle.load(open(OUT / "graph.pkl", "rb"))
    texts = set()
    for s in g["steps"]:
        texts.add(s["prompt"])
    for evs in g["events"].values():
        for ev in evs:
            texts.add(ev[0])
    texts = sorted(texts)
    print(f"고유 문장 {len(texts):,}개 인코딩 시작")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device)
    t0 = time.time()
    E = encode(texts, tok, model, device)
    np.savez_compressed(OUT / "emb_cache.npz", texts=np.array(texts, dtype=object), emb=E)
    print(f"완료: {E.shape} ({time.time()-t0:.0f}s, {device}) → emb_cache.npz")

if __name__ == "__main__":
    main()
