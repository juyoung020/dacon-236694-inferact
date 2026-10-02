# -*- coding: utf-8 -*-
"""e5-base au 임베딩 어휘 프루닝 — 전체 train 프롬프트에 등장한 토큰만 유지(+특수토큰).
임베딩 250k→사용토큰만(~50k)로 축소 → 552MB→~245MB(fp16, 품질손실 0). test는 같은 생성기라 커버.
저장: e5_ft_pruned/(축소모델) + id_map.npy(old→new). 추론시 input_ids 리맵 필요."""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
def _p(rel):  # 저장소 루트 기준 경로
    return _os.path.join(_ROOT, *[x for x in rel.split('\\') if x])
import os, sys, json, pickle, shutil
import numpy as np
OUT = _p(r"work\onto")
os.environ["HF_HOME"] = OUT + r"\hf_cache"; os.environ["HF_HUB_OFFLINE"] = "1"
import torch, torch.nn as nn
from transformers import AutoModel, AutoTokenizer
sys.path.insert(0, OUT)
sys.stdout.reconfigure(encoding="utf-8")
SRC = OUT + "\\" + os.environ.get("PRUNE_SRC", "e5_ft_200"); DST = OUT + r"\e5_ft_pruned"
HU, HA, HR = 3, 4, 2


def ctx(s):
    us=[c for t,c,r in s["hist"] if t=="U"][-HU:]; ac=[c for t,c,r in s["hist"] if t=="A"][-HA:]; rs=[r for t,c,r in s["hist"] if t=="A" and r][-HR:]
    return s["prompt"]+" || U "+" ".join(us)+" || A "+" ".join(ac)+" || R "+" ".join(rs)+f" || {s['tier']} {s['ci']} {s['lang']} loc{int(s['loc']>2000)} bud{int(s['budget']>100000)}"


tok = AutoTokenizer.from_pretrained(SRC)
g = pickle.load(open(OUT + r"\graph.pkl", "rb")); steps = g["steps"]
# 전체 train의 au+sim 프롬프트 컨텍스트 전부 토크나이즈 → 사용 토큰
used = set()
texts = ["query: " + ctx(s) for s in steps]
for lo in range(0, len(texts), 2000):
    enc = tok(texts[lo:lo + 2000], truncation=True, max_length=320)
    for ids in enc["input_ids"]:
        used.update(ids)
V = tok.vocab_size
special = set(tok.all_special_ids)
# 안전 버퍼: 특수토큰 + 사용토큰 전부 유지
keep = sorted(used | special)
print(f"vocab {V} → keep {len(keep)} ({len(keep)/V:.0%})", flush=True)
id_map = np.full(V, keep.index(tok.unk_token_id) if tok.unk_token_id in keep else 0, dtype=np.int64)
for new, old in enumerate(keep):
    id_map[old] = new
np.save(OUT + r"\_au_id_map.npy", id_map)

enc = AutoModel.from_pretrained(SRC, torch_dtype=torch.float16)
we = enc.get_input_embeddings().weight.data     # [V, dim]
new_pad = int(id_map[tok.pad_token_id])         # ★ XLM-R는 pad_token_id로 position_id 계산 → 새 pad id 반영
new_emb = nn.Embedding(len(keep), we.shape[1], padding_idx=new_pad).half()
new_emb.weight.data = we[torch.tensor(keep)].clone()
enc.set_input_embeddings(new_emb)
enc.config.vocab_size = len(keep)
enc.config.pad_token_id = new_pad
if hasattr(enc, "embeddings") and hasattr(enc.embeddings, "padding_idx"):
    enc.embeddings.padding_idx = new_pad
os.makedirs(DST, exist_ok=True)
enc.save_pretrained(DST); tok.save_pretrained(DST)
shutil.copy(SRC + r"\head.pt", DST + r"\head.pt")
np.save(DST + r"\id_map.npy", id_map)
print(f"→ e5_ft_pruned 저장 (vocab {len(keep)})", flush=True)

# 검증: 프루닝+리맵 모델이 원본 au acc와 일치하나
from build_graph import ACTIONS
au = [s for s in steps if s["au"]][:500]
head = nn.Linear(enc.config.hidden_size, 14).half(); head.load_state_dict(torch.load(SRC + r"\head.pt", map_location="cpu"))
dev = "cuda"; enc = enc.to(dev).eval(); head = head.to(dev).half().eval()
idm = torch.tensor(id_map).to(dev)
cor = 0
with torch.no_grad():
    for lo in range(0, len(au), 128):
        b = au[lo:lo + 128]; e = tok(["query: " + ctx(s) for s in b], padding=True, truncation=True, max_length=320, return_tensors="pt").to(dev)
        e["input_ids"] = idm[e["input_ids"]]       # ★ 리맵
        h = enc(**e).last_hidden_state; m = e["attention_mask"].unsqueeze(-1); p = torch.softmax(head((h * m).sum(1) / m.sum(1)).float(), 1).cpu().numpy()
        for r, s in enumerate(b): cor += (ACTIONS[p[r].argmax()] == s["label"])
print(f"프루닝+리맵 in-sample au acc {cor/len(au):.3f} (원본 fp16과 동일해야=프루닝 정상)", flush=True)
import os as _o
print(f"e5_ft_pruned 크기: {sum(_o.path.getsize(_o.path.join(DST,f)) for f in _o.listdir(DST))/1e6:.0f}MB", flush=True)
