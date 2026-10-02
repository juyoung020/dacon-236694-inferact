# -*- coding: utf-8 -*-
"""e5-base au 모델 int8 양자화(저장크기 절반, 추론시 fp16 디퀀트). 외부 라이브러리 불필요.
2D weight만 per-channel int8, 나머지(norm/bias/emb)는 fp16. 저장: e5_ft_int8/ (int8_state.pt + config/tokenizer)."""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
def _p(rel):  # 저장소 루트 기준 경로
    return _os.path.join(_ROOT, *[x for x in rel.split('\\') if x])
import os, sys, shutil, json
OUT = _p(r"work\onto")
os.environ["HF_HOME"] = OUT + r"\hf_cache"; os.environ["HF_HUB_OFFLINE"] = "1"
import torch, torch.nn as nn
from transformers import AutoModel, AutoTokenizer
sys.stdout.reconfigure(encoding="utf-8")
SRC = OUT + r"\e5_ft_200"; DST = OUT + r"\e5_ft_int8"

enc = AutoModel.from_pretrained(SRC, torch_dtype=torch.float32)
head = nn.Linear(enc.config.hidden_size, 14); head.load_state_dict(torch.load(SRC + r"\head.pt", map_location="cpu"))
sd = dict(enc.state_dict()); sd = {"enc." + k: v for k, v in sd.items()}
sd.update({"head." + k: v for k, v in head.state_dict().items()})

q = {}
n_int8 = n_fp16 = 0
for k, v in sd.items():
    if v.dim() == 2 and v.dtype == torch.float32 and min(v.shape) > 16:   # 큰 2D weight만 int8(per-out-channel)
        scale = v.abs().amax(dim=1, keepdim=True) / 127.0
        scale = torch.clamp(scale, min=1e-8)
        qi = torch.clamp((v / scale).round(), -127, 127).to(torch.int8)
        q[k] = qi; q[k + ".__scale__"] = scale.half(); n_int8 += 1
    else:
        q[k] = v.half(); n_fp16 += 1
torch.save(q, OUT + r"\_au_int8_state.pt")
os.makedirs(DST, exist_ok=True)
for f in ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model"]:
    if os.path.isfile(SRC + "\\" + f):
        shutil.copy(SRC + "\\" + f, DST + "\\" + f)
shutil.copy(OUT + r"\_au_int8_state.pt", DST + r"\int8_state.pt")
sz = os.path.getsize(DST + r"\int8_state.pt") / 1e6
print(f"int8 저장: {n_int8} tensor int8, {n_fp16} fp16 → {sz:.0f}MB (원본 552MB)", flush=True)

# 디퀀트 검증: 재구성 후 예측이 원본과 일치하나
import pickle, numpy as np
from build_graph import ACTIONS
g = pickle.load(open(OUT + r"\graph.pkl", "rb")); au = [s for s in g["steps"] if s["au"]][:400]
HU, HA, HR = 3, 4, 2
def ctx(s):
    us=[c for t,c,r in s["hist"] if t=="U"][-HU:]; ac=[c for t,c,r in s["hist"] if t=="A"][-HA:]; rs=[r for t,c,r in s["hist"] if t=="A" and r][-HR:]
    return s["prompt"]+" || U "+" ".join(us)+" || A "+" ".join(ac)+" || R "+" ".join(rs)+f" || {s['tier']} {s['ci']} {s['lang']} loc{int(s['loc']>2000)} bud{int(s['budget']>100000)}"
# 재구성
q = torch.load(DST + r"\int8_state.pt", map_location="cpu")
rec = {}
for k in list(q):
    if k.endswith(".__scale__"): continue
    if k + ".__scale__" in q: rec[k] = (q[k].float() * q[k + ".__scale__"].float()).half()
    else: rec[k] = q[k]
enc2 = AutoModel.from_config(enc.config).half(); enc2.load_state_dict({k[4:]: v for k, v in rec.items() if k.startswith("enc.")})
head2 = nn.Linear(enc.config.hidden_size, 14).half(); head2.load_state_dict({k[5:]: v for k, v in rec.items() if k.startswith("head.")})
tok = AutoTokenizer.from_pretrained(SRC); dev="cuda"; enc2=enc2.to(dev).eval(); head2=head2.to(dev).eval()
cor=0
with torch.no_grad():
    for lo in range(0,len(au),128):
        b=au[lo:lo+128]; e=tok(["query: "+ctx(s) for s in b],padding=True,truncation=True,max_length=320,return_tensors="pt").to(dev)
        h=enc2(**e).last_hidden_state; m=e["attention_mask"].unsqueeze(-1); p=torch.softmax(head2((h*m).sum(1)/m.sum(1)).float(),1).cpu().numpy()
        for r,s in enumerate(b): cor+=(ACTIONS[p[r].argmax()]==s["label"])
print(f"int8 디퀀트 in-sample au acc {cor/len(au):.3f} (원본 fp16과 유사해야=int8 정상)", flush=True)
