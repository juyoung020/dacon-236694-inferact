# -*- coding: utf-8 -*-
"""au 모델 수프 — e5_ft(42)+e5_ft_1+e5_ft_7 가중치 평균 → 단일 강화 au모델(크기 불변).
저장 e5_ft_soup/. 배포시 e5_ft를 이걸로 교체."""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
def _p(rel):  # 저장소 루트 기준 경로
    return _os.path.join(_ROOT, *[x for x in rel.split('\\') if x])
import os, sys, json, glob
OUT = _p(r"work\onto")
os.environ["HF_HOME"] = OUT + r"\hf_cache"; os.environ["HF_HUB_OFFLINE"] = "1"
import torch
from transformers import AutoModel, AutoTokenizer
sys.stdout.reconfigure(encoding="utf-8")

dirs = [OUT + r"\e5_ft"] + [d for d in [OUT + r"\e5_ft_1", OUT + r"\e5_ft_7"] if os.path.isdir(d)]
print(f"수프 대상 {len(dirs)}개: {[os.path.basename(d) for d in dirs]}", flush=True)

# 인코더 수프
encs = [AutoModel.from_pretrained(d, torch_dtype=torch.float32) for d in dirs]
sd = {k: sum(e.state_dict()[k].float() for e in encs) / len(encs) for k in encs[0].state_dict()}
encs[0].load_state_dict(sd)
os.makedirs(OUT + r"\e5_ft_soup", exist_ok=True)
encs[0].half().save_pretrained(OUT + r"\e5_ft_soup")
AutoTokenizer.from_pretrained(dirs[0]).save_pretrained(OUT + r"\e5_ft_soup")
# 헤드 수프
heads = [torch.load(d + r"\head.pt", map_location="cpu") for d in dirs]
hsd = {k: sum(h[k].float() for h in heads) / len(heads) for k in heads[0]}
torch.save(hsd, OUT + r"\e5_ft_soup\head.pt")
print(f"→ e5_ft_soup 저장 ({len(dirs)}-model soup)", flush=True)
