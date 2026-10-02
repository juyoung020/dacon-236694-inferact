# -*- coding: utf-8 -*-
"""au 풀컨텍스트 e5 파인튜닝 — 프롬프트-only가 아니라 프롬프트+히스토리+메타(ctx). au는 궤적 신호가 큼(Detective B).
au 0.8694(프롬프트only) 넘길 수 있나. 단일모델=전이 유력. OOF→ft_oof_full.pkl, --full→e5_ft_full/."""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
def _p(rel):  # 저장소 루트 기준 경로
    return _os.path.join(_ROOT, *[x for x in rel.split('\\') if x])
import argparse, json, os, pickle, random, sys, time
import numpy as np
OUT = _p(r"work\onto")
os.environ["HF_HOME"] = OUT + r"\hf_cache"; os.environ["HF_HUB_OFFLINE"] = "1"
import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel
sys.path.insert(0, OUT)
from build_graph import ACTIONS
NAME = os.environ.get("FT_MODEL", "dragonkue/multilingual-e5-small-ko-v2")
A_IDX = {a: i for i, a in enumerate(ACTIONS)}
EPOCHS = int(os.environ.get("FT_EPOCHS", "4")); LR = 3e-5; BATCH = 32; MAXLEN = 320
HU, HA, HR = 3, 4, 2


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def ctx(s):
    us = [c for t, c, r in s["hist"] if t == "U"][-HU:]; ac = [c for t, c, r in s["hist"] if t == "A"][-HA:]
    rs = [r for t, c, r in s["hist"] if t == "A" and r][-HR:]
    return (s["prompt"] + " || U " + " ".join(us) + " || A " + " ".join(ac) + " || R " + " ".join(rs)
            + f" || {s['tier']} {s['ci']} {s['lang']} loc{int(s['loc']>2000)} bud{int(s['budget']>100000)}")


class FtModel(nn.Module):
    def __init__(s):
        super().__init__(); s.enc = AutoModel.from_pretrained(NAME); s.head = nn.Linear(s.enc.config.hidden_size, len(ACTIONS))

    def forward(s, **e):
        h = s.enc(**e).last_hidden_state; m = e["attention_mask"].unsqueeze(-1)
        return s.head((h * m).sum(1) / m.sum(1))


def train_one(tok, tr, dev, seed):
    set_seed(seed); model = FtModel().to(dev)
    cnt = np.bincount([A_IDX[s["label"]] for s in tr], minlength=len(ACTIONS))
    w = torch.tensor((cnt.sum() / np.maximum(cnt, 1)) ** 0.5, dtype=torch.float32).to(dev)
    lossf = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1); opt = torch.optim.AdamW(model.parameters(), lr=LR)
    texts = [ctx(s) for s in tr]; ys = [A_IDX[s["label"]] for s in tr]
    order = list(range(len(tr))); steps = EPOCHS * ((len(tr) + BATCH - 1) // BATCH)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1); model.train()
    for ep in range(EPOCHS):
        random.Random(seed + ep).shuffle(order)
        for lo in range(0, len(order), BATCH):
            idx = order[lo:lo + BATCH]
            e = tok(["query: " + texts[i] for i in idx], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(dev)
            y = torch.tensor([ys[i] for i in idx], dtype=torch.long).to(dev)
            opt.zero_grad(); loss = lossf(model(**e), y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
    return model


@torch.no_grad()
def predict(tok, model, rows, dev):
    model.eval(); out = {}; texts = [ctx(s) for s in rows]
    for lo in range(0, len(rows), 128):
        e = tok(["query: " + t for t in texts[lo:lo + 128]], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(dev)
        p = torch.softmax(model(**e), 1).float().cpu().numpy()
        for r, s in enumerate(rows[lo:lo + 128]):
            out[s["id"]] = p[r]
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(); ap.add_argument("--full", action="store_true"); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(NAME)
    g = pickle.load(open(OUT + r"\graph.pkl", "rb")); au = [s for s in g["steps"] if s["au"]]
    print(f"au {len(au)} 행, 풀컨텍스트, epochs={EPOCHS}", flush=True)
    if a.full:
        m = train_one(tok, au, dev, a.seed)
        m.enc.half().save_pretrained(OUT + r"\e5_ft_full"); tok.save_pretrained(OUT + r"\e5_ft_full")
        torch.save(m.head.state_dict(), OUT + r"\e5_ft_full\head.pt")
        print("→ e5_ft_full/ 저장", flush=True); return
    folds = json.load(open(OUT + r"\folds.json", encoding="utf-8")); preds = {}
    from sklearn.metrics import f1_score
    for k in (0, 1, 2):
        t0 = time.time(); tr = [s for s in au if folds[s["id"]] != k]; va = [s for s in au if folds[s["id"]] == k]
        m = train_one(tok, tr, dev, a.seed); preds.update(predict(tok, m, va, dev))
        acc = np.mean([ACTIONS[preds[s["id"]].argmax()] == s["label"] for s in va])
        print(f"fold{k}: acc {acc:.4f} ({time.time()-t0:.0f}s)", flush=True); del m; torch.cuda.empty_cache()
    pickle.dump(preds, open(OUT + r"\ft_oof_full.pkl", "wb")); print("→ ft_oof_full.pkl 저장", flush=True)


if __name__ == "__main__":
    main()
