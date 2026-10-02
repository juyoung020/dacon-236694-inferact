# -*- coding: utf-8 -*-
"""멀티태스크 e5 — action 헤드 + 보조 group 헤드(joint loss). 공유 인코더 정규화 시도(구조변경).
추론은 action softmax만. OOF → ft_full_oof_mt.pkl. λ_group=0.3."""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
def _p(rel):  # 저장소 루트 기준 경로
    return _os.path.join(_ROOT, *[x for x in rel.split('\\') if x])
import argparse, os, pickle, json, sys, time, random
import numpy as np
OUT = _p(r"work\onto")
os.environ["HF_HOME"] = OUT + r"\hf_cache"; os.environ["HF_HUB_OFFLINE"] = "1"
import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel
sys.path.insert(0, OUT)
from build_graph import ACTIONS, GROUPS, GROUP_LIST
from sklearn.metrics import f1_score
AI = {a: i for i, a in enumerate(ACTIONS)}
GI = {g: i for i, g in enumerate(GROUP_LIST)}
GofA = np.array([GI[GROUPS[a]] for a in ACTIONS])
NAME = "dragonkue/multilingual-e5-small-ko-v2"
EPOCHS = int(os.environ.get("FT_EPOCHS", "4")); LR = 3e-5; BATCH = 32; MAXLEN = 320
HU, HA, HR = 3, 4, 2; SEED = int(os.environ.get("FT_SEED", "42")); LAM = float(os.environ.get("FT_LAM", "0.3"))


def seed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)


def ctx(s):
    us = [c for t, c, r in s["hist"] if t == "U"][-HU:]; ac = [c for t, c, r in s["hist"] if t == "A"][-HA:]
    rs = [r for t, c, r in s["hist"] if t == "A" and r][-HR:]
    return (s["prompt"] + " || U " + " ".join(us) + " || A " + " ".join(ac) + " || R " + " ".join(rs)
            + f" || {s['tier']} {s['ci']} {s['lang']} loc{int(s['loc']>2000)} bud{int(s['budget']>100000)}")


class M(nn.Module):
    def __init__(s):
        super().__init__(); s.enc = AutoModel.from_pretrained(NAME)
        s.head = nn.Linear(s.enc.config.hidden_size, 14); s.ghead = nn.Linear(s.enc.config.hidden_size, 4)

    def forward(s, **e):
        h = s.enc(**e).last_hidden_state; m = e["attention_mask"].unsqueeze(-1)
        pooled = (h * m).sum(1) / m.sum(1)
        return s.head(pooled), s.ghead(pooled)


g = pickle.load(open(OUT + r"\graph.pkl", "rb")); steps = g["steps"]
Y = [s["label"] for s in steps]; TEXTS = [ctx(s) for s in steps]
folds = json.load(open(OUT + r"\folds.json", encoding="utf-8")); FK = np.array([folds[s["id"]] for s in steps])
cnt = np.bincount([AI[a] for a in Y], minlength=14); CW = (cnt.sum() / np.maximum(cnt, 1)) ** 0.5


def train(rows, cw, dev, tok):
    seed(); model = M().to(dev); opt = torch.optim.AdamW(model.parameters(), lr=LR)
    cwt = torch.tensor(cw, dtype=torch.float32).to(dev); lf = nn.CrossEntropyLoss(weight=cwt); lg = nn.CrossEntropyLoss()
    tot = EPOCHS * ((len(rows) + BATCH - 1) // BATCH)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=tot, pct_start=0.1)
    model.train()
    for ep in range(EPOCHS):
        order = list(rows); random.Random(SEED + ep).shuffle(order)
        for lo in range(0, len(order), BATCH):
            idx = order[lo:lo + BATCH]
            e = tok(["query: " + TEXTS[i] for i in idx], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(dev)
            ya = torch.tensor([AI[Y[i]] for i in idx], dtype=torch.long).to(dev)
            yg = torch.tensor([int(GofA[AI[Y[i]]]) for i in idx], dtype=torch.long).to(dev)
            opt.zero_grad(); la, lgt = model(**e)
            loss = lf(la, ya) + LAM * lg(lgt, yg)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
    return model


@torch.no_grad()
def predict(model, rows, dev, tok):
    model.eval(); out = {}
    for lo in range(0, len(rows), 128):
        idx = rows[lo:lo + 128]
        e = tok(["query: " + TEXTS[i] for i in idx], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(dev)
        p = torch.softmax(model(**e)[0], 1).cpu().numpy()
        for r, i in enumerate(idx):
            out[steps[i]["id"]] = p[r]
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    dev = "cuda" if torch.cuda.is_available() else "cpu"; tok = AutoTokenizer.from_pretrained(NAME)
    print(f"multi-task e5 λ_group={LAM} seed={SEED}", flush=True)
    oof = {}
    for k in (0, 1, 2):
        t0 = time.time(); tr = [i for i in range(len(steps)) if FK[i] != k]; va = [i for i in range(len(steps)) if FK[i] == k]
        m = train(tr, CW, dev, tok); oof.update(predict(m, va, dev, tok))
        yv = [Y[i] for i in va]; pv = [ACTIONS[oof[steps[i]["id"]].argmax()] for i in va]
        print(f"fold{k}: Macro-F1 {f1_score(yv, pv, average='macro'):.4f} ({time.time()-t0:.0f}s)", flush=True)
        del m; torch.cuda.empty_cache()
        pickle.dump(oof, open(OUT + r"\ft_full_oof_mt.pkl", "wb"))
    print("→ ft_full_oof_mt.pkl 완료", flush=True)


if __name__ == "__main__":
    main()
