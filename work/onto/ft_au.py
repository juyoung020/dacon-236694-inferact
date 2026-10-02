# -*- coding: utf-8 -*-
"""au 한정 e5 파인튜닝 — 14클래스 분류기, fold별 정직 OOF 예측.

- 데이터: au 행(5,025)만. sim은 의미가 뒤섞여 파인튜닝이 노이즈 학습이 됨(실측 원칙).
- fold k 예측은 fold k를 학습에서 뺀 모델이 수행 (기존 folds.json 그대로).
- 산출: ft_oof.pkl {id: 14차원 softmax}  /  --full: 전체 au로 학습 → e5_ft/ 저장(제출 동봉용)
사용: python ft_au.py          → 3-fold OOF
      python ft_au.py --full   → 최종 모델 저장
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
def _p(rel):  # 저장소 루트 기준 경로
    return _os.path.join(_ROOT, *[x for x in rel.split('\\') if x])
import argparse, json, os, pickle, random, sys, time
import numpy as np

OUT = _p(r"work\onto")
os.environ["HF_HOME"] = OUT + r"\hf_cache"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, OUT)
from build_graph import ACTIONS

NAME = os.environ.get("FT_MODEL", "dragonkue/multilingual-e5-small-ko-v2")
A_IDX = {a: i for i, a in enumerate(ACTIONS)}
SEED = 42          # --seed로 변경 가능 (시드 배깅용)
EPOCHS = int(os.environ.get("FT_EPOCHS", "3"))
LR = 3e-5
BATCH = 32
MAXLEN = 192

def set_seed():
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

class FtModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(NAME)
        self.head = nn.Linear(self.enc.config.hidden_size, len(ACTIONS))

    def forward(self, **enc_in):
        h = self.enc(**enc_in).last_hidden_state
        m = enc_in["attention_mask"].unsqueeze(-1)
        pooled = (h * m).sum(1) / m.sum(1)
        return self.head(pooled)

def make_loader(tok, rows, shuffle):
    texts = ["query: " + s["prompt"] for s in rows]
    ys = torch.tensor([A_IDX[s["label"]] for s in rows])
    def collate(idx):
        e = tok([texts[i] for i in idx], padding=True, truncation=True,
                max_length=MAXLEN, return_tensors="pt")
        return e, ys[list(idx)]
    return DataLoader(range(len(rows)), batch_size=BATCH, shuffle=shuffle, collate_fn=collate)

def train_one(tok, tr_rows, dev):
    set_seed()
    model = FtModel().to(dev)
    cnt = np.bincount([A_IDX[s["label"]] for s in tr_rows], minlength=len(ACTIONS))
    w = torch.tensor((cnt.sum() / np.maximum(cnt, 1)) ** 0.5, dtype=torch.float32).to(dev)  # 완화된 균형 가중
    lossf = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    dl = make_loader(tok, tr_rows, shuffle=True)
    steps = EPOCHS * len(dl)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    model.train()
    for ep in range(EPOCHS):
        for e, y in dl:
            e = {k: v.to(dev) for k, v in e.items()}; y = y.to(dev)
            opt.zero_grad()
            loss = lossf(model(**e), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
    return model

@torch.no_grad()
def predict(tok, model, rows, dev):
    model.eval()
    out = {}
    texts = ["query: " + s["prompt"] for s in rows]
    for lo in range(0, len(rows), 128):
        e = tok(texts[lo:lo + 128], padding=True, truncation=True, max_length=MAXLEN,
                return_tensors="pt").to(dev)
        p = torch.softmax(model(**e), dim=1).float().cpu().numpy()
        for r, s in enumerate(rows[lo:lo + 128]):
            out[s["id"]] = p[r]
    return out

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(); ap.add_argument("--full", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    global SEED
    SEED = args.seed
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}")
    g = pickle.load(open(OUT + r"\graph.pkl", "rb"))
    au_rows = [s for s in g["steps"] if s["au"]]
    print(f"au 행 {len(au_rows)}")
    tok = AutoTokenizer.from_pretrained(NAME)

    if args.full:
        t0 = time.time()
        model = train_one(tok, au_rows, dev)
        sfx = "" if SEED == 42 else f"_{SEED}"
        model.enc.half().save_pretrained(OUT + rf"\e5_ft{sfx}")
        tok.save_pretrained(OUT + rf"\e5_ft{sfx}")
        torch.save(model.head.state_dict(), OUT + rf"\e5_ft{sfx}\head.pt")
        print(f"--full(seed {SEED}) 저장 완료 ({time.time()-t0:.0f}s) → e5_ft{sfx}/")
        return

    folds = json.load(open(OUT + r"\folds.json", encoding="utf-8"))
    preds = {}
    for k in (0, 1, 2):
        t0 = time.time()
        tr = [s for s in au_rows if folds[s["id"]] != k]
        va = [s for s in au_rows if folds[s["id"]] == k]
        model = train_one(tok, tr, dev)
        preds.update(predict(tok, model, va, dev))
        acc = np.mean([ACTIONS[np.argmax(preds[s["id"]])] == s["label"] for s in va])
        print(f"fold{k}: train {len(tr)} → val {len(va)}, argmax 정확도 {acc:.4f} ({time.time()-t0:.0f}s)")
        del model; torch.cuda.empty_cache()
    sfx = "" if SEED == 42 else f"_{SEED}"
    with open(OUT + rf"\ft_oof{sfx}.pkl", "wb") as f:
        pickle.dump(preds, f)
    print(f"→ ft_oof{sfx}.pkl 저장 ({len(preds)}행, seed {SEED})")

if __name__ == "__main__":
    main()
