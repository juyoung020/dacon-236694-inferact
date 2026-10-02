# -*- coding: utf-8 -*-
"""au KD — e5-base au(teacher, 0.8953) → e5-small au(student, 241MB 배포가능). 풀컨텍스트.
teacher soft = ft_oof_200(e5-base OOF). loss=0.5 CE(hard)+0.5 KL(soft,T=2). student가 teacher 경계 학습.
OOF→ft_oof_kd.pkl (au F1 측정), --full→e5_ft_kd/(teacher=e5-base full preds)."""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
def _p(rel):  # 저장소 루트 기준 경로
    return _os.path.join(_ROOT, *[x for x in rel.split('\\') if x])
import argparse, json, os, pickle, random, sys, time
import numpy as np
OUT = _p(r"work\onto")
os.environ["HF_HOME"] = OUT + r"\hf_cache"; os.environ["HF_HUB_OFFLINE"] = "1"
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
sys.path.insert(0, OUT)
from build_graph import ACTIONS
NAME = "dragonkue/multilingual-e5-small-ko-v2"; A_IDX = {a: i for i, a in enumerate(ACTIONS)}
EPOCHS = 5; LR = 3e-5; BATCH = 32; MAXLEN = 320; HU, HA, HR = 3, 4, 2; T = 2.0; ALPHA = 0.5


def ctx(s):
    us=[c for t,c,r in s["hist"] if t=="U"][-HU:]; ac=[c for t,c,r in s["hist"] if t=="A"][-HA:]; rs=[r for t,c,r in s["hist"] if t=="A" and r][-HR:]
    return s["prompt"]+" || U "+" ".join(us)+" || A "+" ".join(ac)+" || R "+" ".join(rs)+f" || {s['tier']} {s['ci']} {s['lang']} loc{int(s['loc']>2000)} bud{int(s['budget']>100000)}"


class M(nn.Module):
    def __init__(s): super().__init__(); s.enc=AutoModel.from_pretrained(NAME); s.head=nn.Linear(s.enc.config.hidden_size,14)
    def forward(s,**e): h=s.enc(**e).last_hidden_state; m=e["attention_mask"].unsqueeze(-1); return s.head((h*m).sum(1)/m.sum(1))


def train(tok, rows, teach, dev, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model=M().to(dev); cnt=np.bincount([A_IDX[s["label"]] for s in rows],minlength=14)
    w=torch.tensor((cnt.sum()/np.maximum(cnt,1))**0.5,dtype=torch.float32).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=LR); texts=[ctx(s) for s in rows]
    ys=[A_IDX[s["label"]] for s in rows]; ts=[teach[s["id"]] for s in rows]
    order=list(range(len(rows))); steps=EPOCHS*((len(rows)+BATCH-1)//BATCH)
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=LR,total_steps=steps,pct_start=0.1); model.train()
    for ep in range(EPOCHS):
        random.Random(seed+ep).shuffle(order)
        for lo in range(0,len(order),BATCH):
            idx=order[lo:lo+BATCH]
            e=tok(["query: "+texts[i] for i in idx],padding=True,truncation=True,max_length=MAXLEN,return_tensors="pt").to(dev)
            y=torch.tensor([ys[i] for i in idx],dtype=torch.long).to(dev)
            tsoft=torch.tensor(np.array([ts[i] for i in idx]),dtype=torch.float32).to(dev)
            opt.zero_grad(); logit=model(**e)
            ce=F.cross_entropy(logit,y,weight=w,label_smoothing=0.1)
            kl=F.kl_div(F.log_softmax(logit/T,1),torch.clamp(tsoft,1e-6,1),reduction="batchmean")*(T*T)
            (ALPHA*ce+(1-ALPHA)*kl).backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sch.step()
    return model


@torch.no_grad()
def predict(tok, model, rows, dev):
    model.eval(); out={}; texts=[ctx(s) for s in rows]
    for lo in range(0,len(rows),128):
        e=tok(["query: "+t for t in texts[lo:lo+128]],padding=True,truncation=True,max_length=MAXLEN,return_tensors="pt").to(dev)
        p=torch.softmax(model(**e),1).float().cpu().numpy()
        for r,s in enumerate(rows[lo:lo+128]): out[s["id"]]=p[r]
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap=argparse.ArgumentParser(); ap.add_argument("--full",action="store_true"); a=ap.parse_args()
    dev="cuda"; tok=AutoTokenizer.from_pretrained(NAME)
    g=pickle.load(open(OUT+r"\graph.pkl","rb")); au=[s for s in g["steps"] if s["au"]]
    teach=pickle.load(open(OUT+r"\ft_oof_200.pkl","rb"))   # e5-base au OOF soft labels
    print(f"au {len(au)}, KD teacher=e5-base(ft_oof_200)",flush=True)
    if a.full:
        m=train(tok,au,teach,dev,42)
        m.enc.half().save_pretrained(OUT+r"\e5_ft_kd"); tok.save_pretrained(OUT+r"\e5_ft_kd"); torch.save(m.head.state_dict(),OUT+r"\e5_ft_kd\head.pt")
        print("→ e5_ft_kd/ 저장",flush=True); return
    folds=json.load(open(OUT+r"\folds.json",encoding="utf-8")); preds={}
    from sklearn.metrics import f1_score
    for k in (0,1,2):
        t0=time.time(); tr=[s for s in au if folds[s["id"]]!=k]; va=[s for s in au if folds[s["id"]]==k]
        m=train(tok,tr,teach,dev,42); preds.update(predict(tok,m,va,dev))
        acc=np.mean([ACTIONS[preds[s["id"]].argmax()]==s["label"] for s in va]); print(f"fold{k}: acc {acc:.4f} ({time.time()-t0:.0f}s)",flush=True); del m; torch.cuda.empty_cache()
    pickle.dump(preds,open(OUT+r"\ft_oof_kd.pkl","wb")); print("→ ft_oof_kd.pkl",flush=True)


if __name__=="__main__": main()
