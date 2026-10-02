# -*- coding: utf-8 -*-
"""풀 파인튜닝 (full-context, 전 14클래스, e5 전파라미터). 소수클래스 메꾸기 목적, 과적합 OK.
class_weight로 희소클래스 recall 부양. OOF fold별 학습→held-out 예측 → v6와 블렌드.
사용: python ft_full.py --fold 0   → fold0 스크리닝
      python ft_full.py             → 3-fold OOF → ft_full_oof.pkl
      python ft_full.py --full      → 전체 학습 → e5_full/ 저장"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
def _p(rel):  # 저장소 루트 기준 경로
    return _os.path.join(_ROOT, *[x for x in rel.split('\\') if x])
import argparse, os, pickle, json, sys, time, random
import numpy as np
OUT = _p(r"work\onto")
os.environ.setdefault("HF_HOME", OUT + r"\hf_cache"); os.environ["HF_HUB_OFFLINE"] = "1"
import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel
sys.path.insert(0, OUT)
from build_graph import ACTIONS, GROUPS
from sklearn.metrics import f1_score

AI = {a: i for i, a in enumerate(ACTIONS)}
NAME = os.environ.get("FT_MODEL", "dragonkue/multilingual-e5-small-ko-v2")
EPOCHS = int(os.environ.get("FT_EPOCHS", "4")); LR = float(os.environ.get("FT_LR", "3e-5"))
BATCH = int(os.environ.get("FT_BATCH", "32")); MAXLEN = int(os.environ.get("FT_MAXLEN", "320"))
PBATCH = int(os.environ.get("FT_PBATCH", "128"))
HU, HA, HR = int(os.environ.get("FT_HU", "3")), int(os.environ.get("FT_HA", "4")), int(os.environ.get("FT_HR", "2"))
SEED = int(os.environ.get("FT_SEED", "42"))   # 시드배깅/수프용 (기본 42=챔피언)

def seed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

def ctx(s):
    us=[c for t,c,r in s["hist"] if t=="U"][-HU:]; ac=[c for t,c,r in s["hist"] if t=="A"][-HA:]
    rs=[r for t,c,r in s["hist"] if t=="A" and r][-HR:]
    return (s["prompt"]+" || U "+" ".join(us)+" || A "+" ".join(ac)+" || R "+" ".join(rs)
            +f" || {s['tier']} {s['ci']} {s['lang']} loc{int(s['loc']>2000)} bud{int(s['budget']>100000)}")

class M(nn.Module):
    def __init__(s): super().__init__(); s.enc=AutoModel.from_pretrained(NAME); s.head=nn.Linear(s.enc.config.hidden_size,14)
    def forward(s,**e):
        h=s.enc(**e).last_hidden_state; m=e["attention_mask"].unsqueeze(-1)
        return s.head((h*m).sum(1)/m.sum(1))

GAMMA=float(os.environ.get("FT_FOCAL","0"))   # >0이면 focal loss(하드예제 집중), 0이면 챔피언 CE

def train(texts, rows, cw, dev, tok):
    seed(); model=M().to(dev); opt=torch.optim.AdamW(model.parameters(),lr=LR)
    cwt=torch.tensor(cw,dtype=torch.float32).to(dev); lf=nn.CrossEntropyLoss(weight=cwt)
    steps_total=EPOCHS*((len(rows)+BATCH-1)//BATCH)
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=LR,total_steps=steps_total,pct_start=0.1)
    model.train()
    for ep in range(EPOCHS):
        order=list(rows); random.Random(SEED+ep).shuffle(order)
        for lo in range(0,len(order),BATCH):
            idx=order[lo:lo+BATCH]
            e=tok(["query: "+texts[i] for i in idx],padding=True,truncation=True,max_length=MAXLEN,return_tensors="pt").to(dev)
            yb=torch.tensor([AI[Y[i]] for i in idx]).to(dev)
            opt.zero_grad(); logits=model(**e)
            if GAMMA>0:
                logp=torch.log_softmax(logits,1)
                ce=nn.functional.nll_loss(logp,yb,weight=cwt,reduction="none")
                pt=logp.gather(1,yb[:,None]).squeeze(1).exp()
                loss=((1-pt)**GAMMA*ce).mean()
            else:
                loss=lf(logits,yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sch.step()
    return model

@torch.no_grad()
def predict(model, texts, rows, dev, tok):
    model.eval(); out={}
    for lo in range(0,len(rows),PBATCH):
        idx=rows[lo:lo+PBATCH]
        e=tok(["query: "+texts[i] for i in idx],padding=True,truncation=True,max_length=MAXLEN,return_tensors="pt").to(dev)
        p=torch.softmax(model(**e),1).cpu().numpy()
        for r,i in enumerate(idx): out[i]=p[r]
    return out

g=pickle.load(open(OUT+r"\graph.pkl","rb")); steps=g["steps"]
Y=[s["label"] for s in steps]; TEXTS=[ctx(s) for s in steps]
folds=json.load(open(OUT+r"\folds.json",encoding="utf-8")); FK=np.array([folds[s["id"]] for s in steps])
cnt=np.bincount([AI[a] for a in Y],minlength=14); CW=(cnt.sum()/np.maximum(cnt,1))**0.5   # 희소 부양(완화)
WEAKBOOST=float(os.environ.get("FT_WEAKBOOST","1.0"))   # 약한클래스 가중 배수(1.0=챔피언)
if WEAKBOOST!=1.0:
    for a in ["ask_user","plan_task","lint_or_typecheck","web_search"]: CW[AI[a]]*=WEAKBOOST

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap=argparse.ArgumentParser(); ap.add_argument("--fold",type=int,default=-1); ap.add_argument("--full",action="store_true")
    a=ap.parse_args(); dev="cuda" if torch.cuda.is_available() else "cpu"
    tok=AutoTokenizer.from_pretrained(NAME)
    print(f"device={dev}, epochs={EPOCHS}, class_weight=sqrt-balanced, FOCAL_GAMMA={GAMMA}, WEAKBOOST={WEAKBOOST}")
    if a.full:
        t0=time.time(); m=train(TEXTS,list(range(len(steps))),CW,dev,tok)
        m.enc.half().save_pretrained(OUT+r"\e5_full"); tok.save_pretrained(OUT+r"\e5_full")
        torch.save(m.head.state_dict(),OUT+r"\e5_full\head.pt")
        json.dump({"maxlen":MAXLEN,"hu":HU,"ha":HA,"hr":HR,"model":NAME,"dim":m.enc.config.hidden_size},
                  open(OUT+r"\e5_full\config_ft.json","w"))   # 패키징이 config-agnostic하게 (pack.py가 읽음)
        print(f"--full 저장 {time.time()-t0:.0f}s → e5_full/ (maxlen{MAXLEN} h{HU}/{HA}/{HR} dim{m.enc.config.hidden_size})"); return
    fl=[a.fold] if a.fold>=0 else [0,1,2]
    save_path=os.environ.get("FT_OOF_PATH", OUT+r"\ft_full_oof.pkl"); id2i={s["id"]:i for i,s in enumerate(steps)}
    oof={}; done=set()
    if a.fold<0 and os.path.exists(save_path):    # 재개: 이미 예측한 fold 건너뜀
        for id_,p in pickle.load(open(save_path,"rb")).items():
            oof[id2i[id_]]=p; done.add(id_)
        print(f"재개: {len(done)}행 이미 저장됨")
    for k in fl:
        t0=time.time(); tr=[i for i in range(len(steps)) if FK[i]!=k]; va=[i for i in range(len(steps)) if FK[i]==k]
        if a.fold<0 and all(steps[i]["id"] in done for i in va):
            print(f"fold{k}: 이미 완료, 건너뜀"); continue
        m=train(TEXTS,tr,CW,dev,tok); oof.update(predict(m,TEXTS,va,dev,tok))
        yv=[Y[i] for i in va]; pv=[ACTIONS[oof[i].argmax()] for i in va]
        print(f"fold{k}: ft 단독 Macro-F1 {f1_score(yv,pv,average='macro'):.4f} ({time.time()-t0:.0f}s)")
        yva=np.array(yv); pva=np.array(pv)
        for wa in ["ask_user","plan_task","lint_or_typecheck","web_search"]:
            f1c=f1_score(yva==wa,pva==wa); rc=(pva[yva==wa]==wa).mean()
            print(f"    {wa:18} F1 {f1c:.3f}  recall {rc:.3f}")
        del m; torch.cuda.empty_cache()
        if a.fold<0:                              # fold마다 증분 저장 (중단 대비)
            pickle.dump({steps[i]["id"]:oof[i] for i in oof}, open(save_path,"wb"))
            print(f"  → 증분 저장 ({len(oof)}행)")
    if a.fold<0:
        print("→ ft_full_oof.pkl 완료")
    # v6와 블렌드 평가 (해당 fold만)
    d=pickle.load(open(OUT+r"\oof_hgb.pkl","rb")); ids2,rl,P,y2=d["ids"],d["rl"],np.asarray(d["P"]),d["y"]
    ftau=pickle.load(open(OUT+r"\ft_oof.pkl","rb")); cfg=json.load(open(OUT+r"\ft_w.json")); w,off=cfg["w"],np.array(cfg["offsets"])
    id2i={s["id"]:i for i,s in enumerate(steps)}
    evalids=[id_ for id_ in ids2 if id2i[id_] in oof]
    Qb=np.zeros((len(evalids),14)); yy=[]; au=[]
    for j,id_ in enumerate(evalids):
        i=list(ids2).index(id_); q=P[i].copy()
        if id_ in ftau: q=(1-w)*P[i]+w*ftau[id_]
        Qb[j]=np.exp(np.log(q+1e-9)+off); Qb[j]/=Qb[j].sum(); yy.append(y2[i]); au.append(id_.startswith("sess_au"))
    au=np.array(au); yy=np.array(yy)
    F=np.array([oof[id2i[id_]] for id_ in evalids])
    rlmap={id_:rl[list(ids2).index(id_)] for id_ in evalids}
    print("\n=== v6 vs v6⊕풀ft 블렌드 (해당 fold) ===")
    for bw in (0.0,0.2,0.3,0.4,0.5):
        Z=(1-bw)*Qb+bw*F; am=Z.argmax(1)
        pb=np.array([rlmap[id_] if rlmap[id_] is not None else ACTIONS[am[j]] for j,id_ in enumerate(evalids)])
        fa=f1_score(yy,pb,average="macro")
        print(f"  w={bw}: Macro-F1 {fa:.4f} (au {f1_score(yy[au],pb[au],average='macro'):.4f}/sim {f1_score(yy[~au],pb[~au],average='macro'):.4f})")

if __name__=="__main__": main()
