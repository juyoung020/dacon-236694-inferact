# -*- coding: utf-8 -*-
"""[2]+[3] 계층 분류기 학습 + [4] 오프셋 튜닝 + OOF 채점.

사용: python train_hier.py            → 5-fold OOF 전체 + 오프셋 + 리포트
      python train_hier.py --fold 0  → fold 0만 (스모크)
      python train_hier.py --full    → 전체 train으로 최종 아티팩트 저장
리포트 시나리오 두 가지(같은 실행에서 산출, 비교실험 아님):
  A = 세션매칭 포함(테스트에 세션 겹침이 있을 때) / B = 세션매칭 제외(겹침 없을 때, 보수적)
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
def _p(rel):  # 저장소 루트 기준 경로
    return _os.path.join(_ROOT, *[x for x in rel.split('\\') if x])
import argparse, json, pickle, sys, time, os
import numpy as np
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from scipy.sparse import hstack
from build_graph import ACTIONS, GROUPS, GROUP_LIST
from retrieval import Retrieval
from features import FeatureBuilder, L1Group
from rules import mine_rules, compile_rules, apply_rules

OUT = Path(_p(r"work\onto"))
A_IDX = {a: i for i, a in enumerate(ACTIONS)}
GCLS = {g: [a for a in ACTIONS if GROUPS[a] == g] for g in GROUP_LIST}

# 데이터 정제(env 게이트, 기본 off=챔피언 불변): sim만 노이즈 라벨 제거, au는 전량 보존.
# clean_sim_drop.json = oof.pkl에서 (sim, EXEC제외, 정라우팅, 오분류&p_true<0.25)로 뽑은 id들.
CLEAN_SIM = os.environ.get("CLEAN_SIM") == "1"
DROP_SIM = (set(json.load(open(OUT / "clean_sim_drop.json", encoding="utf-8")))
            if (CLEAN_SIM and (OUT / "clean_sim_drop.json").exists()) else set())
if CLEAN_SIM:
    print(f"  [CLEAN_SIM] sim 노이즈 {len(DROP_SIM)}행 L2 학습에서 제외 (au 전량 보존)")

# 구조 실측 (07-04, fold0): 현행 계층 0.7471 > 이진 캐스케이드 0.7444 > 평탄 0.7436
# → 그룹까지 나누고 그 아래는 안 나누는 현행 깊이가 최적. 캐스케이드/평탄 재검토 금지

EMB = None
def load_emb():
    """얼린 e5 임베딩 캐시 (있으면 사용, 없으면 임베딩 블록 생략)"""
    global EMB
    if EMB is None and (OUT / "emb_cache.npz").exists():
        z = np.load(OUT / "emb_cache.npz", allow_pickle=True)
        EMB = dict(zip(z["texts"].tolist(), z["emb"]))
        print(f"  emb_cache 로드: {len(EMB):,}문장")
    return EMB

def fit_model(tr_steps, events):
    t0 = time.time()
    emb = load_emb()
    rules = compile_rules(mine_rules(tr_steps))
    retr = Retrieval().fit(tr_steps, events, emb=emb)
    fb = FeatureBuilder().fit(tr_steps, retr, emb=emb)
    Xtr = fb.transform(tr_steps)
    print(f"  피처 {Xtr.shape} ({time.time()-t0:.0f}s)")
    l1 = L1Group().fit(tr_steps, [GROUPS[s["label"]] for s in tr_steps])   # SVC + 프롬프트 + l1_seq(순서/args)
    # 소프트 라벨(LDL) — au 한정 게이트: au 행만 시그니처 LOO 경험분포를 타깃에 0.5 블렌드.
    # sim(93%)은 하드 라벨 유지 — 전 행 적용 시 sim이 프라이어로 뭉개져 -0.006 실측(07-04)
    from collections import Counter as _C, defaultdict as _dd
    sigC = _dd(_C)
    for s in tr_steps:
        if s["au"]:
            sigC[retr._sig_of(s["prompt"])][s["label"]] += 1
    # 현행 계층 (v6 챔피언 구성). LDL kNN 백오프는 fold0 0.7439(-0.0032)로 기각(07-04)
    l2 = {}
    for g in GROUP_LIST:
        idx = [i for i, s in enumerate(tr_steps) if GROUPS[s["label"]] == g]
        rows, ys, ws = [], [], []
        for i in idx:
            s = tr_steps[i]
            if not s["au"]:
                if not (CLEAN_SIM and s["id"] in DROP_SIM):   # sim 노이즈면 스킵
                    rows.append(i); ys.append(s["label"]); ws.append(1.0)
                continue
            if os.environ.get("DISABLE_LDL") == "1":       # 다이어트: LDL 소프트라벨 끔 (au 하드라벨, 가중3)
                rows.append(i); ys.append(s["label"]); ws.append(3.0)
                continue
            c = dict(sigC[retr._sig_of(s["prompt"])])
            c[s["label"]] = c.get(s["label"], 0) - 1             # leave-one-out
            c = {a: v for a, v in c.items() if v > 0 and GROUPS[a] == g}
            tot = sum(c.values())
            mix = {s["label"]: 0.5} if tot >= 2 else {s["label"]: 1.0}
            if tot >= 2:
                for a, v in c.items():
                    mix[a] = mix.get(a, 0.0) + 0.5 * v / tot
            for a, w in mix.items():
                rows.append(i); ys.append(a); ws.append(3.0 * w)  # au 가중 3배
        clf = HistGradientBoostingClassifier(max_iter=int(os.environ.get("L2_ITER", "800")), learning_rate=0.05,
                                             max_leaf_nodes=int(os.environ.get("L2_LEAF", "127")),
                                             early_stopping=True, random_state=42, class_weight="balanced")
        clf.fit(Xtr[rows], np.array(ys), sample_weight=np.array(ws))
        l2[g] = clf
    # 혼동쌍 스페셜리스트 제거(07-06): held-out -0.0005 브리틀 수동개입 → 코드 다이어트
    print(f"  L1+L2 학습 완료 ({time.time()-t0:.0f}s)")
    return {"rules": rules, "retr": retr, "fb": fb, "l1": l1, "l2": l2}

def predict_probs(model, steps):
    """하드라우팅 14-클래스 확률 (n,14): L1이 고른 그룹 g*의 L2만 채우고 나머지 그룹은 0.
    소프트곱셈(P(g)×P(a|g)) 대신 g* 하드 선택 — L1 확률 오보정 전파 차단. non-g*=0이 그룹마스크 겸함."""
    X = model["fb"].transform(steps)
    gstar = model["l1"].predict_group(steps)           # 그룹 라벨 (n,)
    P = np.zeros((len(steps), len(ACTIONS)))
    for g in GROUP_LIST:
        rows = np.where(gstar == g)[0]
        if len(rows) == 0:
            continue
        Pw = model["l2"][g].predict_proba(X[rows])
        for ci, cls in enumerate(model["l2"][g].classes_):
            P[rows, A_IDX[cls]] = Pw[:, ci]
    return P


def final_preds(rl, P, offsets):
    """우선순위 적용: rl(고정밀 룰) → argmax(logP+offset). (세션매칭 층 제거 — 테스트 무발동으로 확인됨)"""
    logP = np.log(P + 1e-9) + offsets
    am = logP.argmax(axis=1)
    return [rl[i] if rl[i] is not None else ACTIONS[am[i]] for i in range(len(P))]

def tune_offsets(rl, P, y):
    """클래스별 로짓 오프셋 coordinate ascent"""
    off = np.zeros(len(ACTIONS))
    best = f1_score(y, final_preds(rl, P, off), average="macro")
    for _ in range(2):
        for c in range(len(ACTIONS)):
            for v in np.arange(-0.8, 0.81, 0.1):
                trial = off.copy(); trial[c] = v
                f = f1_score(y, final_preds(rl, P, trial), average="macro")
                if f > best:
                    best, off = f, trial
    return off, best

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    g = pickle.load(open(OUT / "graph.pkl", "rb"))
    steps, events = g["steps"], g["events"]
    folds = json.load(open(OUT / "folds.json", encoding="utf-8"))

    if args.full:
        model = fit_model(steps, events)
        off = np.load(OUT / "offsets.npy") if (OUT / "offsets.npy").exists() else np.zeros(len(ACTIONS))
        joblib.dump({"rules": model["rules"], "retr": model["retr"], "fb": model["fb"],
                     "l1": model["l1"], "l2": model["l2"], "offsets": off},
                    OUT / "bundle.joblib", compress=3)
        print(f"→ bundle.joblib 저장 ({(OUT/'bundle.joblib').stat().st_size/1e6:.0f}MB)")
        return

    run_folds = [args.fold] if args.fold >= 0 else range(3)
    oof = {}
    for k in run_folds:
        print(f"[fold {k}]")
        tr = [s for s in steps if folds[s["id"]] != k]
        va = [s for s in steps if folds[s["id"]] == k]
        model = fit_model(tr, events)
        rl = [apply_rules(s["prompt"], model["rules"]) for s in va]
        P = predict_probs(model, va)
        for i, s in enumerate(va):
            oof[s["id"]] = (rl[i], P[i], s["label"])
        y = [s["label"] for s in va]
        b = f1_score(y, final_preds(rl, P, np.zeros(14)), average="macro")
        print(f"  fold{k} Macro-F1 (개입 전) = {b:.4f}")

    if args.fold < 0:
        ids = list(oof)
        rl = [oof[i][0] for i in ids]
        P = np.vstack([oof[i][1] for i in ids]); y = [oof[i][2] for i in ids]
        au_arr = np.array([1 if i.startswith("sess_au") else 0 for i in ids])
        with open(OUT / "oof.pkl", "wb") as f:
            pickle.dump({"ids": ids, "rl": rl, "P": P, "y": y, "info": {}}, f)
        off, bestB = tune_offsets(rl, P, y)      # 스페셜리스트 개입 제거(07-06) → base P 직접 오프셋 튜닝
        np.save(OUT / "offsets.npy", off)
        pred = final_preds(rl, P, off)
        fB = f1_score(y, pred, average="macro")
        au_m = au_arr.astype(bool); yB = np.array(y); pB = np.array(pred)
        print("\n===== OOF 최종 =====")
        print(f"Macro-F1 = {fB:.4f}  (au {f1_score(yB[au_m], pB[au_m], average='macro'):.4f}"
              f" / sim {f1_score(yB[~au_m], pB[~au_m], average='macro'):.4f})")
        for a_, f_ in zip(ACTIONS, f1_score(y, pred, average=None, labels=ACTIONS)):
            print(f"  {a_:20} {f_:.3f}")
        with open(OUT / "report.md", "a", encoding="utf-8") as f:
            f.write(f"\nOOF: {fB:.4f} offsets={np.round(off,2).tolist()}\n")

if __name__ == "__main__":
    main()
