# -*- coding: utf-8 -*-
"""학습 전 검증: 시그니처×메타 조건부 테이블의 지지도/누수차단/차원 점검 (학습 아님, 집계만)"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
def _p(rel):  # 저장소 루트 기준 경로
    return _os.path.join(_ROOT, *[x for x in rel.split('\\') if x])
import pickle, sys
import numpy as np
sys.path.insert(0, _p(r"work\onto"))
from build_graph import ACTIONS
from retrieval import Retrieval

sys.stdout.reconfigure(encoding="utf-8")
g = pickle.load(open(_p(r"work\onto\graph.pkl"), "rb"))
steps, events = g["steps"], g["events"]

r = Retrieval()
# kNN/NN 학습 없이 테이블만 검증하기 위해 fit의 테이블 파트만 수동 재현 대신 fit 호출(수 분) 회피:
# -> fit은 NN 인덱스도 만들지만 이는 결정적 인덱싱(모델 학습 아님). 전체 fit 실행.
r.fit(steps, events)

feat = r.template_feats(steps[0])
print(f"template_feats 차원: {len(feat)} (기대 75)")

# 지지도: sigm 조회가 실제로 걸리는 행 비율 (세션 차감 후 지지도>0)
hit = tot = 0
neg = 0
for s in steps[:20000]:
    gsig = r._sig_of(s["prompt"])
    for k in ("loc", "budget"):
        b = int(np.searchsorted(r.qb[k], s[k]))
        c = r.sigm.get((s["au"], gsig, k, b))
        if c:
            from retrieval import _cvec
            v = _cvec(c) - _cvec(r.sigm_sess.get((s["sess"], gsig, k, b), {}))
            if (v < 0).any():
                neg += 1
            if v.sum() > 0:
                hit += 1
        tot += 1
print(f"시그니처×메타 조회 적중(자기세션 차감 후): {hit}/{tot} ({hit/tot:.1%}), 음수 발생 {neg}건 (기대 0)")
print(f"버킷 경계 loc={np.round(r.qb['loc']).astype(int).tolist()}, budget={np.round(r.qb['budget']).astype(int).tolist()}")
