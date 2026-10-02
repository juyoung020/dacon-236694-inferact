# -*- coding: utf-8 -*-
"""[3] RAG 계층: 템플릿 조회표 + char TF-IDF kNN.

- 인덱스 = 학습 스텝 (prompt, label) + observed_event (prompt, action)
- 누수 차단: 조회·kNN 모두 질의 스텝과 같은 세션의 기여분을 제외 (테스트 상황과 동일 조건)
- lazy kNN: 템플릿 완전일치 지지도 ≥3이면 kNN 생략(분포가 사실상 동일) → 서버 시간 절약
"""
import re
import numpy as np
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from build_graph import ACTIONS, GROUPS, GROUP_LIST, norm

A_IDX = {a: i for i, a in enumerate(ACTIONS)}
G_OF = np.array([GROUP_LIST.index(GROUPS[a]) for a in ACTIONS])

_EXT = r'(py|ts|tsx|js|jsx|vue|java|kt|rs|go|sql|yaml|yml|json|toml|md|css|html|sh|lock|txt|ipynb|swift)'

def template_sig(t):
    """프롬프트 → 템플릿 시그니처 (파일경로/식별자/숫자를 슬롯으로 치환)"""
    t = t.lower()
    t = re.sub(r'[\w./-]+\.' + _EXT + r'\b', '<FILE>', t)
    t = re.sub(r'[a-z_][a-z0-9_]*\([^)]*\)', '<FN>', t)
    t = re.sub(r'\d+', '<N>', t)
    t = re.sub(r'[a-zA-Z_][a-zA-Z0-9_]{3,}',
               lambda m: '<ID>' if re.search(r'[A-Z_]', m.group()) or '_' in m.group() else m.group(), t)
    return norm(t)

def _cvec(counter):
    v = np.zeros(len(ACTIONS))
    for a, c in counter.items():
        v[A_IDX[a]] += c
    return v

class Retrieval:
    def __init__(self, k=50, extra=15):
        self.k, self.extra = k, extra
        # 주의: k100/가중² 시도는 부품 단독 +0.012였으나 파이프라인 -0.002로 기각(07-04) —
        # HGB는 불확실성이 보존된 부드러운 히스토그램에서 더 잘 배움. k_type/pw_type 미설정 = 전 타입 k50/가중¹
        self.k_type = None
        self.pw_type = None

    def fit(self, steps, events_by_sess, emb=None):
        # 조회·검색은 전부 세션 타입(au/sim) 조건부 — sim의 뒤섞인 통계가 au(의미정합)를 오염시키지 않게 분리
        texts, labels, sess, typ = [], [], [], []
        for s in steps:
            texts.append(s["prompt"]); labels.append(s["label"]); sess.append(s["sess"]); typ.append(s["au"])
        au_of_sess = {s["sess"]: s["au"] for s in steps}
        # 순회 순서가 kNN 동거리 tie-break에 영향(-0.004 실측). PYTHONHASHSEED=0으로 고정
        # (reproduce.ps1 런처에서 설정). sorted()는 다른 순서라 챔피언과 불일치 → 사용 안 함
        for sid in {s["sess"] for s in steps}:
            for ev in events_by_sess.get(sid, []):
                p, a = ev[0], ev[1]                       # (발화, 액션, 결과, 확증) 튜플
                texts.append(p); labels.append(a); sess.append(sid); typ.append(au_of_sess[sid])
        self.idx_labels = np.array([A_IDX[l] for l in labels], dtype=np.int32)
        self.idx_sess = np.array(sess)
        self.idx_au = np.array(typ, dtype=np.int8)
        # 템플릿 조회표 (타입 조건부) + 세션별 기여분(누수 차단용)
        self.tpl = defaultdict(Counter); self.tpl_sess = defaultdict(Counter)
        for t, l, sid, a_ in zip(texts, labels, sess, typ):
            self.tpl[(a_, t)][l] += 1; self.tpl_sess[(sid, t)][l] += 1
        self.tplc = defaultdict(Counter); self.tplc_sess = defaultdict(Counter)
        for s in steps:
            key = (s["au"], s["prompt"], s["tier"], s["ci"])
            self.tplc[key][s["label"]] += 1
            self.tplc_sess[(s["sess"],) + key[1:]][s["label"]] += 1
        # 템플릿 시그니처 계층 (타입 조건부, 파일명/식별자 슬롯 치환)
        self.sig = defaultdict(Counter); self.sig_sess = defaultdict(Counter)
        self._sig_cache = {}
        for t, l, sid, a_ in zip(texts, labels, sess, typ):
            g = self._sig_of(t)
            self.sig[(a_, g)][l] += 1; self.sig_sess[(sid, g)][l] += 1
        # 시그니처 × loc/budget 구간 조건부 (실측: loc·budget이 plan↔ask 등을 69% 일관성으로 기울임)
        # 중앙값 2버킷 — 신호가 이진 방향성(크다/작다)이고 4분위는 지지도 희박(6.4%)해서
        self.qb = {k: np.quantile([s[k] for s in steps], [0.5]) for k in ("loc", "budget")}
        self.sigm = defaultdict(Counter); self.sigm_sess = defaultdict(Counter)
        for s in steps:
            g = self._sig_of(s["prompt"])
            for k in ("loc", "budget"):
                b = int(np.searchsorted(self.qb[k], s[k]))
                self.sigm[(s["au"], g, k, b)][s["label"]] += 1
                self.sigm_sess[(s["sess"], g, k, b)][s["label"]] += 1
        self.tpl, self.tpl_sess = dict(self.tpl), dict(self.tpl_sess)
        self.tplc, self.tplc_sess = dict(self.tplc), dict(self.tplc_sess)
        self.sig, self.sig_sess = dict(self.sig), dict(self.sig_sess)
        self.sigm, self.sigm_sess = dict(self.sigm), dict(self.sigm_sess)
        # kNN 인덱스 — 타입별 독립 (au 질의는 au 기출에서만 검색)
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, sublinear_tf=True)
        X = self.vec.fit_transform(texts)
        self.sub = {}
        for a_ in (0, 1):
            m = self.idx_au == a_
            if m.sum() == 0:
                continue
            kk = self.k_type.get(a_, self.k) if getattr(self, "k_type", None) else self.k
            self.sub[a_] = {
                "labels": self.idx_labels[m], "sess": self.idx_sess[m],
                "nn": NearestNeighbors(n_neighbors=min(kk + self.extra, int(m.sum())),
                                       metric="cosine", algorithm="brute", n_jobs=-1).fit(X[m]),
            }
        # 임베딩 kNN 인덱스 (얼린 e5) — au만 구축: emb_knn_feats_batch(au_only=True)가 sim을
        # 절대 조회하지 않아 sim 인덱스는 번들 ~90MB 죽은 짐이었음 (코드리뷰 B, 07-04)
        self.esub = {}
        if emb is not None:
            Z = np.zeros(384, dtype=np.float16)
            for a_ in (1,):
                m = self.idx_au == a_
                if m.sum() == 0:
                    continue
                V = np.array([emb.get(texts[j], Z) for j in np.where(m)[0]], dtype=np.float32)
                self.esub[a_] = {
                    "labels": self.idx_labels[m], "sess": self.idx_sess[m],
                    "nn": NearestNeighbors(n_neighbors=min(self.k + self.extra, int(m.sum())),
                                           metric="cosine", algorithm="brute", n_jobs=-1).fit(V),
                }
        return self

    def _sig_of(self, text):
        s = self._sig_cache.get(text)
        if s is None:
            s = template_sig(text)
            self._sig_cache[text] = s
        return s

    def _tpl_counts(self, step):
        a_ = step["au"]
        c = _cvec(self.tpl.get((a_, step["prompt"]), {})) - _cvec(self.tpl_sess.get((step["sess"], step["prompt"]), {}))
        key = (a_, step["prompt"], step["tier"], step["ci"])
        cc = _cvec(self.tplc.get(key, {})) - _cvec(self.tplc_sess.get((step["sess"],) + key[1:], {}))
        return np.maximum(c, 0), np.maximum(cc, 0)

    def template_feats(self, step):
        c, cc = self._tpl_counts(step)
        g = self._sig_of(step["prompt"])
        cs = _cvec(self.sig.get((step["au"], g), {})) - _cvec(self.sig_sess.get((step["sess"], g), {}))
        cs = np.maximum(cs, 0)
        t, tc, ts = c.sum(), cc.sum(), cs.sum()
        h = c / t if t else c
        hc = cc / tc if tc else cc
        hs = cs / ts if ts else cs
        parts = [h, [np.log1p(t)], hc, [np.log1p(tc)], hs, [np.log1p(ts)]]
        for k in ("loc", "budget"):                      # 시그니처×메타 조건부 (+30차원)
            b = int(np.searchsorted(self.qb[k], step[k]))
            cm = _cvec(self.sigm.get((step["au"], g, k, b), {})) - _cvec(self.sigm_sess.get((step["sess"], g, k, b), {}))
            cm = np.maximum(cm, 0)
            tm = cm.sum()
            parts += [cm / tm if tm else cm, [np.log1p(tm)]]
        return np.concatenate(parts)                     # 75차원

    def knn_feats_batch(self, steps, lazy_min=3, batch=4000):
        """35차원 × n (히스토그램 21 + 그룹내 재정규화 14). 템플릿 지지도 ≥lazy_min이면 kNN 생략."""
        n = len(steps)
        out = np.zeros((n, len(ACTIONS) * 2 + len(GROUP_LIST) + 3))
        need = []
        for i, s in enumerate(steps):
            c, _ = self._tpl_counts(s)
            t = c.sum()
            if t >= lazy_min:
                lh = c / t
                gh = np.zeros(len(GROUP_LIST)); np.add.at(gh, G_OF, lh)
                out[i] = np.concatenate([lh, gh, [1.0, 1.0, 1.0], self._within_group_norm(lh)])
            else:
                need.append(i)
        self._knn_fill(out, need, steps, self.sub, lambda ids: self.vec.transform([steps[i]["prompt"] for i in ids]),
                       k_type=getattr(self, "k_type", None), pw_type=getattr(self, "pw_type", None))
        return out

    def emb_knn_feats_batch(self, steps, emb, lazy_min=3, batch=4000, au_only=True):
        """임베딩 kNN 35차원 × n. au 행에만 적용(비대칭 게이트) — sim은 의미가 노이즈라 0 유지.
        lazy 규칙은 char kNN과 동일."""
        n = len(steps)
        out = np.zeros((n, len(ACTIONS) * 2 + len(GROUP_LIST) + 3))
        need = []
        for i, s in enumerate(steps):
            if au_only and not s["au"]:
                continue
            c, _ = self._tpl_counts(s)
            t = c.sum()
            if t >= lazy_min:
                lh = c / t
                gh = np.zeros(len(GROUP_LIST)); np.add.at(gh, G_OF, lh)
                out[i] = np.concatenate([lh, gh, [1.0, 1.0, 1.0], self._within_group_norm(lh)])
            else:
                need.append(i)
        Z = np.zeros(384, dtype=np.float16)
        self._knn_fill(out, need, steps, self.esub,
                       lambda ids: np.array([emb.get(steps[i]["prompt"], Z) for i in ids], dtype=np.float32),
                       batch=batch)
        return out

    @staticmethod
    def _within_group_norm(lh):
        """라벨 히스토그램을 그룹 안에서 재정규화 → P(클래스 | 이웃, 그룹) 명시 공급 (14차원)"""
        wg = np.zeros(len(ACTIONS))
        for gi in range(len(GROUP_LIST)):
            m = G_OF == gi
            s = lh[m].sum()
            if s > 0:
                wg[m] = lh[m] / s
        return wg

    def _knn_fill(self, out, need, steps, subs, qbuild, batch=4000, k_type=None, pw_type=None):
        for a_ in (0, 1):
            grp = [i for i in need if steps[i]["au"] == a_]
            if not grp or a_ not in subs:
                continue
            sub = subs[a_]
            kk = (k_type or {}).get(a_, self.k)
            pw = (pw_type or {}).get(a_, 1.0)
            for lo in range(0, len(grp), batch):
                ids = grp[lo:lo + batch]
                Q = qbuild(ids)
                dist, nb = sub["nn"].kneighbors(Q)
                sim = 1.0 - dist
                for r, i in enumerate(ids):
                    mask = sub["sess"][nb[r]] != steps[i]["sess"]          # 같은 세션 이웃 제외
                    nbr, w = nb[r][mask][:kk], np.maximum(sim[r][mask][:kk], 0) ** pw
                    lh = np.zeros(len(ACTIONS)); np.add.at(lh, sub["labels"][nbr], w)
                    if lh.sum(): lh /= lh.sum()
                    gh = np.zeros(len(GROUP_LIST)); np.add.at(gh, G_OF[sub["labels"][nbr]], w)
                    if gh.sum(): gh /= gh.sum()
                    srt = np.sort(w)[::-1] if len(w) else np.zeros(1)
                    out[i] = np.concatenate([lh, gh, [srt[0], srt[:5].mean(), srt.mean()],
                                             self._within_group_norm(lh)])
