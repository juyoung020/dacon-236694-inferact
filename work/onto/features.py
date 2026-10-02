# -*- coding: utf-8 -*-
"""[3] L2 피처 조립 — train/test 공용.

블록: RAG(템플릿 30 + kNN 21) / 순차 맥락(전이확률 인코딩 포함 ~69)
      / 상황 메타(~30) / 텍스트(SVD 128 + 필러 28)
"""
import os
import numpy as np
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from scipy.sparse import hstack, csr_matrix
from build_graph import ACTIONS, GROUPS, GROUP_LIST

class L1Group:
    """[2] 그룹 분류기 (4클래스, 하드라우팅). SVC(C=1) + 프롬프트 TF-IDF(word+char) + 순서/args(l1_seq 67).
    모듈 레벨 정의 — joblib 직렬화 경로 안정성. LinearSVC는 predict_proba 없음 → argmax 그룹만 제공(하드라우팅 전용)."""
    def _X(self, steps, fit=False):
        prompts = [s["prompt"] for s in steps]
        seq = csr_matrix(np.array([s["l1_seq"] for s in steps], dtype=float))
        tf = (self.w.fit_transform if fit else self.w.transform)(prompts)
        cf = (self.c.fit_transform if fit else self.c.transform)(prompts)
        return hstack([tf, cf, seq]).tocsr()

    def fit(self, steps, groups):
        self.w = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
        self.c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, sublinear_tf=True)
        self.clf = LinearSVC(C=1, class_weight="balanced", max_iter=3000)
        self.clf.fit(self._X(steps, fit=True), groups)
        return self

    def predict_group(self, steps):
        return self.clf.predict(self._X(steps))     # 그룹 라벨 (n,) — 하드라우팅용 argmax

A_IDX = {a: i for i, a in enumerate(ACTIONS)}
G_IDX = {g: i for i, g in enumerate(GROUP_LIST)}
TIERS = ["free", "pro", "enterprise"]
LANGS = ["ko", "en", "mixed"]
CIS = ["passed", "failed", "none"]
MIXES = ["py", "ts", "tsx", "vue", "java", "go", "rs", "yaml", "sql", "js"]
FILLERS = ["여기부터", "이번 것만", "천천히", "가볍게", "꼼꼼히", "간단히", "빨리", "시간 될 때", "시간 되실 때",
           "한 번만", "한번 더", "먼저", "우선", "지금", "asap", "please", "plz", "thx", "thanks", "cheers",
           "ㅠ", "ㅎㅎ", "ㅋㅋ", "no pressure", "when you can", "when free", "real quick", "sorry to bug"]

def hist_actions(step):
    acts = [a for t, a, _ in step["hist"] if t == "A"]
    return acts

def last_result_status(step):
    res = next((r for t, _, r in reversed(step["hist"]) if t == "A"), "")
    if res.startswith(("PASS", "ok")): return 0
    if res.startswith("FAIL"): return 1
    if res.startswith("ERROR"): return 2
    return 3

class SeqStats:
    """전이 통계: P(y|직전액션), P(y|직전2경로), P(y|직전액션+결과상태). 라플라스 스무딩."""
    def fit(self, steps):
        self.p1 = defaultdict(Counter); self.p2 = defaultdict(Counter)
        self.pr = defaultdict(Counter); self.prior = Counter()
        for s in steps:
            acts = hist_actions(s)
            a_ = s["au"]
            last = acts[-1] if acts else "NONE"
            last2 = "|".join(acts[-2:]) if len(acts) >= 2 else "NONE"
            self.p1[(a_, last)][s["label"]] += 1
            self.p2[(a_, last2)][s["label"]] += 1
            self.pr[(a_, last, last_result_status(s))][s["label"]] += 1
            self.prior[s["label"]] += 1
        return self

    def _vec(self, table, key):
        c = table.get(key, self.prior)
        tot = sum(c.values()) + len(ACTIONS)
        return np.array([(c.get(a, 0) + 1) / tot for a in ACTIONS])

    def feats(self, step):
        acts = hist_actions(step)
        a_ = step["au"]
        last = acts[-1] if acts else "NONE"
        last2 = "|".join(acts[-2:]) if len(acts) >= 2 else "NONE"
        return np.concatenate([self._vec(self.p1, (a_, last)), self._vec(self.p2, (a_, last2)),
                               self._vec(self.pr, (a_, last, last_result_status(step)))])   # 42

def onehot(val, cats):
    v = np.zeros(len(cats) + 1)
    v[cats.index(val) if val in cats else len(cats)] = 1
    return v

class FeatureBuilder:
    def fit(self, steps, retr, emb=None):
        self.retr = retr
        self.emb = emb                                       # prompt → 384차원 (얼린 e5), 없으면 None
        self.seq = SeqStats().fit(steps)
        self.q = {k: np.quantile([s[k] for s in steps], [i / 8 for i in range(1, 8)]) for k in ["loc", "budget", "elapsed"]}
        self.svd_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=5, sublinear_tf=True)
        Xt = self.svd_vec.fit_transform([s["prompt"] for s in steps])
        self.svd = TruncatedSVD(n_components=128, random_state=42).fit(Xt)
        return self

    def _bucket(self, x, edges):
        return int(np.searchsorted(edges, x))

    def transform(self, steps):
        knn = self.retr.knn_feats_batch(steps)                       # 35 (히스토그램 21 + 그룹내 재정규화 14)
        svd = self.svd.transform(self.svd_vec.transform([s["prompt"] for s in steps]))  # 128
        if self.emb is not None:
            eknn = self.retr.emb_knn_feats_batch(steps, self.emb)   # 35 (au 행만, sim=0)
            Z = np.zeros(384, dtype=np.float16)
            E = np.array([self.emb.get(s["prompt"], Z) if s["au"] else Z for s in steps], dtype=np.float32)  # 384 (au만)
        rows = []
        for i, s in enumerate(steps):
            acts = hist_actions(s)
            last = acts[-1] if acts else None
            cnt = Counter(acts)
            seq_block = np.concatenate([
                onehot(last, ACTIONS),                               # 15
                onehot(GROUPS.get(last), GROUP_LIST),                # 5
                self.seq.feats(s),                                   # 28
                np.array([cnt.get(a, 0) for a in ACTIONS]),          # 14
                np.eye(4)[last_result_status(s)],                    # 4
                [len(acts), s["step"], s["turn"]],                   # 3
            ])
            meta_block = np.concatenate([
                onehot(s["tier"], TIERS), onehot(s["lang"], LANGS), onehot(s["ci"], CIS),
                onehot(s["mix"], MIXES),
                [s["dirty"], s["open_n"], s["au"]],
                [self._bucket(s["loc"], self.q["loc"]), self._bucket(s["budget"], self.q["budget"]),
                 self._bucket(s["elapsed"], self.q["elapsed"])],
                [np.log1p(s["loc"]), np.log1p(s["budget"]), np.log1p(s["elapsed"])],
                s.get("meta_vec", [0.0] * 50),                    # 미개척 세션메타 50차원(open_files 등)
            ])
            text_block = np.array([1.0 if m in s["prompt"] else 0.0 for m in FILLERS] + [len(s["prompt"]) / 100.0])
            _e = os.environ                               # 다이어트 절제 플래그 (기본 전부 유지)
            parts = []
            if _e.get("DISABLE_RAG") != "1":
                parts += [self.retr.template_feats(s), knn[i]]   # RAG 검색(템플릿+kNN)
            parts += [seq_block, meta_block]
            if _e.get("DISABLE_SVD") != "1":
                parts.append(svd[i])                      # 프롬프트 SVD 128
            if _e.get("DISABLE_TEXT") != "1":
                parts.append(text_block)                  # 필러 플래그
            if self.emb is not None:
                if _e.get("DISABLE_RAG") != "1":
                    parts.append(eknn[i])                 # au emb-kNN(검색)
                if _e.get("DISABLE_E") != "1":
                    parts.append(E[i])                    # au raw e5 임베딩 384
            rows.append(np.concatenate(parts))
        return np.vstack(rows)
