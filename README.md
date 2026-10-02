# InferAct — AI 에이전트 다음 행동 예측

**2026 AI·SW중심대학 디지털 경진대회 AI부문 「AI Agent 행동(Action) 의사결정 예측 챌린지」** (DACON, 2026.07)  
[🏆 최종 리더보드 (Private)](https://dacon.io/competitions/official/236694/leaderboard)  
**335팀 중 27위 · Team InferAct** · 최종(Private) Macro-F1 **0.79344** (1위 0.79878)

AI 코딩 에이전트 세션의 한 시점(사용자 발화, 직전 대화·행동 이력, 세션 메타정보)을 보고
에이전트가 **다음에 할 행동을 14개 중 하나로 예측**하는 문제입니다.
T4 GPU 1장, 오프라인, 추론 10분, 제출물 1GB라는 제약 안에서 돌아가는 경량 모델을 만들어야 했습니다.

이 저장소는 대회 기간 동안의 전체 파이프라인과, **무엇이 점수를 올렸고 무엇이 왜 실패했는지**에 대한 실험 기록입니다.

---

## 결과

| | Private Macro-F1 |
|---|---:|
| 1위 | 0.79878 |
| 12위 (본선 진출선) | 0.79642 |
| **27위 InferAct** | **0.79344** (1위와 0.0053 차이) |

출처: [DACON 리더보드 Private 탭](https://dacon.io/competitions/official/236694/leaderboard)

이 대회는 Private 점수가 예선 종료 시점의 Public 점수와 같아, 최종 제출의 Public 점수(0.7934409992)가 그대로 최종 순위 점수가 되었습니다.
대회 기간 중 Public LB 변화는 아래와 같습니다.

| 단계 | Public LB | 바뀐 점 |
|---|---:|---|
| v1 | 0.7520 | 세션 타임라인 복원 + 검색(RAG) + 계층 분류 |
| v2 | 0.7608 | 두 종류의 데이터 생성기(au/sim) 분리 |
| v6 | 0.7671 | au 전용 e5 파인튜닝 블렌드 |
| v8 | 0.7697 | 전체 문맥 e5 파인튜닝 + 코렉터(행동 인자·순서 스택) |
| 메타 | 0.7746 | **모델이 버리던 세션 메타정보(열린 파일 등) 활용** |
| 하드 라우팅 | 0.7802 | **그룹 확률을 곱하던 구조 → 그룹을 확정하고 그 안에서만 분류** |
| 07-08 | 0.7858 | au 전용 모델을 multilingual-e5-base로 교체 (어휘 프루닝으로 552MB → 201MB) |
| **07-15 최종 제출** | **0.79344** | **코렉터에 신호 두 가지를 더 스택**: 직전 검색 결과의 숫자(매치·파일·줄·항목 수)와 세션 메타 상호작용(토큰 예산×턴 등)을 au·전체 모델 확률과 함께 넣음. 직전 검색 결과가 탐색 그룹의 다음 행동 선택을 보강. held-out CV 0.7757 → 0.7780. 추론 7분 32초 |

> 최종 제출(07-15)의 코드는 이 저장소에 포함되어 있지 않습니다. 저장소의 코드는 07-08 기준(Public 0.7858)이며, 최종 제출은 그 위에 위 코렉터 피처를 추가한 것입니다.

- 하드 라우팅 버전 기준 추론 4분 26초 / 600초, 제출물 747MB / 1GB

---

## 문제 구조에서 찾은 것

### 1. 14개 행동은 4개 그룹으로 나뉘고, 그룹은 발화만으로 99% 결정된다

| 그룹 | 행동 |
|---|---|
| 탐색 SEARCH | `read_file` `grep_search` `glob_pattern` `list_directory` |
| 수정 MODIFY | `edit_file` `write_file` `apply_patch` |
| 실행 EXEC | `run_bash` `run_tests` `lint_or_typecheck` |
| 대화 TALK | `ask_user` `plan_task` `web_search` `respond_only` |

발화만으로 그룹을 맞히는 정확도가 99%였습니다. 즉 **틀리는 건 전부 그룹 안에서 세부 행동을 고를 때**입니다.
그래서 모델을 "그룹 판별(L1) → 그룹 내부 분류(L2)" 계층으로 쪼갰습니다.

### 2. 데이터는 성격이 다른 두 생성기에서 나왔다

샘플 ID가 `sess_au_*`(약 7%)와 `sess_sim_*`(약 93%) 두 계열로 나뉘고, 두 계열은 행동 분포와 규칙성이 다릅니다.
au는 발화의 의미를 따르는 편이라 언어 모델이 잘 풀고(F1 0.85 이상), sim은 이력이 무관한 주제로 섞여 있어 구조적 신호에 의존해야 합니다.
au 전용 모델을 따로 두고 au 행에만 블렌드한 것이 끝까지 가장 확실하게 LB로 이어진 개선이었습니다.

### 3. 탐색 그룹에서 발화는 미끼였다 — 숨은 키는 "두 스텝 전 행동"

탐색 4종(파일 읽기·검색·패턴 찾기·목록 보기, 전체의 41%)은 발화 내용과 정답이 거의 상관이 없었습니다.
LLM에게 데이터를 다 주고 풀게 해도 정확도 31%로, 다수결(34%)보다 낮았습니다.
대신 **두 스텝 전 행동(prev2)** 이 다음 탐색 행동을 강하게 결정하는 2주기 사이클이 있었습니다.

| 두 스텝 전 행동 | → 다음 행동 | 비율 |
|---|---|---:|
| `read_file` | `grep_search` | 85% |
| `grep_search` | `glob_pattern` | 82% |
| `list_directory` | `read_file` | 65% |

직전 행동(prev1)만 보면 38%, 두 스텝 전(prev2)을 보면 53%입니다. 다만 사이클이 깨끗한 구간은 26% 정도이고,
나머지는 같은 조건에서도 정답이 갈리는 확률적 샘플링이었습니다. 다섯 가지 방법으로 교차 검증한 결과,
**탐색 그룹 정확도 약 60%가 데이터가 허락하는 상한**이었고 우리 모델은 이미 그 근처(60.8%)에 있었습니다.
자세한 분석은 [`search_exploration/`](search_exploration/) 에 있습니다.

### 4. 모델이 버리고 있던 신호 — 열린 파일 목록

기본 피처는 `session_meta.workspace.open_files` 에서 **개수만** 쓰고 있었습니다.
파일 경로·확장자·종류, 그리고 "발화가 언급한 파일이 지금 열려 있는가" 같은 교차 피처 50차원을 추가하자
대형 모델이나 앙상블 없이 순수 피처만으로 **LB +0.0049** 가 올랐습니다.
예를 들어 발화가 열려 있는 파일을 언급하면 `grep_search` 확률이 평소의 2.5배였습니다.

### 5. 확률을 곱하지 말고 그룹을 확정하라

처음에는 P(그룹) × P(행동 | 그룹)으로 확률을 곱했는데, 그룹 판별의 작은 불확실성이 그룹 내부 분류를 흐리게 만들었습니다.
L1이 그룹을 하나로 확정하고 그 그룹의 L2만 쓰도록 바꾸자 held-out CV +0.0018, **LB +0.0035** 가 올랐습니다.
같은 시점에 사람이 손으로 넣은 규칙·혼동쌍 전문가 같은 개입은 held-out 에서 오히려 손해라 모두 걷어냈습니다.

---

## 파이프라인

```
train.jsonl ─► build_graph.py   겹치는 이력 윈도우를 이어 붙여 세션 타임라인 복원 (99.3% 정렬)
                  │
                  ├─ rules.py         정밀도 99% 이상 어휘 규칙 91개
                  ├─ retrieval.py     템플릿·시그니처 조회 + 문자 n-gram kNN (au/sim 분리)
                  ├─ features.py      발화 TF-IDF + 이력 순서·인자 + 세션 메타 (801차원)
                  │
                  ▼
           L1  그룹 판별 (LinearSVC, 정확도 99.2%) ── 그룹 확정 (하드 라우팅)
                  ▼
           L2  그룹별 HistGradientBoosting (class_weight=balanced)
                  ▼
           + 전체 문맥 e5 파인튜닝 확률 블렌드 (ft_full.py)
                  ▼
           코렉터  행동 인자·순서·세션 메타를 스택한 HGB (corrector.py, meta_feats.py)
                  ▼
           au 행만  au 전용 multilingual-e5-base 재블렌드 (ft_au.py, 어휘 프루닝)
                  ▼
           클래스별 오프셋 (Macro-F1 직접 최적화) ─► submission.csv
```

`pack.py` 가 위 구성 요소를 묶어 제출용 `submit.zip` 을 만들고, 3만 행 모의 채점으로 시간·용량을 검증합니다.

### 제약 안에 넣기
- **1GB 용량**: e5-base의 어휘 25만 개 중 실제 등장하는 8,970개만 남기는 어휘 프루닝으로 552MB → 201MB (`prune_vocab.py`)
- **10분 추론**: 모델 fp16 저장, 3만 행 모의 채점으로 매 제출 전 시간 검증 (하드 라우팅 버전 4분 26초)
- **재현성**: `PYTHONHASHSEED=0` 고정, 세션 단위 GroupKFold 3-fold 고정(`folds.json`)

---

## 안 된 것들

점수를 올리지 못한 시도도 held-out 실측과 함께 남겼습니다. 전체 목록은 [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) 에 있습니다.

| 시도 | 결과 |
|---|---|
| 평탄 14클래스 / 이진 캐스케이드 | 계층 분류보다 낮음 (0.7436, 0.7444 < 0.7471) |
| AutoGluon 전체 모델·앙상블 | 0.7204 ~ 0.7430, 직접 만든 계층 HGB에 패배 |
| 시드 배깅 / 가중치 수프 | +0.0003 / -0.0044 |
| 더 큰 임베딩 모델(e5-base·large) 전체 적용 | 용량 대비 이득 없음. au 전용일 때만 유효 |
| 세션 시퀀스 모델(GRU·어텐션) | 파인튜닝 모델이 이미 이력을 읽고 있어 +0.0005 ~ 0.0016로 미미 |
| 혼동쌍 전문가·수동 규칙 덮어쓰기 | held-out 각각 -0.0005 (학습 데이터 튜닝 이득이 일반화되지 않음) |
| 학습 세트 세션 매칭(누수 활용) | 학습에선 86.5%를 100% 정밀도로 맞혔지만 테스트에선 발동 0건 |

교훈은 두 가지였습니다. 대형 모델·앙상블보다 **모델이 아직 안 쓰고 있는 데이터 필드를 찾는 쪽**이 이득이 컸고,
사람이 손으로 넣은 개입은 학습 데이터에선 좋아 보여도 held-out 에선 상쇄되는 경우가 대부분이었습니다.

---

## 재현

대회 데이터는 재배포가 금지되어 있어 포함하지 않았습니다. [DACON 대회 페이지](https://dacon.io/competitions/official/236694)에서 받은 `open.zip` 의 `data/` 폴더를 저장소 루트의 `data/data/` 에 둡니다.

```
data/data/train.jsonl
data/data/train_labels.csv
```

```powershell
# Python 3.11
python -m venv work\venv311
work\venv311\Scripts\pip install -r requirements.txt
# torch 는 CUDA 버전에 맞게 별도 설치 (예: cu128)

powershell -ExecutionPolicy Bypass -File work\onto\reproduce.ps1
```

`reproduce.ps1` 은 타임라인 복원 → 임베딩 캐시 → 규칙 마이닝 → 3-fold OOF → au 파인튜닝 → 전체 학습 → 패키징·모의 채점까지 한 번에 실행하고 `work/onto/submit.zip` 을 만듭니다.
사전학습 모델은 [`dragonkue/multilingual-e5-small-ko-v2`](https://huggingface.co/dragonkue/multilingual-e5-small-ko-v2) 와 [`intfloat/multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base) (둘 다 MIT) 를 쓰고, 외부 데이터는 쓰지 않았습니다. e5-base 는 `FT_MODEL` 환경 변수로 지정합니다.

## 구조

```
work/onto/            파이프라인 (학습·추론·패키징)
  build_graph.py      세션 타임라인 복원, fold 분할
  rules.py            어휘 규칙
  retrieval.py        템플릿·kNN 검색
  features.py         피처
  meta_feats.py       세션 메타 피처 (열린 파일 등)
  train_hier.py       계층 분류 학습·OOF·오프셋
  corrector.py        코렉터 (train_corrector*.py 로 학습)
  ft_full.py ft_au.py e5 파인튜닝
  prune_vocab.py      어휘 프루닝
  pack.py             제출물 패키징 + 모의 채점
  reproduce.ps1       전체 재현
  folds.json          대회 기간 내내 고정해 쓴 3-fold 분할 (샘플 ID만 포함)
  *_probe.py 등        채택되지 않은 실험 스크립트
search_exploration/   탐색 그룹 숨은 규칙 분석 (prev2 사이클)
docs/                 대회 핵심 정리 튜토리얼
EXPERIMENT_LOG.md     LB 이력과 실패·기각 실험 기록
```

데이터 원문은 저장소 어디에도 포함되어 있지 않습니다. `folds.json` 과 `clean_sim_drop.json` 에는 샘플 ID만 있습니다.
