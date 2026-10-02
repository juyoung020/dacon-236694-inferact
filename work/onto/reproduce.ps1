# DACON 236694 — 전체 재현 스크립트 (본선 "Private 복원 학습코드" 겸용)
# 사용: powershell -ExecutionPolicy Bypass -File reproduce.ps1
# 산출: work\onto\submit.zip (그래프 → 룰 → 학습 → 패키징 → 30k 모의채점까지)
# 요구: work\venv311 (Python 3.11.15, numpy==1.26.4, scikit-learn==1.8.0, joblib==1.5.3,
#       pandas==2.0.3, scipy==1.15.3, torch cu128, transformers==4.46.3)
#       work\onto\hf_cache (dragonkue/multilingual-e5-small-ko-v2, MIT)

$ErrorActionPreference = "Stop"
$env:PYTHONHASHSEED = "0"          # 재현성: 인터프리터 시작 전 고정 필수
$env:PYTHONIOENCODING = "utf-8"

$ONTO = Split-Path -Parent $MyInvocation.MyCommand.Path
$PY = Join-Path (Split-Path -Parent $ONTO) "venv311\Scripts\python.exe"
Set-Location $ONTO

Write-Host "[1/7] 타임라인 복원 + 3-fold 고정 → graph.pkl, folds.json"
& $PY -u build_graph.py; if (-not $?) { throw "build_graph 실패" }

Write-Host "[2/7] 임베딩 캐시 (얼린 e5, GPU 있으면 ~20초)"
if (-not (Test-Path "$ONTO\emb_cache.npz")) { & $PY -u embed_cache.py; if (-not $?) { throw "embed_cache 실패" } }

Write-Host "[3/7] 어휘 룰 마이닝 → rules.json"
& $PY -u rules.py; if (-not $?) { throw "rules 실패" }

Write-Host "[4/7] 3-fold OOF (오프셋·개입강도 산출) → oof.pkl"
& $PY -u train_hier.py; if (-not $?) { throw "OOF 실패" }

Write-Host "[5/7] au e5 파인튜닝 (블렌드용) → ft_oof.pkl, e5_ft/"
& $PY -u ft_au.py; if (-not $?) { throw "ft OOF 실패" }
& $PY -u ft_au.py --full; if (-not $?) { throw "ft full 실패" }
& $PY -u ft_blend.py; if (-not $?) { throw "ft blend 실패" }

Write-Host "[6/7] 전체 학습 → bundle.joblib"
& $PY -u train_hier.py --full; if (-not $?) { throw "full 학습 실패" }

Write-Host "[7/7] 패키징 + 30,000행 모의채점 → submit.zip"
& $PY -u pack.py; if (-not $?) { throw "pack 실패" }

Write-Host "완료: $ONTO\submit.zip"
