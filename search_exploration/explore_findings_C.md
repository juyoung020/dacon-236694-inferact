# Data Detective — SEARCH sub-action hidden KEY (Angle C: synthetic artifact / max-purity split)

## TL;DR
- **The hidden KEY is `prev2` — the assistant action TWO steps back in the history.** It ALONE
  predicts the next SEARCH sub-action at **53.5% held-out** (vs 34.4% majority, vs only 38.1% for
  `prev1`, the *last* action). The team was looking one step too shallow.
- The generation follows a **period-2 action cycle**, so the action 2-back is far more predictive
  than the last action. Core rules (held-out, sim subset):
  - `prev2 = read_file  → grep_search` (**85%**)
  - `prev2 = grep_search → glob_pattern` (**82%**)
  - `prev2 = list_directory → read_file` (**65%**)
  - `prev2 = edit_file / apply_patch → grep_search` (67% / 64%)
- A raw-feature model using only sequence + state (`p1,p2,p3`, prior-action counts, budget, loc,
  elapsed, n_open_files, src) reaches **58.8% held-out** on SEARCH — essentially matching the
  team's trained model (60.8%). **The prev2 sequence structure explains almost the entire signal.**
- **No fully-deterministic (>90%) key exists.** Best pockets are pair-conditioned ~84–88%
  (e.g. `(read_file,read_file)→grep 88%`). The generator is a *state-conditioned stochastic policy*
  whose dominant conditioning variable is prev2 — which is why even a strong LLM only gets 30.8%
  (it reasons over the prompt, but the prompt is a **decoy**).

## Key numbers (all held-out via folds.json, 3-fold)
Majority baseline (grep_search): **0.3444**

| Model / lookup | held-out SEARCH acc |
|---|---|
| prev1 (last action) lookup | 0.3809 |
| **prev2 (2-back) lookup** | **0.5349** |
| prev2 + src lookup | 0.5524 |
| (prev2,prev1) pair lookup | 0.5489 |
| (prev2,prev1)+src lookup | 0.5752 |
| Raw shallow DTree depth6 (state only, no p2 encoded) | 0.5046 |
| **Raw HistGBM (p1,p2,p3 + state + src)** | **0.5877** |
| — restricted to sim subset | 0.5799 |
| Team's trained model (reference) | 0.608 |

## Why prev1 fails but prev2 wins
In the history, `assistant_action` items alternate with `user` items, and the synthetic generator
emits the assistant's actions in a repeating 2-cycle. The immediately-preceding action (prev1) is
noisy filler (often an edit/test/bash), but the action **two back** locks the next SEARCH pick.
This is the single most important raw signal and the reason weak "prev1 / session-state" cues
capped the team's analysis.

Learned `prev2 → label` map (full data):
```
prev2=grep_search      -> glob_pattern   (p=0.801, n=3373)
prev2=read_file        -> grep_search    (p=0.813, n=2771)
prev2=list_directory   -> read_file      (p=0.646, n=1810)
prev2=edit_file        -> grep_search    (p=0.672, n=2749)
prev2=apply_patch      -> grep_search    (p=0.642, n=974)
prev2=run_tests        -> read_file      (p=0.637, n=1000)
prev2=run_bash         -> read_file      (p=0.567, n=1526)
prev2=glob_pattern     -> read_file      (p=0.377, n=1963)   <- WEAK (residual confusion)
prev2=NONE (fresh)     -> list_directory/read_file (~0.34)   <- WEAK (residual confusion)
```
Residual confusion concentrates in exactly two regimes: **fresh sessions (prev2=NONE, ~33% of
SEARCH rows)** and **prev2=glob_pattern**. Everything else is 65–85% pinned by prev2.

Highest-purity pair splits (still <90%, so not deterministic):
```
(read_file,   read_file)   -> grep_search  0.881
(apply_patch, grep_search) -> grep_search  0.859
(list_directory, read_file)-> read_file    0.837 / (list_directory,grep_search)->read 0.850
(read_file,   glob_pattern)-> grep_search  0.852
(grep_search, apply_patch) -> glob_pattern 0.840 / (grep_search,run_tests)->glob 0.841
```

## The prompt is a DECOY (confirms "wording is flat")
Explicit action words in `current_prompt` do NOT map to the matching action:
- prompt contains "list" → grep_search 36% (NOT list_directory)
- prompt contains "glob" → grep_search 38% (NOT glob_pattern)
- prompt contains "open"/"read" → grep/read tie, ~34%
Per-class log-odds tokens are all noise (in-class rate ~0.005). Real examples where the prompt
lies:
```
prompt "...Home.tsx 열어봐" (open Home.tsx)            LABEL = glob_pattern  (prev2=grep_search)
prompt "...열어줘 지금"     (open it now)              LABEL = glob_pattern  (prev2=grep_search)
prompt "open it..."                                    LABEL = grep_search   (prev2=read_file)
```
The label ignores the request and follows the prev2 cycle. This is why LLM reasoning over context
(30.8%) underperforms a dumb prev2 lookup (53.5%).

## au vs sim: the KEY lives in `sim`
Two id families: `sess_sim_*` (64,975 rows) and `sess_au_*` (5,025 rows). They are different regimes.
- **au** SEARCH rows (n=2039): read_file-dominant (63%), grep 27%, list/glob ~5% each. prev2 rule
  degenerates — au → read_file almost regardless of prev2 (grep→read 77%, read→read 57%). au is a
  near-constant read_file regime; the 2-cycle does NOT apply.
- **sim** SEARCH rows (n=26743): balanced (grep .35 / read .30 / glob .19 / list .16) and this is
  where the prev2 cycle is strong (read→grep .85, grep→glob .82). **The confusable core and the KEY
  are both in the sim subset.** A model should branch on `src_au` first (it was the tree's early
  split) and treat au ≈ read_file.

## Generation drift / timestamp: NONE
All sim ids share one date `20260522`; sorting SEARCH rows by (date, counter) into 5 buckets gives
identical label distributions (read ~.30 / grep ~.35 / glob ~.19 / list ~.16 in every bucket). Folds
are label-balanced. No time-ordering or position key — the generator sampled i.i.d. from a fixed
state-conditioned distribution.

## Actionable takeaway for the team
1. Add **prev2 (action 2 steps back)** as an explicit categorical feature — plus prev3. This is the
   missing key; it single-handedly lifts a lookup from 38%→53% and drives a raw GBM to ~59%.
2. Encode the **prev2→label cycle** (read→grep, grep→glob, list→read) — these are 65–85% priors.
3. **Branch on `src_au`**: predict read_file for au rows (the 2-cycle doesn't hold there).
4. Down-weight / ignore `current_prompt` lexical content for the SEARCH tie-break — it is an
   adversarial decoy. Session sequence state, not text, decides the sub-action.
5. Expect a ceiling near ~60–62% from raw features; the residual is genuinely stochastic
   (concentrated in fresh sessions and prev2=glob_pattern), so no >90% deterministic key exists.

## Scripts (in work/onto, prefix _detC_)
- `_detC_load.py` — join labels/folds, parse id into SRC/SEQ/SUB/STEP
- `_detC_tree.py` — baseline state tree (50.5%) + importances
- `_detC_cond.py` — conditional purity of prior_grep / n_open_files / etc.
- `_detC_trans.py` — (last-action, last-result) → next transition table
- `_detC_words.py` — prompt log-odds + action-word probes (decoy proof)
- `_detC_prev2.py` — prev2 / prev1 / pair transition tables (the key)
- `_detC_lookup.py` — held-out lookup-table accuracies
- `_detC_final.py` — definitive raw GBM ceiling (58.8%), au/sim split, drift check
