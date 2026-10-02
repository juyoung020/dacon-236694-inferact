# Detective B — Trajectory / Args / Determinism findings (SEARCH group)

Data: train.jsonl (70000), labels from oof.pkl['y'] (verified = ground truth; oof.pkl also holds
'P' = the team model's OOF probs and 'rl' = a rule prediction). SEARCH = {read_file, grep_search,
glob_pattern, list_directory}, N=28782 (sim 26743 + au 2039). Base rate = predict grep_search = 34.4%.

All accuracies below are HELD-OUT, GroupKFold(5) grouped by session id (no sibling-step leakage),
using a pure lookup-table (train-fold majority per key). This is the honest ceiling of each feature.

## TL;DR
- The determining KEY is the **prior ACTION-TYPE TRAJECTORY**, specifically the **positional action
  TWO steps back (prev2)**, inside a **period-2 escalation cycle**. Confirms detectives A/C on my data.
- **prompt wording, prior-action ARGS, and result_summary are essentially NOISE for the label.**
  Prompt is *actively anti-informative* (below base rate). Args/result add only +0.5 pt over trajectory.
- Two regimes: **au = flat read_file regime** (predict read_file); **sim = the escalation cycle**.
  Route on id prefix first.
- Honest all-feature ceiling ≈ **57.9%** (matches A/C's ~58.8%). Trained model = 60.8% (≈+2–3 pt,
  finer patterns / mild memorization; NOT gross session leakage — see §6).
- **~26–30% of SEARCH is cleanly explained (80–85% pure); the rest (~70%) is genuinely stochastic**
  and could NOT be cracked with any arg/result/phase feature I tried.

## 1. prev2 period-2 cycle — CONFIRMED, per regime (in-sample majority; held-out in §5)

sim (n=26743), P(next | prev2):
```
prev2=read_file      -> grep_search  84.9%   (n=2551)   <-- clean
prev2=grep_search    -> glob_pattern 81.8%   (n=3300)   <-- clean
prev2=edit_file      -> grep_search  68.6%   (n=2669)
prev2=apply_patch    -> grep_search  65.4%   (n=950)
prev2=list_directory -> read_file    64.9%   (n=1762)
prev2=run_tests      -> read_file    63.5%   (n=973)
prev2=run_bash       -> read_file    56.4%   (n=1506)
prev2=glob_pattern   -> read_file    37.5%   (n=1938)   <-- STOCHASTIC (read/grep tie)
prev2=plan_task      -> read_file    35.6%   (n=1000)   <-- STOCHASTIC
prev2=ask_user       -> read_file    40.2%   (n=920)    <-- STOCHASTIC
prev2=<none>         -> list_dir     33.9%   (n=8027)   <-- STOCHASTIC (0/1 prior action)
```
Escalation ladder: list_directory -> read_file -> grep_search -> glob_pattern (search broadens each step).

au (n=2039) is a DIFFERENT regime — near-constant read_file regardless of prev2:
```
every prev2 state -> read_file, 56–83%.  au label dist: read 1291, grep 550, glob 89, list 109.
prev1 (67.2%) slightly beats prev2 (63.3%) in au.  Just predict read_file (or use prev1).
```

## 2. prev2 is POSITIONAL, not "most recent search action" (my hypothesis, REFUTED)
- last_search_action (most recent read/grep/glob/list in history) held-out = **39.8%** (weak).
- Escalation fractions from last_search are only 32–53% — the ladder is NOT "escalate from last search."
- SMOKING GUN: on the subset where prev1 is ITSELF a search action (n=13185):
  prev1 held-out = **37.0%**, prev2 held-out = **60.9%**. Even when the last action is a search, it's the
  action two-steps-back that decides. prev1 is effectively noise for the label. => genuine period-2 coupling.

## 3. Higher order adds NOTHING beyond order-2 (sim, held-out GroupKFold)
```
prev1                 = 38.6%
prev2                 = 54.5%    <-- dominant single feature
prev1,prev2           = 56.8%    <-- prev1 adds +2.3 as a *pair* only
prev1,prev2,prev3     = 56.7%    (no gain from prev3)
prev2,prev3           = 54.3%    prev2,prev4 = 54.5%   (skipping prev1 = no help)
```
No longer cycle. Memory is order-2, prev2-dominant with a small prev1 interaction.

## 4. Phase / run-length (Q3) — weak, subsumed by prev1
```
prev2 alone            = 54.45%
prev2 + nact_parity    = 55.77%   (+1.3)
prev2 + turn_parity    = 55.78%   (+1.3)
prev2 + last_runlen    = 55.82%   (+1.4)
prev1,prev2            = 56.75%   (better; parity is just a weaker proxy for prev1)
```
Parity/run-length pin the cycle phase a little, but (prev1,prev2) already captures it and more.

## 5. Cracking the stochastic ~70% with ARGS/RESULT (Q4) — mostly FAILS
Per-prev2-state, held-out, adding (prev1 name + prev1 result_summary bucket):
```
prev2=read_file      84.9% -> 84.9%   (no change; already clean)
prev2=grep_search    81.8% -> 81.8%   (no change)
prev2=edit_file      68.6% -> 68.5%
prev2=list_directory 64.9% -> 63.6%
prev2=run_bash       56.4% -> 55.5%
prev2=glob_pattern   37.5% -> 39.8%   (+2.3 only; STILL stochastic)
prev2=plan_task      35.6% -> 39.5%
prev2=ask_user       40.2% -> 38.0%   (no help)
prev2=<none>         33.9% -> 42.7%   (+8.8 — but from n_actions split + prev1, i.e. TRAJECTORY, not args)
```
Concrete arg tests requested by coordinator:
- grep "0 matches" vs "many": result_summary bucket of prev1 gives at most +2 pt in any state; never separates.
- glob result (n files) of prev2: ZERO lift (37.51% -> 37.51%).
- prev2=glob_pattern read-vs-grep tie is unbreakable — e.g. conditioning on prev1:
  ```
  prev1=grep_search  n=537  {grep 173, glob 171, read 175}   <- near-perfect 3-way tie
  prev1=glob_pattern n=505  {grep 172, read 172, glob 126}
  prev1=read_file    n=265  {read 133, grep 107}
  ```
  No arg/result/name feature resolves it. This state is a fixed random draw by the simulator.

Only genuine recovery: prev2=<none> (8027 recs, biggest bucket) is not irreducibly random — split
0-action vs 1-action, and for 1-action use prev1: 33.9% -> 42.7%. (Sub-case "exactly 1 prior action",
n=3865: base 34.9% -> +prev1 42.95% -> +prev1+result 43.93%.) But this is trajectory info, not args.

## 6. Consolidated honest ceiling (FULL SEARCH sim+au, GroupKFold, lookup table)
```
base (grep_search)                              = 34.44%
prev2 only (no regime)                          = 53.36%
regime + prev2                                  = 55.11%
regime + prev1 + prev2                          = 57.44%
regime + prev1 + prev2 + prev1_result           = 57.94%   <-- best; args add only +0.5
regime + prev1 + prev2 + res1 + res2            = 57.78%
regime + prev1 + prev2 + parity                 = 57.40%
```
=> raw-feature honest ceiling ≈ **57.9%**. Team model (from oof P, restricted to 4) = 58.7% on my
measure / 60.8% reported. Model is ~1–3 pt above the raw-feature ceiling.

Leakage check (why the model can exceed feature ceiling a bit):
- Records are prefix-snapshots of ONE underlying session trajectory; for 63.7% of records the exact
  (history + label) action-seq is a verbatim prefix of a SIBLING step's history. So labels ARE
  recoverable across records IF you don't group CV by session.
- BUT for a feature model this barely matters: LR on trajectory feats, ungrouped KFold = 55.14% vs
  GroupKFold = 55.03% (+0.11 pt). Full-sequence memorization doesn't generalize (each session's
  trajectory is ~unique). Sibling-step labels also do NOT agree within a session (25.1% pairwise,
  BELOW the 27.8% chance). So no big practical leak; the model's edge is finer pattern capture.
- One unexplained slice: model scores 48.1% on empty-history (step_01) records where a strong
  char+word tfidf+meta classifier gets only 33.8% (BELOW the 37.0% base). No feature I built beats
  base there. Likely the FT transformer memorizes weak prompt-template regularities; on truly novel
  sessions this slice should regress toward ~37%.

## 7. Why an LLM reasoning gets 30.8% (below base) — the DECOY
- Prompt semantics are DELIBERATELY DECORRELATED from the label. Same intent scatters across all 4:
  "그 파일 보여줘/열어봐" (show/open file) -> grep_search; "몇 군데인지 훑어줘 / 호출하는 곳 찾아줘"
  (find occurrences) -> list_directory; "재귀로 뽑아봐 / 다 긁어줘" -> read_file; "*_test.go 패턴으로
  찾아봐" -> glob_pattern.
- Quantified: prompt char-ngram LR on empty-history = 34.7% (< 37% base). Prompt-only on full SEARCH
  = 31.3% (≈ the LLM's 30.8%). Adding prompt to the trajectory feature HURTS by ~4 pt.
- => the label follows a STRUCTURAL statistical process (positional prev2 cycle), not content. Any
  content-based reasoner is actively misled. This is the whole trap.

## 8. Actionable recommendations for the model
1. ROUTE on id-prefix regime first: au -> bias hard to read_file; sim -> use the cycle.
2. Core features: prev2 (positional, incl. non-search actions), prev1, regime. Add n_actions and the
   0/1-action split for the prev2=<none> bucket. These get you to the ~57.9% honest ceiling.
3. DROP prompt text and prior-action args/result as label features — they are noise/decoys and can
   only hurt calibration within SEARCH. (Keep them only for the group-level decision if it helps there.)
4. Do NOT expect to beat ~58–61%: ~70% of SEARCH (esp. prev2 in {glob_pattern, plan_task, ask_user,
   <none>}) is a fixed random draw; the read↔grep and read↔grep↔glob confusions there are irreducible.
5. Make sure CV is GroupKFold by session to avoid a small optimistic bias and to keep LB honest.
```
```
