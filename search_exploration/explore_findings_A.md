# SEARCH sub-action — Field & Identifier Audit (findings A)

**Question:** Is there a hidden KEY that decides the 4-way SEARCH sub-action
(read_file / grep_search / glob_pattern / list_directory)? Team model = ~60.8%.

**Verdict:** NO single deterministic key exists. The real structure is a **2nd-order
Markov policy over the ACTION-NAME SEQUENCE** (last two actions). It fully explains
~26% of SEARCH records (>80% pure) but leaves ~74% genuinely stochastic (~45%). The
data is a sampled policy, not a lookup — a ~60% ceiling is largely real.

Base rates (SEARCH only, n=28,782): grep 0.344 / read 0.322 / glob 0.184 / list 0.150.
Majority baseline = 0.344.

## What was RULED OUT (all ≈ base rate, honest held-out)
| Candidate key | Result |
|---|---|
| `id` number mod {2,3,4,5,7,10,100} | 0.349 = base. **No id leak.** |
| session number / au vs sim id parse | no modular signal |
| `turn_index` | == step number ALWAYS (0/70000 mismatch); redundant with id |
| user_tier / language_pref / last_ci_status / git_dirty | 0.34–0.37 |
| budget_tokens_remaining, elapsed_session_sec, loc (buckets) | ≤0.37 |
| **language_mix dominant lang / max-ratio / #langs** | glob stays 0.18–0.19 across ALL ratio buckets. **Glob hypothesis is FALSE** — glob *args* track the dominant language (`**/*.py`), but WHETHER glob is chosen does not. |
| open_files count/extensions | ≤0.40 |
| first-action records (empty history, 17% of SEARCH) | list_dir 0.37 mode, and **NO field beats 0.37** — pure stochastic start |
| **other search steps in SAME session (majority)** | **0.234 — WORSE than base.** Search sub-actions are drawn near-independently within a session; there is no per-session "tool signature". |

## What WORKS — the actual key = last-two action names (order-2 Markov)
Honest 70/30 held-out accuracy:
- majority 0.349 · last1 0.383 · **last2 0.550** · last3 0.540 (sparser) ·
  last_result_template 0.379 · richer combos overfit and DROP (kitchensink 0.415).
- Regularized LogisticRegression on all features (4-fold CV) = **0.558** (near the team's 0.608; the gap is likely prompt-text n-grams).

**Pure last2 cells (train purity ≥0.75) cover 26.2% of test at 81.2% acc; the other
73.8% sit at 45.5%.** Highest-purity pairs (n≥150):

| last two actions | → next | purity |
|---|---|---|
| read_file, read_file | grep_search | 0.88 |
| apply_patch, grep_search | grep_search | 0.86 |
| read_file, glob_pattern | grep_search | 0.85 |
| list_directory, grep_search | read_file | 0.85 |
| list_directory, read_file | read_file | 0.84 |
| grep_search, edit_file | glob_pattern | 0.83 |
| read_file, edit_file | grep_search | 0.82 |
| grep_search, grep_search | glob_pattern | 0.82 |
| grep_search, glob_pattern | glob_pattern | 0.82 |
| run_bash, grep_search | read_file | 0.82 |

(Full 67-row rule table saved to `markov2_rule_table.txt`.)

## The interpretable sub-rule: an ESCALATION LADDER
Repeating the **same narrow** search tool twice escalates to the next-broader tool:
- [read_file, read_file] → **grep_search 0.88**
- [grep_search, grep_search] → **glob_pattern 0.82**
- [glob_pattern, glob_pattern] → mixed **0.34** (ladder breaks)
- [list_directory, list_directory] → mixed **0.39** (ladder breaks)

The ladder collapses exactly at glob/list — which is precisely where the team reports
the model "collapses on glob/list." A single (non-repeated) last search action is only
~40% predictive. So glob and list are the tail of an escalation process with high
residual entropy, not a separable class.

## Why the LLM gets only 30.8%
Prompts are deliberately noised/decoupled from the action (e.g. user says "find all
.vue files" while the action is `glob **/*.py`; user says "open the file" while action
is grep). Semantic reasoning fights the data. The label is a function of the recent
ACTION TRAJECTORY, not the prompt meaning — a statistical policy, as suspected.

## Actionable for the team
1. Ensure the model has explicit **last-2 and last-3 action-name n-gram** features
   (bag-of-last-actions is NOT enough — last1 alone = 0.38 vs last2 = 0.55).
2. Add a **"repeat-run-length of the last search tool"** feature (captures the
   escalation ladder — the single most pure signal).
3. Accept ~40% residual on non-repeated / glob-vs-list contexts as **irreducible
   noise**. No field, id, or session key reaches 75%; ~60% is close to the real ceiling.
