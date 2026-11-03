# Phase 1 comparison protocol

Status: Phase 1 only. No training, no generation, no LLM judge, no README change.
Plan: `C:\Users\zaidk\.plannotator\plans\evaluation-plan-choose-the-rea-2026-11-03-approved.md`

## Data

- Experiments root: `D:\Preprint Critic\preprint_critic_review_experiments` (external, untracked).
- Each run is a zip containing `reviews.csv`, `summary_by_arm.csv`, `reviews.jsonl`, `run_config.json`.
- Use `reviews.csv`; the JSONL files are malformed for several runs because `reference_review` contains raw newlines.
- Do not copy or commit the zips, the test dataset, checkpoints, or adapters.

There are 15 expected runs:

```
a_bit_altered_prompt_60_papers
a_bit_altered_prompt_conservative_settings
gpt_prompt_high_temp
gpt_prompt_low_temp
new_prompt_with_conservative_settings
new_prompt_with_constrains_60_papers
original_prompt_high_temp
original_prompt_low_temp
original_prompt_original_settings
relaxed_references_prompt_high_temp
relaxed_references_prompt_low_temp
tweaked_prompt_high_temp
tweaked_prompt_mid_temp
very_big_prompt_high_temp
very_big_prompt_very_high_temp
```

The first 10 papers are shared by every run:

```
UgNQM-LcVpN, N78I92JIqOJ, wClmeg9u7G, pW2Q2xLwIMD, fdyxLGHE6bU,
g4szfsQUdy3, rylq7mF8IS, Jpxd93u2vK-, mF5tmqUfdsw, dZEZu7zxJBF
```

Samples are nested prefixes. Validate the nested-prefix property against a canonical ordered list built from the longest run.

## Loader contract

`experiments/analysis/load_runs.py`

- `EXPERIMENTS_DIR` defaults to `D:\Preprint Critic\preprint_critic_review_experiments`; override with `PREPRINT_EXPERIMENTS_DIR`.
- `load_all(root=EXPERIMENTS_DIR) -> (runs_df, reviews_df)`.
- `runs_df` has one row per run with:
  `run`, `n_papers`, `arms`, `paper_ids`, `prompt_family`, `max_paper_tokens`,
  `max_new_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty`,
  `no_repeat_ngram_size`, `do_sample`, `base_model`, `adapter_dir`.
- `reviews_df` has one row per `(run, paper_id, arm)` with at least:
  `run`, `paper_id`, `arm`, `review`, `reference_review`, `prompt_tokens`,
  `output_tokens`, `seconds`, `words`, `rouge_l`, `repeat5`, `hit_cap`, `seed`,
  `max_new_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty`,
  `no_repeat_ngram_size`.
- Derive `prompt_family` from the run name, not from the prompt text:
  `original`, `gpt`, `relaxed`, `tweaked`, `very_big`, `a_bit`, `new`.
- Validation errors must be explicit and aggregated, not a bare assert.
- Validate:
  - all 15 zips exist and contain the four files;
  - `reviews.csv` has the required columns;
  - exactly one row per `(paper_id, arm)`;
  - no empty reviews or references;
  - arms are a subset of `{base, adapter}` and match the run config/expected arms;
  - paper IDs equal the run config's `paper_ids` in order;
  - row count equals `n_papers * len(arms)`;
  - every run's paper list is a prefix of the canonical ordered list;
  - recomputed `summary_by_arm.csv` means match the CSV within `1e-3` for words,
    output tokens, ROUGE-L, repeat5, and cap count;
  - the reference review for a paper is identical across runs (normalize whitespace).
- `validate_all(...) -> list[str]` returns human-readable errors; empty means valid.
- `load_all` raises `ValueError` with all errors if validation fails.

## Local metrics contract

`experiments/analysis/local_metrics.py`

Pure functions are testable:

- `normalise(text)` collapses whitespace.
- `count_words(text)`.
- `repeat_stats(text, n=5)` -> repeat rate over word n-grams.
- `loop_stats(text, n=10)` -> `(repeat_rate, max_count, degenerate)`.
  `degenerate=True` only when a verbatim 10-gram repeats at least 3 times,
  or the 10-gram repeat rate is at least 0.5 with at least 50 words.
- `structure_flags(text)` -> dict with:
  `summary_opening`, `has_strengths`, `has_weaknesses`, `has_verdict`,
  `meta_commentary`, `meta_phrases`, `location_refs`, `section_refs`,
  `table_refs`, `figure_refs`, `equation_refs`, `location_any`.
- Meta phrases include, case-insensitively:
  `post-editing notes`, `i haven't read`, `i have not read`, `as an ai`,
  `initial summary`, `i cannot read`, `the provided text`, `based on the excerpt`.
- A location reference is any of:
  `Section N`, `Sec. N`, `§N`, `Table N`, `Figure N`, `Fig. N`, `Equation N`, `Eq. N`.
- `structure_score(row)` is the documented weighted score:
  `0.2*summary_opening + 0.2*has_strengths + 0.2*has_weaknesses + 0.4*location_any`.
- `length_fit(words, reference_words)`:
  `exp(-abs(log((words+1)/(reference_words+1))))`; 1.0 is a perfect length match.
- `build_metrics(reviews_df, papers_df, embedder=None)` -> per-review DataFrame.
  If `embedder` is None, embedding columns are NaN; otherwise add
  `emb_ref_cos`, `emb_paper_cos`, `emb_paper_cos_top3`, `emb_div`.
- Per-review schema:
  `run, arm, paper_id, review, reference_review, words, output_tokens, seconds,
  hit_cap, rouge_l, repeat5, repeat10, max_10gram_count, degenerate_loop,
  summary_opening, has_strengths, has_weaknesses, has_verdict,
  meta_commentary, meta_phrases, location_refs, section_refs, table_refs,
  figure_refs, equation_refs, location_any, structure_score, length_fit,
  emb_ref_cos, emb_paper_cos, emb_paper_cos_top3, emb_div`.

## Embeddings

- Model: `nomic-embed-text:latest`, 768 dimensions, local Ollama.
- Server: start `D:\Ollama\ollama.exe serve` with
  `OLLAMA_MODELS=D:\ollama`, `OLLAMA_HOST=127.0.0.1:11435`.
- Endpoint: `POST /api/embed`, body
  `{"model":"nomic-embed-text:latest","input":[...],"truncate":true,"options":{"num_ctx":8192}}`.
- Cache embeddings by SHA-256 of `(model, text)` in
  `experiments/cache/embeddings.sqlite`; never commit the cache.
- Paper chunks: clean `whole_paper`, split into 2,000-character chunks with
  200-character overlap, maximum 24 chunks per paper (about 48k characters,
  close to the runs' 17,500-token paper budget).
- `emb_ref_cos`: cosine(review embedding, reference embedding).
- `emb_paper_cos`: max cosine over paper chunks; `_top3` is the mean of the
  top three chunk cosines.
- `emb_div`: for each `(run, arm)`, sample up to 100 review pairs from different
  papers, mean cosine, and report the mean. Lower is more diverse.
- If the Ollama server fails, raise a clear error; do not silently substitute
  a different model. The fallback is documented but only used with an explicit
  `--fallback-minilm` flag.

## Analysis contract

`experiments/analysis/analyze_phase1.py`

- Adapter candidates are the 15 adapter arms. Base arms are paired controls.
- Aggregate per run and arm:
  `n`, words mean/median/p10/p90, output tokens mean/median, cap rate,
  `<150` and `>800` word rates, `repeat5` mean, `repeat10` mean,
  `degenerate_rate`, `meta_rate`, `location_any_rate`, `structure_score`,
  `length_fit`, `emb_ref_cos`, `emb_paper_cos`, `emb_paper_cos_top3`.
- Guardrails (adapter):
  `degenerate_loop` on 2+ papers, `meta_commentary` on 2+ papers,
  `hit_cap` rate > 0.05. A config with any guardrail failure is ineligible
  for the recommended winner, but still appears in the ranking.
- Composite:
  - rank each adapter config descending on `length_fit`, `emb_ref_cos`,
    `emb_paper_cos`, and `structure_score`;
  - average the four ranks; lower is better;
  - report every component, not only the composite.
- Force-in controls: `a_bit_altered_prompt_60_papers` and
  `original_prompt_original_settings`, fixed by name in `FORCED_CONTROLS`
  (not selected dynamically). `original_prompt_original_settings` is chosen
  because it has the highest ROUGE-L among runs with 20+ papers (0.1687); the
  absolute ROUGE leader, `gpt_prompt_low_temp` (0.1711), is a 10-paper run and
  is not used as the control. If either control is outside the top 6, it
  replaces the lowest-ranked non-control config and the report names the
  displaced run. Mark `forced_control=True`.
- Guardrails use two rules, both reported: a count rule (degenerate loop or
  meta-commentary on 2+ papers, `eligible`) and a rate rule (degenerate rate or
  meta rate above 5%, `eligible_strict`). The recommendation uses the strict
  rule when it is available and names the count-rule leader it disqualified.
- Paired bootstrap over papers (10,000 resamples, fixed RNG seed 42):
  adapter minus base for `words`, `length_fit`, `emb_ref_cos`,
  `emb_paper_cos`, `repeat5`, `structure_score`, and `hit_cap`.
  Report mean difference, 95% percentile CI, and win rate.
- Confound map: one row per pair of the 15 runs, listing which of
  `prompt_family`, `n_papers`, `temperature`, `top_p`, `top_k`,
  `repetition_penalty`, `max_paper_tokens`, `max_new_tokens`, `arms` differ.
  If more than one axis differs, `interpretation=package_effect`; otherwise
  `single_axis` or `identical`.
- Output `experiments/results/phase1_ranking.csv` with one row per adapter
  candidate and the columns:
  `rank, run, prompt_family, n_papers, eligible, eligible_strict,
  strict_failures, guardrail_failures,
  words_mean, length_fit, emb_ref_cos, emb_paper_cos, structure_score,
  repeat5_mean, degenerate_rate, meta_rate, cap_rate, rouge_l,
  composite_rank, forced_control, shortlisted, shortlist_reason`.
- Output `experiments/results/confound_map.csv`.
- Output `experiments/results/PHASE1_RESULTS.md` with:
  - one-paragraph verdict and stop-or-proceed recommendation;
  - validation summary;
  - ranking table;
  - guardrail table;
  - component metrics table;
  - paired adapter-vs-base table for runs that have both arms;
  - shortlist and reasons;
  - confound summary;
  - explicit limitations (10 common papers are dev papers, no human study,
    embeddings are proxies, sample sizes differ, 100-paper runs are adapter-only).
- Stop after Phase 1. Do not call the judge or write `judge_scores.jsonl`.

## Tests

`tests/test_experiment_analysis.py`

A few focused tests only:

- fixed-value tests for `repeat_stats`, `loop_stats`, `structure_flags`,
  `length_fit`, and `structure_score`;
- loader validation on the real data, skipped if the external directory is absent;
- a tiny synthetic two-run fixture to prove duplicate/arm/row-count validation fails;
- embedding cache key stability with a fake embedder (no network);
- ranking/shortlist logic on a tiny synthetic metrics table, including forced controls.

No exhaustive matrix, no network call in tests, no test that depends on the
Ollama server being up.

## Phase 1B addendum (adapter vs base split, top 7 screening set)

The user asked for the comparison to be split by arm and for the top 7
configurations to go forward, so Phase 1B changes three things.

**Separate leaderboards.** `rank_candidates(..., arm=...)` ranks one arm at a
time. Adapter and base are never mixed, because the base model writes roughly
1,000-2,000 more words per review. The base board answers a different question:
which prompt is best when the adapter is switched off. Artifacts:
`phase1_ranking.csv` (adapter), `phase1_ranking_base.csv` (base).

**Adapter lift.** `adapter_lift(aggregates)` compares the two arms of the same
run on the same papers, since that is the only clean comparison in the archive.
`components_won` counts how many of the four composite components the adapter
improves, and `verdict` is `helped` (3-4 wins), `hurt` (3-4 losses), or `wash`.
Artifact: `adapter_lift.csv`.

**Screening set size 7.** `SHORTLIST_SIZE = 7` and no configuration is forced
in. Guardrails are reported as annotations and do not remove a run from the
screening set. `CONTROL_RUNS` is retained for reporting only.

**All-human-reference similarity.** Each paper has 1-4 human reviews in the
dataset, but the run CSVs ship only one selected reference.
`load_all_references()` returns every human review per paper and
`add_all_reference_metrics()` adds `emb_ref_cos_all_max`,
`emb_ref_cos_all_mean`, and `n_human_reviews` to the per-review table.
`reference_sensitivity_sentence()` reports how much the shipped reference
choice moved the scores. The composite still uses `emb_ref_cos`, so the
ranking stays comparable with Phase 1A.

## Phase 2 addendum: LLM judge

Status: executed after Phase 1B approval. Scope fixed by the Phase 1B ranking.

- `experiments/analysis/judge.py` calls `POST http://127.0.0.1:4096/api/experimental/generate/text`
  with model `{"providerID":"opencode-go","id":"deepseek-v4.1-flash"}` and **no `variant` field**
  (the pilot used `max`; the user asked for the default).
- Auth is Basic `opencode:<password from ~/.config/opencode/service.json>`. The password is read at
  run time and never logged.
- Stateless: no session, no `sessionID`, no subagent, no per-call CLI. Each call is one HTTP request.
- Screening set: the top 7 adapter runs of `phase1_ranking.csv`, plus the base arms of the top 2.
  That is 9 configurations x the 10 shared papers = 90 absolute calls.
- Pairwise: the 3 adapter pairs among the top 3, plus adapter-vs-base for the top 2, on the same
  10 papers = 50 calls, with deterministic A/B order from a seeded RNG.
- Order-swap audit: the first 10 pairwise tasks repeated with A/B reversed.
- Every record is appended to `experiments/results/judge_scores.jsonl` and keyed by a SHA-256 of
  `(prompt_version, mode, paper_id, left, right, order)`. Completed keys are skipped, so the run
  is resumable. Invalid JSON is retried up to 3 times, then logged as a failure.
- Absolute rubric is byte-identical to the 5-call pilot so scores stay comparable.
- Guardrails are the pre-registered count rule: invented location on 2+ of 10 papers, meta-commentary
  on 2+, degenerate repetition on 2+, or cap rate above 5%. A configuration that trips a rule cannot
  be the recommended winner, but still appears in the tables. If nothing passes, the least severe
  configuration is named and the rule is not relaxed after the fact.

`experiments/analysis/analyze_phase2.py` writes `phase2_scores.csv`, `phase2_pairwise.csv`, and
`decision.md`. Statistics: paired bootstrap over papers (10,000 resamples, seed 42) for adapter vs
base, and Bradley-Terry with a 0.5 pseudo-count prior on the pairwise wins.

## Phase 3 addendum: token efficiency, length, settings, wider coverage

Status: requested after Phase 2. Still no training and no new review generation.
The judge is the only component that spends quota.

**Coverage expansion and like-for-like scores.** `judge.py` can judge more than the ten shared
papers: `--paper-limit N` uses the nested-prefix canonical order, and `--all-adapters` includes every
adapter run instead of only the top seven. Because a wider limit leaves some configurations with
twenty judged papers and others with ten, `analyze_phase2.common_paper_scores()` and
`attach_common_paper_scores()` add `overall_common` and `n_common`, the mean restricted to the ten
papers every run generated. `overall` is the full judged mean for that configuration;
`overall_common` is the comparable one, and `final_pick.py` selects on it.

**Quality per token.** `per_review_metrics.csv` has no `prompt_tokens`, so token
counts are read from the archived run CSVs. `analyze_phase2.load_cost_frame()`
builds `[config, arm, paper_id, prompt_tokens, output_tokens, words]`;
`token_costs()` averages it over the paper IDs that were actually judged; and
`attach_token_efficiency()` adds `output_tokens`, `prompt_tokens`, `words`,
`points_per_1k_output`, and `points_per_1k_total` to `phase2_scores.csv`.
Both ratios are reported because they answer different questions: output tokens
are what the review costs, total tokens are what the request costs.

**Review lengths.** `experiments/analysis/length_and_settings.py` writes
`length_bands.csv` (score per word band and arm), `settings_pairs.csv` (matched
run pairs with `other_axes_differ`), `settings_weights.csv` (ridge coefficients),
and `LENGTH_AND_SETTINGS.md`. A pair is only treated as a clean single-setting
comparison when exactly one of `temperature`, `top_p`, `top_k`,
`repetition_penalty`, `max_paper_tokens`, `max_new_tokens`, `n_papers` differs.
The ridge is fitted on the composite rank with leave-one-run-out and its R2 is
reported even when it is negative, because a negative R2 is the finding: seven
settings across fifteen runs do not explain the ranking.

**Wider judge coverage.** `judge.py` gained `canonical_papers(limit)`, which
takes the first `limit` IDs of the longest run's ordered paper list, so extra
papers are shared by every run that has them. New flags: `--paper-limit`,
`--pair-top`, `--all-adapters`. The expansion judges the top 7 adapters and 2
bases on 20 papers, all 15 adapters on the 10 shared papers, more adapter pairs
on 20 papers, and a reversed-order audit of 20 pairs. Resume keys are unchanged,
so already-scored calls are never paid for twice.

**Power.** Adapter-vs-adapter paired differences have a standard deviation near
13.6 points on ten papers, so the standard error at n=10 is 4.3 points and at
n=20 is 3.0. Adapter configurations differ by 1-5 points. Extra calls therefore
buy coverage and a paired design, not a statistically separated ranking, and the
report says so instead of claiming a significant ordering.

**Final configuration.** `experiments/analysis/final_pick.py` writes
`FINAL_CONFIG.md` using a rule declared before it reads the numbers: hard gates
from whole-run local metrics (degenerate loop rate <= 5%, meta-commentary rate
<= 5%, cap rate <= 5%), then mean judge usefulness, then a noise band of 5
points, then generated-token efficiency, paper cosine, and length fit. The
output lists the exact decoding settings and prompt file so the Kaggle run
reproduces the chosen configuration.

## Phase 4 addendum: full coverage and two gate families

### Coverage

The ten shared papers are too few to separate configurations: per-paper judge scores vary by
about 13 points, so one paper moves a config mean by more than a point. The two finalists were
therefore judged on every paper they generated, on both arms.

- `a_bit_altered_prompt_60_papers`: 60 papers x 2 arms.
- `very_big_prompt_high_temp`: 40 papers x 2 arms.

Their forty shared papers are the largest paired comparison available between them. Judge log
after this expansion: 532 records, 0 parse failures.

``judge --mode absolute --configs <a>,<b> --paper-limit 60 --workers 4``

`--configs` overrides the screening set and judges both arms of the named runs; `absolute_tasks`
skips papers a run never generated, so one paper limit serves both.

### Scores

`phase2_scores.csv` now carries two means per (config, arm):

- `overall` — mean usefulness over every paper that configuration was judged on. This is the
  primary score, because it uses all available evidence.
- `overall_common` / `n_common` — mean over the ten papers every run shares, kept as the
  like-for-like diagnostic across all fifteen runs.

### Gates

Two independent gate families, both applied to adapter arms only:

1. Local whole-run rates from `phase1_ranking.csv`: `degenerate_rate` <= 5%, `meta_rate` <= 5%,
   `cap_rate` <= 5%.
2. Judge-flagged shares of the papers read: `invented_location_count`, `meta_commentary_count`
   and `degenerate_repetition_count` each <= 25% of `n`. Failing labels are `judge_invented`,
   `judge_meta`, `judge_degenerate`.

`MAX_JUDGE_FLAG_RATE = 0.25` is the same standard that disqualified `gpt_prompt_high_temp` in
Phase 1 for a 10% meta-commentary rate, applied uniformly.

**Disclosure:** the judge-flag gate was added after full coverage showed that the ten-paper
ordering was misleading — the ten-paper leader `a_bit` (41.1) falls to 38.6 across sixty papers
with invented locations on 27% and role-play on 37%. The earlier ranking, produced with local
gates only, is preserved in the Phase 1 and Phase 3 sections and in the commit history.

### Selection

`final_pick.py` applies, in order: both gate families, mean `overall`, then a 5-point noise band
(`NOISE_BAND`), then the widest paired comparison between the tied configurations when exactly
two are tied, then points per 1,000 output tokens, then `emb_paper_cos`, then `length_fit`. It
emits the exact decoding settings and prompt file from the archived `run_config.json`.

Result: `very_big_prompt_high_temp` is the only configuration passing both gate families, at
40.2 across 40 judged papers. `original_prompt_high_temp` has the highest adapter mean (42.1) but
fails `judge_meta` at 45%; `a_bit` fails `judge_invented` at 27% and `judge_meta` at 37%.
