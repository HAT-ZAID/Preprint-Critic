# Phase 2 decision: LLM judge on the top-7 adapter screening set

Judge model `deepseek-v4.1-flash` at the default variant, stateless one-shot calls, 663 scored calls. No human study was run; the judge is a proxy, not ground truth.

## Verdict

- **Highest judged usefulness: `original_prompt_high_temp` [base] at 73.7/100.**
- **Fine-tuning did not help on this judge.** The best adapter, `original_prompt_high_temp`, scores 42.1, which is 31.5 points below the best base arm, `original_prompt_high_temp`, at 73.7.
- **No configuration passes the pre-registered guardrails.** The least severe is `original_prompt_high_temp` [base] with severity 1 (meta_commentary 3/20); the rule is applied as written rather than relaxed after the fact.

## Absolute rubric

| config | arm | n | overall | n common | overall (10 shared) | factual | grounding | coverage | actionability | argument | calibration | length | eligible | failures |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| original_prompt_high_temp | base | 20 | 73.7 | 10 | 72.7 | 6.2 | 7.2 | 7.5 | 8.0 | 6.7 | 6.3 | 7.0 | no | meta_commentary 3/20 |
| a_bit_altered_prompt_60_papers | base | 60 | 73.6 | 10 | 70.1 | 6.1 | 7.2 | 7.9 | 8.1 | 6.8 | 6.1 | 6.7 | no | invented_location 16/60; meta_commentary 12/60 |
| new_prompt_with_constrains_60_papers | base | 60 | 72.2 | 10 | 70.6 | 6.3 | 7.3 | 7.9 | 7.8 | 6.7 | 6.1 | 5.3 | no | invented_location 27/60; meta_commentary 59/60; degenerate_repetition 3/60 |
| very_big_prompt_high_temp | base | 40 | 71.3 | 10 | 66.0 | 6.0 | 7.2 | 7.8 | 8.0 | 6.5 | 6.0 | 5.4 | no | invented_location 12/40; meta_commentary 32/40 |
| gpt_prompt_high_temp | base | 10 | 55.1 | 10 | 55.1 | 4.9 | 5.9 | 5.6 | 6.5 | 5.0 | 4.6 | 3.6 | no | invented_location 7/10; meta_commentary 10/10 |
| original_prompt_high_temp | adapter | 20 | 42.1 | 10 | 36.0 | 4.7 | 4.3 | 3.7 | 4.0 | 3.5 | 4.0 | 4.9 | no | invented_location 3/20; meta_commentary 9/20 |
| very_big_prompt_high_temp | adapter | 40 | 40.2 | 10 | 37.0 | 4.8 | 4.2 | 3.5 | 3.7 | 3.3 | 4.0 | 4.2 | no | invented_location 7/40; meta_commentary 10/40; degenerate_repetition 3/40 |
| a_bit_altered_prompt_60_papers | adapter | 60 | 38.6 | 10 | 41.1 | 4.2 | 4.1 | 3.5 | 3.8 | 3.4 | 3.7 | 4.6 | no | invented_location 16/60; meta_commentary 22/60; degenerate_repetition 2/60 |
| new_prompt_with_constrains_60_papers | adapter | 60 | 38.5 | 10 | 40.0 | 4.7 | 4.1 | 3.4 | 3.7 | 3.4 | 3.7 | 4.2 | no | invented_location 18/60; meta_commentary 26/60 |
| tweaked_prompt_mid_temp | adapter | 10 | 37.3 | 10 | 37.3 | 4.6 | 4.5 | 3.8 | 3.7 | 3.4 | 4.0 | 4.1 | no | meta_commentary 2/10; degenerate_repetition 2/10 |
| tweaked_prompt_high_temp | adapter | 20 | 37.0 | 10 | 33.2 | 4.4 | 3.8 | 3.6 | 3.5 | 3.1 | 3.5 | 4.1 | no | invented_location 9/20; meta_commentary 9/20 |
| original_prompt_original_settings | adapter | 20 | 36.5 | 10 | 36.2 | 4.7 | 4.3 | 3.2 | 3.5 | 3.2 | 3.6 | 3.8 | no | invented_location 3/20; meta_commentary 3/20; degenerate_repetition 7/20 |
| gpt_prompt_high_temp | adapter | 10 | 35.5 | 10 | 35.5 | 4.5 | 3.8 | 3.3 | 3.5 | 3.0 | 3.4 | 3.6 | no | invented_location 5/10; meta_commentary 4/10 |
| original_prompt_low_temp | adapter | 20 | 35.2 | 10 | 32.1 | 5.2 | 3.6 | 3.1 | 3.1 | 3.1 | 3.4 | 3.7 | no | invented_location 5/20; degenerate_repetition 4/20 |
| a_bit_altered_prompt_conservative_settings | adapter | 20 | 35.0 | 10 | 34.6 | 5.2 | 3.5 | 3.2 | 3.1 | 2.9 | 3.8 | 3.6 | no | invented_location 3/20; meta_commentary 4/20; degenerate_repetition 3/20 |
| relaxed_references_prompt_low_temp | adapter | 10 | 34.4 | 10 | 34.4 | 4.9 | 3.5 | 2.9 | 3.3 | 2.9 | 3.6 | 3.5 | no | degenerate_repetition 3/10 |
| relaxed_references_prompt_high_temp | adapter | 10 | 33.9 | 10 | 33.9 | 4.0 | 3.6 | 2.8 | 3.2 | 3.0 | 3.4 | 3.1 | no | invented_location 2/10; meta_commentary 6/10 |
| gpt_prompt_low_temp | adapter | 10 | 31.2 | 10 | 31.2 | 4.9 | 3.8 | 2.7 | 2.7 | 2.7 | 3.6 | 3.0 | no | degenerate_repetition 3/10 |
| new_prompt_with_conservative_settings | adapter | 20 | 30.1 | 10 | 30.3 | 4.8 | 3.2 | 2.7 | 2.6 | 2.5 | 3.1 | 2.9 | no | invented_location 3/20; meta_commentary 4/20; degenerate_repetition 2/20 |
| very_big_prompt_very_high_temp | adapter | 10 | 29.0 | 10 | 29.0 | 3.6 | 3.0 | 3.0 | 2.6 | 2.6 | 2.9 | 3.7 | no | invented_location 5/10; meta_commentary 4/10 |

`overall (10 shared)` restricts each configuration to the ten papers that every run generated, so configurations judged on twenty papers can be compared like with like. The plain `overall` column uses every judged paper for that configuration.

## Quality per token

| config | arm | overall | words | output tokens | prompt tokens | pts / 1k output | pts / 1k total |
|---|---|---|---|---|---|---|---|
| original_prompt_low_temp | adapter | 35.2 | 281 | 373 | 11237 | 94.3 | 3.03 |
| original_prompt_high_temp | adapter | 42.1 | 349 | 456 | 11237 | 92.4 | 3.60 |
| new_prompt_with_conservative_settings | adapter | 30.1 | 249 | 342 | 11479 | 88.1 | 2.55 |
| relaxed_references_prompt_high_temp | adapter | 33.9 | 300 | 389 | 11832 | 87.1 | 2.77 |
| gpt_prompt_low_temp | adapter | 31.2 | 271 | 367 | 11794 | 85.0 | 2.57 |
| very_big_prompt_high_temp | adapter | 40.2 | 374 | 495 | 12783 | 81.2 | 3.03 |
| gpt_prompt_high_temp | adapter | 35.5 | 342 | 447 | 11794 | 79.4 | 2.90 |
| new_prompt_with_constrains_60_papers | adapter | 38.5 | 377 | 497 | 11345 | 77.5 | 3.25 |
| tweaked_prompt_high_temp | adapter | 37.0 | 357 | 481 | 11400 | 77.1 | 3.12 |
| a_bit_altered_prompt_60_papers | adapter | 38.6 | 389 | 507 | 11138 | 76.2 | 3.32 |
| very_big_prompt_very_high_temp | adapter | 29.0 | 361 | 496 | 12986 | 58.5 | 2.15 |
| original_prompt_original_settings | adapter | 36.5 | 381 | 641 | 11223 | 56.8 | 3.07 |
| a_bit_altered_prompt_conservative_settings | adapter | 35.0 | 507 | 685 | 11272 | 51.1 | 2.92 |
| tweaked_prompt_mid_temp | adapter | 37.3 | 634 | 836 | 11948 | 44.6 | 2.92 |
| relaxed_references_prompt_low_temp | adapter | 34.4 | 591 | 965 | 11832 | 35.7 | 2.69 |
| original_prompt_high_temp | base | 73.7 | 1459 | 2102 | 11237 | 35.0 | 5.52 |
| a_bit_altered_prompt_60_papers | base | 73.6 | 1727 | 2506 | 11138 | 29.4 | 5.39 |
| very_big_prompt_high_temp | base | 71.3 | 2322 | 3366 | 12783 | 21.2 | 4.42 |
| new_prompt_with_constrains_60_papers | base | 72.2 | 2306 | 3416 | 11345 | 21.1 | 4.89 |
| gpt_prompt_high_temp | base | 55.1 | 1994 | 3104 | 11794 | 17.7 | 3.70 |

- Most efficient on generated tokens: `original_prompt_low_temp` [adapter] at 94.3 usefulness points per 1,000 output tokens.
- Most efficient on all tokens: `original_prompt_high_temp` [base] at 5.52 points per 1,000 total tokens.
- These two answers disagree on purpose. The paper prompt is roughly 17,500 tokens and is identical for both arms, so it dominates any total-token count. The adapter saves output tokens only; it does not save the paper reading. Per generated token the adapter is the cheaper reviewer, per end-to-end token the longer base review still returns more usefulness per token spent.
- The score is a 0-100 rubric total, not a linear quantity of value, so a ratio of score to tokens is a coarse efficiency index, not a budget forecast.

## Flag counts

| config | arm | invented | meta | degenerate | filler | cap_rate |
|---|---|---|---|---|---|---|
| original_prompt_high_temp | base | 1 | 3 | 0 | 2 | 0.000 |
| a_bit_altered_prompt_60_papers | base | 16 | 12 | 1 | 6 | 0.000 |
| new_prompt_with_constrains_60_papers | base | 27 | 59 | 3 | 40 | 0.000 |
| very_big_prompt_high_temp | base | 12 | 32 | 1 | 20 | 0.050 |
| gpt_prompt_high_temp | base | 7 | 10 | 1 | 8 | 0.000 |
| original_prompt_high_temp | adapter | 3 | 9 | 1 | 13 | 0.000 |
| very_big_prompt_high_temp | adapter | 7 | 10 | 3 | 29 | 0.000 |
| a_bit_altered_prompt_60_papers | adapter | 16 | 22 | 2 | 46 | 0.000 |
| new_prompt_with_constrains_60_papers | adapter | 18 | 26 | 0 | 43 | 0.000 |
| tweaked_prompt_mid_temp | adapter | 1 | 2 | 2 | 9 | 0.100 |
| tweaked_prompt_high_temp | adapter | 9 | 9 | 1 | 15 | 0.000 |
| original_prompt_original_settings | adapter | 3 | 3 | 7 | 18 | 0.033 |
| gpt_prompt_high_temp | adapter | 5 | 4 | 0 | 6 | 0.000 |
| original_prompt_low_temp | adapter | 5 | 1 | 4 | 19 | 0.000 |
| a_bit_altered_prompt_conservative_settings | adapter | 3 | 4 | 3 | 20 | 0.040 |
| relaxed_references_prompt_low_temp | adapter | 1 | 0 | 3 | 10 | 0.100 |
| relaxed_references_prompt_high_temp | adapter | 2 | 6 | 0 | 9 | 0.000 |
| gpt_prompt_low_temp | adapter | 0 | 0 | 3 | 10 | 0.000 |
| new_prompt_with_conservative_settings | adapter | 3 | 4 | 2 | 20 | 0.020 |
| very_big_prompt_very_high_temp | adapter | 5 | 4 | 0 | 9 | 0.000 |

## Pairwise

| A | B | A wins | B wins | ties |
|---|---|---|---|---|
| gpt_prompt_high_temp [adapter] | gpt_prompt_high_temp [base] | 4 | 6 | 0 |
| gpt_prompt_high_temp [adapter] | original_prompt_high_temp [adapter] | 7 | 3 | 0 |
| gpt_prompt_high_temp [adapter] | original_prompt_original_settings [adapter] | 4 | 6 | 0 |
| gpt_prompt_high_temp [adapter] | very_big_prompt_high_temp [adapter] | 6 | 4 | 0 |
| original_prompt_high_temp [adapter] | original_prompt_high_temp [base] | 0 | 20 | 0 |
| original_prompt_high_temp [adapter] | original_prompt_original_settings [adapter] | 12 | 8 | 0 |
| original_prompt_high_temp [adapter] | very_big_prompt_high_temp [adapter] | 12 | 8 | 0 |
| original_prompt_original_settings [adapter] | very_big_prompt_high_temp [adapter] | 9 | 11 | 0 |

## Bradley-Terry strength

| rank | config | strength |
|---|---|---|
| 1 | original_prompt_high_temp [base] | 34.03 |
| 2 | gpt_prompt_high_temp [base] | 0.87 |
| 3 | gpt_prompt_high_temp [adapter] | 0.54 |
| 4 | original_prompt_high_temp [adapter] | 0.46 |
| 5 | original_prompt_original_settings [adapter] | 0.37 |
| 6 | very_big_prompt_high_temp [adapter] | 0.37 |

## Paired bootstrap (adapter vs base, 10,000 resamples)

| comparison | n | mean diff | 95% CI | adapter win rate |
|---|---|---|---|---|
| a_bit_altered_prompt_60_papers: adapter vs base | 60 | -35.0 | [-38.2, -31.5] | 2% |
| gpt_prompt_high_temp: adapter vs base | 10 | -19.6 | [-30.9, -8.1] | 20% |
| new_prompt_with_constrains_60_papers: adapter vs base | 60 | -33.7 | [-37.2, -30.2] | 0% |
| original_prompt_high_temp: adapter vs base | 20 | -31.5 | [-36.2, -26.4] | 0% |
| very_big_prompt_high_temp: adapter vs base | 40 | -31.2 | [-34.7, -27.5] | 0% |

## Order-swap audit

Repeated 13 pairwise calls with A/B reversed: **12/13 agreed** (92%), 0 tie(s).

## Limitations

- 10–60 papers, all used earlier for prompt selection; they are dev papers, not a holdout.
- Judged paper counts differ across configurations (10, 20, 40, 60), because only the shorter runs share the full set. The `overall (10 shared)` column is the like-for-like comparison; the plain `overall` column is not comparable across differing counts.
- One LLM judge, one prompt version, no repeated sampling of the same review, so judge variance is unmeasured.
- The absolute rubric weights coverage and grounding, which structurally favours longer reviews.
- Base arms are longer (roughly 1,000–2,000 more words) and slower at generation time; the efficiency table above is the only token accounting, and it is an index rather than a cost model.
