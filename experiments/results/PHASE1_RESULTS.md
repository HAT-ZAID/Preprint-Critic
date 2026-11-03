# Phase 1 results: local comparison of the 15 archived runs

Phase 1 only. No generative model was called, no OpenCode session was created, no training or new review generation was run.

## Verdict

**Phase 2 screening set: the top 7 adapter configurations below.** The user asked for the top 7 by local rank, so guardrails are reported as annotations and do not remove a config from the screening set.

Local-only leader: `original_prompt_high_temp` (composite mean rank 3.50, 20 papers). `original_prompt_high_temp` passes the guardrails, leads the composite by 2.8 rank positions over `very_big_prompt_high_temp`, and is best on 2 of 4 components. The count-based rule alone would have ranked `gpt_prompt_high_temp` first, but it is excluded by the rate-based rule (meta_rate 10.0% > 5%).

Adapter and base are ranked on **separate leaderboards** because the untuned model writes roughly 1,000-2,000 more words per review; mixing them would compare length policies, not review quality.

## Validation

- All 15 runs validated: required files, columns, row counts, paper order, summary aggregates, and reference identity.
- Runs loaded: 15; review rows: 960.
- Metrics source: loaded `per_review_metrics.csv`.
- Embedding model: `nomic-embed-text:latest` (768-d, local Ollama).

## Ranking (adapter arms)

| rank | run | n_papers | eligible | eligible_strict | composite_rank | length_fit | emb_ref_cos | emb_paper_cos | structure_score | rouge_l | shortlisted |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | gpt_prompt_high_temp | 10 | yes | no | 1.250 | 0.652 | 0.861 | 0.859 | 0.680 | 0.158 | yes |
| 2 | original_prompt_high_temp | 20 | yes | yes | 3.500 | 0.664 | 0.851 | 0.835 | 0.610 | 0.142 | yes |
| 3 | original_prompt_original_settings | 60 | no | no | 4.500 | 0.628 | 0.853 | 0.856 | 0.480 | 0.169 | yes |
| 4 | very_big_prompt_high_temp | 40 | yes | yes | 6.250 | 0.609 | 0.858 | 0.832 | 0.505 | 0.153 | yes |
| 5 | tweaked_prompt_high_temp | 20 | yes | yes | 6.500 | 0.603 | 0.851 | 0.829 | 0.560 | 0.144 | yes |
| 6 | original_prompt_low_temp | 20 | no | no | 8.250 | 0.537 | 0.843 | 0.842 | 0.480 | 0.156 | yes |
| 7 | relaxed_references_prompt_high_temp | 10 | yes | yes | 8.250 | 0.540 | 0.829 | 0.838 | 0.540 | 0.144 | yes |
| 8 | gpt_prompt_low_temp | 10 | no | no | 8.750 | 0.514 | 0.858 | 0.855 | 0.440 | 0.171 | no |
| 9 | a_bit_altered_prompt_60_papers | 60 | yes | yes | 9.250 | 0.613 | 0.827 | 0.823 | 0.537 | 0.146 | no |
| 10 | new_prompt_with_constrains_60_papers | 60 | yes | yes | 9.500 | 0.621 | 0.828 | 0.823 | 0.503 | 0.145 | no |
| 11 | very_big_prompt_very_high_temp | 40 | yes | yes | 9.500 | 0.583 | 0.838 | 0.813 | 0.560 | 0.131 | no |
| 12 | a_bit_altered_prompt_conservative_settings | 100 | no | no | 10.000 | 0.589 | 0.830 | 0.832 | 0.452 | 0.147 | no |
| 13 | relaxed_references_prompt_low_temp | 10 | no | no | 10.250 | 0.536 | 0.839 | 0.832 | 0.460 | 0.154 | no |
| 14 | tweaked_prompt_mid_temp | 20 | no | no | 10.250 | 0.565 | 0.838 | 0.824 | 0.470 | 0.151 | no |
| 15 | new_prompt_with_conservative_settings | 100 | no | no | 14.000 | 0.532 | 0.818 | 0.818 | 0.460 | 0.139 | no |

Every component is shown. `composite_rank` is the mean of the four component ranks (lower is better); it is not a quality score.

## Ranking (base arms, same prompts, untuned model)

| rank | run | n_papers | eligible | composite_rank | length_fit | emb_ref_cos | emb_paper_cos | structure_score | rouge_l |
|---|---|---|---|---|---|---|---|---|---|
| 1 | relaxed_references_prompt_high_temp | 10 | yes | 3.500 | 0.252 | 0.860 | 0.858 | 0.760 | 0.101 |
| 2 | tweaked_prompt_high_temp | 20 | yes | 4.250 | 0.203 | 0.850 | 0.867 | 0.720 | 0.087 |
| 3 | relaxed_references_prompt_low_temp | 10 | yes | 4.750 | 0.254 | 0.852 | 0.864 | 0.640 | 0.103 |
| 4 | original_prompt_high_temp | 20 | yes | 6.000 | 0.290 | 0.839 | 0.848 | 0.650 | 0.099 |
| 5 | original_prompt_original_settings | 60 | yes | 7.000 | 0.281 | 0.838 | 0.862 | 0.523 | 0.104 |
| 6 | tweaked_prompt_mid_temp | 20 | yes | 7.000 | 0.206 | 0.848 | 0.860 | 0.630 | 0.088 |
| 7 | original_prompt_low_temp | 20 | yes | 7.000 | 0.306 | 0.843 | 0.846 | 0.520 | 0.103 |
| 8 | gpt_prompt_high_temp | 10 | yes | 7.750 | 0.265 | 0.805 | 0.791 | 0.740 | 0.098 |
| 9 | gpt_prompt_low_temp | 10 | no | 8.250 | 0.282 | 0.787 | 0.759 | 0.700 | 0.102 |
| 10 | very_big_prompt_very_high_temp | 40 | yes | 8.500 | 0.176 | 0.845 | 0.840 | 0.685 | 0.083 |
| 11 | new_prompt_with_constrains_60_papers | 60 | yes | 9.000 | 0.178 | 0.827 | 0.844 | 0.693 | 0.081 |
| 12 | very_big_prompt_high_temp | 40 | yes | 9.000 | 0.182 | 0.846 | 0.837 | 0.635 | 0.086 |
| 13 | a_bit_altered_prompt_60_papers | 60 | yes | 9.000 | 0.239 | 0.831 | 0.837 | 0.650 | 0.091 |

Base reviews are much longer, so `length_fit` is low for almost every base run. This board answers a different question: which prompt produces the best output when the adapter is switched off.

## Adapter lift vs base (same run, same papers)

| run | n_papers | d_length_fit | d_emb_ref_cos | d_emb_paper_cos | d_structure_score | d_words_mean | d_repeat5_mean | d_cap_rate | components_won | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| a_bit_altered_prompt_60_papers | 60 | 0.375 | -0.003 | -0.014 | -0.113 | -1338.350 | 0.005 | 0.000 | 1 | hurt |
| gpt_prompt_high_temp | 10 | 0.387 | 0.056 | 0.068 | -0.060 | -1651.700 | -0.011 | 0.000 | 3 | helped |
| gpt_prompt_low_temp | 10 | 0.232 | 0.072 | 0.096 | -0.260 | -1634.400 | 0.092 | 0.000 | 3 | helped |
| new_prompt_with_constrains_60_papers | 60 | 0.444 | 0.001 | -0.022 | -0.190 | -1928.733 | 0.003 | 0.000 | 2 | wash |
| original_prompt_high_temp | 20 | 0.374 | 0.012 | -0.012 | -0.040 | -1109.150 | 0.021 | 0.000 | 2 | wash |
| original_prompt_low_temp | 20 | 0.231 | 0.000 | -0.004 | -0.040 | -1110.750 | 0.053 | 0.000 | 2 | wash |
| original_prompt_original_settings | 60 | 0.347 | 0.015 | -0.007 | -0.043 | -1049.050 | 0.130 | 0.033 | 2 | wash |
| relaxed_references_prompt_high_temp | 10 | 0.288 | -0.031 | -0.020 | -0.220 | -1844.200 | 0.003 | 0.000 | 1 | hurt |
| relaxed_references_prompt_low_temp | 10 | 0.282 | -0.013 | -0.033 | -0.180 | -1500.500 | 0.100 | 0.100 | 1 | hurt |
| tweaked_prompt_high_temp | 20 | 0.400 | 0.000 | -0.038 | -0.160 | -1706.900 | 0.002 | 0.000 | 2 | wash |
| tweaked_prompt_mid_temp | 20 | 0.358 | -0.011 | -0.036 | -0.160 | -1422.400 | 0.110 | 0.100 | 1 | hurt |
| very_big_prompt_high_temp | 40 | 0.427 | 0.013 | -0.006 | -0.130 | -1948.350 | 0.006 | -0.050 | 2 | wash |
| very_big_prompt_very_high_temp | 40 | 0.407 | -0.007 | -0.027 | -0.125 | -1915.775 | -0.001 | -0.050 | 1 | hurt |

The adapter helped on 2 runs, hurt on 5, and was a wash on 6. `d_` columns are adapter minus base. A negative `d_words_mean`, `d_repeat5_mean`, or `d_cap_rate` is an improvement.

## Guardrails

| run | degenerate_papers | meta_papers | cap_rate | eligible | guardrail_failures | eligible_strict | strict_failures |
|---|---|---|---|---|---|---|---|
| a_bit_altered_prompt_60_papers | 0 | 0 | 0.000 | yes |  | yes |  |
| a_bit_altered_prompt_conservative_settings | 7 | 1 | 0.040 | no | degenerate_loop on 7 papers | no | degenerate_loop on 7 papers; degenerate_rate 7.0% > 5% |
| gpt_prompt_high_temp | 0 | 1 | 0.000 | yes |  | no | meta_rate 10.0% > 5% |
| gpt_prompt_low_temp | 2 | 0 | 0.000 | no | degenerate_loop on 2 papers | no | degenerate_loop on 2 papers; degenerate_rate 20.0% > 5% |
| new_prompt_with_conservative_settings | 7 | 0 | 0.020 | no | degenerate_loop on 7 papers | no | degenerate_loop on 7 papers; degenerate_rate 7.0% > 5% |
| new_prompt_with_constrains_60_papers | 0 | 0 | 0.000 | yes |  | yes |  |
| original_prompt_high_temp | 1 | 0 | 0.000 | yes |  | yes |  |
| original_prompt_low_temp | 2 | 0 | 0.000 | no | degenerate_loop on 2 papers | no | degenerate_loop on 2 papers; degenerate_rate 10.0% > 5% |
| original_prompt_original_settings | 15 | 0 | 0.033 | no | degenerate_loop on 15 papers | no | degenerate_loop on 15 papers; degenerate_rate 25.0% > 5% |
| relaxed_references_prompt_high_temp | 0 | 0 | 0.000 | yes |  | yes |  |
| relaxed_references_prompt_low_temp | 2 | 0 | 0.100 | no | degenerate_loop on 2 papers; cap_rate 10.0% > 5% | no | degenerate_loop on 2 papers; cap_rate 10.0% > 5%; degenerate_rate 20.0% > 5% |
| tweaked_prompt_high_temp | 0 | 0 | 0.000 | yes |  | yes |  |
| tweaked_prompt_mid_temp | 4 | 0 | 0.100 | no | degenerate_loop on 4 papers; cap_rate 10.0% > 5% | no | degenerate_loop on 4 papers; cap_rate 10.0% > 5%; degenerate_rate 20.0% > 5% |
| very_big_prompt_high_temp | 0 | 0 | 0.000 | yes |  | yes |  |
| very_big_prompt_very_high_temp | 0 | 0 | 0.000 | yes |  | yes |  |

Thresholds: degenerate loop on 2+ papers, meta-commentary on 2+ papers, cap rate above 5%. A failed guardrail blocks the winner, not the ranking.

`eligible` applies those count rules. `eligible_strict` additionally fails a run when more than 5% of its papers have a degenerate loop or more than 5% contain reviewer meta-commentary. The rate rule is the deciding one: on a 10-paper run a single flagged review is already 10% of the sample. The count-only result is kept so the effect of the threshold is visible.

### Guardrail evidence

The rate rule alone excludes `gpt_prompt_high_temp` (meta rate 10.0%, degenerate rate 0.0%), flagged on `UgNQM-LcVpN`, which the count rule keeps. The flagged papers are listed so they can be read directly. The single flagged review did not contain the matched phrase in its reference review, so the flag is reviewer voice produced by the model, not reference copying.

Reference copying is not the explanation for the embedding scores: `reference_copy12` is the share of a review's 12-word sequences that also occur in its reference, and the highest adapter value is 0.000233 (`original_prompt_original_settings`), so the similarity is stylistic and topical agreement rather than verbatim reuse.

## Component detail

| words_mean | words_median | words_p10 | words_p90 | words_lt150_rate | words_gt800_rate | cap_rate | repeat5_mean | degenerate_rate | meta_rate | reference_copy12 | location_any_rate | emb_ref_cos | emb_ref_cos_all_max | emb_ref_cos_all_mean | n_human_reviews | emb_paper_cos | emb_paper_cos_top3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 342.100 | 339.000 | 193.100 | 504.200 | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | 0.100 | 0.000 | 0.600 | 0.861 | 0.870 | 0.857 | 1.700 | 0.859 | 0.821 |
| 349.450 | 360.500 | 195.400 | 477.300 | 0.000 | 0.000 | 0.000 | 0.021 | 0.050 | 0.000 | 0.000 | 0.600 | 0.851 | 0.858 | 0.845 | 1.650 | 0.835 | 0.808 |
| 413.550 | 331.000 | 223.000 | 580.100 | 0.017 | 0.067 | 0.033 | 0.132 | 0.250 | 0.000 | 0.000 | 0.267 | 0.853 | 0.861 | 0.850 | 1.617 | 0.856 | 0.827 |
| 373.625 | 336.000 | 201.900 | 616.600 | 0.050 | 0.050 | 0.000 | 0.007 | 0.000 | 0.000 | 0.000 | 0.425 | 0.858 | 0.867 | 0.854 | 1.700 | 0.832 | 0.805 |
| 357.150 | 298.000 | 146.500 | 661.500 | 0.100 | 0.050 | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 | 0.650 | 0.851 | 0.854 | 0.839 | 1.650 | 0.829 | 0.800 |
| 280.850 | 215.500 | 155.800 | 496.600 | 0.100 | 0.000 | 0.000 | 0.054 | 0.100 | 0.000 | 0.000 | 0.250 | 0.843 | 0.854 | 0.844 | 1.650 | 0.842 | 0.812 |
| 299.700 | 233.500 | 185.800 | 472.000 | 0.000 | 0.000 | 0.000 | 0.004 | 0.000 | 0.000 | 0.000 | 0.600 | 0.829 | 0.855 | 0.837 | 1.700 | 0.838 | 0.806 |
| 271.100 | 245.000 | 156.100 | 494.100 | 0.100 | 0.000 | 0.000 | 0.112 | 0.200 | 0.000 | 0.000 | 0.000 | 0.858 | 0.868 | 0.857 | 1.700 | 0.855 | 0.821 |
| 389.033 | 366.000 | 212.400 | 598.700 | 0.017 | 0.017 | 0.000 | 0.006 | 0.000 | 0.000 | 0.000 | 0.517 | 0.827 | 0.836 | 0.825 | 1.617 | 0.823 | 0.802 |
| 377.383 | 341.000 | 182.700 | 618.600 | 0.050 | 0.033 | 0.000 | 0.004 | 0.000 | 0.000 | 0.000 | 0.483 | 0.828 | 0.836 | 0.824 | 1.617 | 0.823 | 0.797 |
| 475.925 | 440.500 | 244.400 | 693.500 | 0.000 | 0.050 | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | 0.600 | 0.838 | 0.848 | 0.837 | 1.700 | 0.813 | 0.795 |
| 448.250 | 293.000 | 176.600 | 558.300 | 0.040 | 0.040 | 0.040 | 0.058 | 0.070 | 0.010 | 0.000 | 0.170 | 0.830 | 0.845 | 0.830 | 1.620 | 0.832 | 0.807 |
| 590.600 | 274.000 | 194.900 | 894.400 | 0.000 | 0.100 | 0.100 | 0.102 | 0.200 | 0.000 | 0.000 | 0.200 | 0.839 | 0.851 | 0.835 | 1.700 | 0.832 | 0.801 |
| 587.450 | 297.500 | 198.200 | 969.500 | 0.050 | 0.100 | 0.100 | 0.110 | 0.200 | 0.000 | 0.000 | 0.300 | 0.838 | 0.846 | 0.835 | 1.650 | 0.824 | 0.792 |
| 363.850 | 218.000 | 129.700 | 447.500 | 0.190 | 0.030 | 0.020 | 0.044 | 0.070 | 0.000 | 0.000 | 0.190 | 0.818 | 0.833 | 0.819 | 1.620 | 0.818 | 0.792 |

`emb_ref_cos` compares against the single human review shipped in the run CSV. `emb_ref_cos_all_max` and `emb_ref_cos_all_mean` compare against every human review the dataset holds for that paper (1-4 reviews), and `n_human_reviews` is the average count. The gap between the two shows how much the score depended on which human review was picked.

Matching against the best of all human reviews instead of the shipped one raises a config's `emb_ref_cos` by 0.0107 on average and by at most 0.0260 (`relaxed_references_prompt_high_temp`), so the shipped reference was not an unusually bad match. Ranking by this component alone keeps 6 of 7 configurations, so at most 1 place would change.

## Adapter vs base (paired bootstrap, 10,000 resamples)

| run | metric | n_papers | mean_diff | ci_low | ci_high | win_rate |
|---|---|---|---|---|---|---|
| a_bit_altered_prompt_60_papers | words | 60 | -1338.350 | -1406.950 | -1271.266 | 0.000 |
| a_bit_altered_prompt_60_papers | length_fit | 60 | 0.375 | 0.305 | 0.442 | 0.917 |
| a_bit_altered_prompt_60_papers | emb_ref_cos | 60 | -0.003 | -0.018 | 0.011 | 0.533 |
| a_bit_altered_prompt_60_papers | emb_paper_cos | 60 | -0.014 | -0.027 | -0.002 | 0.400 |
| a_bit_altered_prompt_60_papers | repeat5 | 60 | 0.005 | 0.002 | 0.008 | 0.450 |
| a_bit_altered_prompt_60_papers | structure_score | 60 | -0.113 | -0.200 | -0.030 | 0.317 |
| a_bit_altered_prompt_60_papers | hit_cap | 60 | 0.000 | 0.000 | 0.000 | 0.000 |
| gpt_prompt_high_temp | words | 10 | -1651.700 | -1755.303 | -1558.900 | 0.000 |
| gpt_prompt_high_temp | length_fit | 10 | 0.387 | 0.283 | 0.494 | 1.000 |
| gpt_prompt_high_temp | emb_ref_cos | 10 | 0.056 | 0.023 | 0.086 | 0.900 |
| gpt_prompt_high_temp | emb_paper_cos | 10 | 0.068 | 0.034 | 0.105 | 0.900 |
| gpt_prompt_high_temp | repeat5 | 10 | -0.011 | -0.016 | -0.006 | 0.000 |
| gpt_prompt_high_temp | structure_score | 10 | -0.060 | -0.180 | 0.040 | 0.200 |
| gpt_prompt_high_temp | hit_cap | 10 | 0.000 | 0.000 | 0.000 | 0.000 |
| gpt_prompt_low_temp | words | 10 | -1634.400 | -1766.505 | -1519.500 | 0.000 |
| gpt_prompt_low_temp | length_fit | 10 | 0.232 | 0.113 | 0.354 | 0.800 |
| gpt_prompt_low_temp | emb_ref_cos | 10 | 0.072 | 0.048 | 0.095 | 0.900 |
| gpt_prompt_low_temp | emb_paper_cos | 10 | 0.096 | 0.065 | 0.137 | 1.000 |
| gpt_prompt_low_temp | repeat5 | 10 | 0.092 | 0.038 | 0.161 | 0.900 |
| gpt_prompt_low_temp | structure_score | 10 | -0.260 | -0.340 | -0.180 | 0.000 |
| gpt_prompt_low_temp | hit_cap | 10 | 0.000 | 0.000 | 0.000 | 0.000 |
| new_prompt_with_constrains_60_papers | words | 60 | -1928.733 | -2025.584 | -1834.642 | 0.000 |
| new_prompt_with_constrains_60_papers | length_fit | 60 | 0.444 | 0.379 | 0.507 | 0.967 |
| new_prompt_with_constrains_60_papers | emb_ref_cos | 60 | 0.001 | -0.010 | 0.011 | 0.550 |
| new_prompt_with_constrains_60_papers | emb_paper_cos | 60 | -0.022 | -0.035 | -0.009 | 0.350 |
| new_prompt_with_constrains_60_papers | repeat5 | 60 | 0.003 | 0.000 | 0.006 | 0.300 |
| new_prompt_with_constrains_60_papers | structure_score | 60 | -0.190 | -0.253 | -0.127 | 0.167 |
| new_prompt_with_constrains_60_papers | hit_cap | 60 | 0.000 | 0.000 | 0.000 | 0.000 |
| original_prompt_high_temp | words | 20 | -1109.150 | -1210.451 | -1004.400 | 0.000 |
| original_prompt_high_temp | length_fit | 20 | 0.374 | 0.249 | 0.492 | 0.900 |
| original_prompt_high_temp | emb_ref_cos | 20 | 0.012 | -0.008 | 0.031 | 0.600 |
| original_prompt_high_temp | emb_paper_cos | 20 | -0.012 | -0.031 | 0.007 | 0.450 |
| original_prompt_high_temp | repeat5 | 20 | 0.021 | 0.002 | 0.054 | 0.400 |
| original_prompt_high_temp | structure_score | 20 | -0.040 | -0.200 | 0.110 | 0.450 |
| original_prompt_high_temp | hit_cap | 20 | 0.000 | 0.000 | 0.000 | 0.000 |
| original_prompt_low_temp | words | 20 | -1110.750 | -1201.151 | -1008.741 | 0.000 |
| original_prompt_low_temp | length_fit | 20 | 0.231 | 0.089 | 0.369 | 0.750 |
| original_prompt_low_temp | emb_ref_cos | 20 | 0.000 | -0.016 | 0.017 | 0.400 |
| original_prompt_low_temp | emb_paper_cos | 20 | -0.004 | -0.018 | 0.010 | 0.400 |
| original_prompt_low_temp | repeat5 | 20 | 0.053 | 0.028 | 0.086 | 0.950 |
| original_prompt_low_temp | structure_score | 20 | -0.040 | -0.160 | 0.090 | 0.300 |
| original_prompt_low_temp | hit_cap | 20 | 0.000 | 0.000 | 0.000 | 0.000 |
| original_prompt_original_settings | words | 60 | -1049.050 | -1139.634 | -922.526 | 0.017 |
| original_prompt_original_settings | length_fit | 60 | 0.347 | 0.265 | 0.427 | 0.883 |
| original_prompt_original_settings | emb_ref_cos | 60 | 0.015 | 0.004 | 0.024 | 0.650 |
| original_prompt_original_settings | emb_paper_cos | 60 | -0.007 | -0.017 | 0.004 | 0.467 |
| original_prompt_original_settings | repeat5 | 60 | 0.130 | 0.094 | 0.170 | 0.983 |
| original_prompt_original_settings | structure_score | 60 | -0.043 | -0.133 | 0.043 | 0.350 |
| original_prompt_original_settings | hit_cap | 60 | 0.033 | 0.000 | 0.083 | 0.033 |
| relaxed_references_prompt_high_temp | words | 10 | -1844.200 | -1974.903 | -1690.300 | 0.000 |
| relaxed_references_prompt_high_temp | length_fit | 10 | 0.288 | 0.139 | 0.435 | 0.900 |
| relaxed_references_prompt_high_temp | emb_ref_cos | 10 | -0.031 | -0.070 | 0.001 | 0.400 |
| relaxed_references_prompt_high_temp | emb_paper_cos | 10 | -0.020 | -0.050 | 0.010 | 0.300 |
| relaxed_references_prompt_high_temp | repeat5 | 10 | 0.003 | -0.000 | 0.006 | 0.400 |
| relaxed_references_prompt_high_temp | structure_score | 10 | -0.220 | -0.440 | -0.000 | 0.200 |
| relaxed_references_prompt_high_temp | hit_cap | 10 | 0.000 | 0.000 | 0.000 | 0.000 |
| relaxed_references_prompt_low_temp | words | 10 | -1500.500 | -1898.503 | -805.395 | 0.100 |
| relaxed_references_prompt_low_temp | length_fit | 10 | 0.282 | 0.113 | 0.451 | 0.800 |
| relaxed_references_prompt_low_temp | emb_ref_cos | 10 | -0.013 | -0.031 | 0.005 | 0.400 |
| relaxed_references_prompt_low_temp | emb_paper_cos | 10 | -0.033 | -0.060 | -0.006 | 0.400 |
| relaxed_references_prompt_low_temp | repeat5 | 10 | 0.100 | 0.017 | 0.228 | 1.000 |
| relaxed_references_prompt_low_temp | structure_score | 10 | -0.180 | -0.260 | -0.100 | 0.000 |
| relaxed_references_prompt_low_temp | hit_cap | 10 | 0.100 | 0.000 | 0.300 | 0.100 |
| tweaked_prompt_high_temp | words | 20 | -1706.900 | -1804.450 | -1604.495 | 0.000 |
| tweaked_prompt_high_temp | length_fit | 20 | 0.400 | 0.282 | 0.513 | 0.900 |
| tweaked_prompt_high_temp | emb_ref_cos | 20 | 0.000 | -0.020 | 0.020 | 0.550 |
| tweaked_prompt_high_temp | emb_paper_cos | 20 | -0.038 | -0.063 | -0.015 | 0.250 |
| tweaked_prompt_high_temp | repeat5 | 20 | 0.002 | 0.000 | 0.004 | 0.250 |
| tweaked_prompt_high_temp | structure_score | 20 | -0.160 | -0.270 | -0.050 | 0.200 |
| tweaked_prompt_high_temp | hit_cap | 20 | 0.000 | 0.000 | 0.000 | 0.000 |
| tweaked_prompt_mid_temp | words | 20 | -1422.400 | -1741.501 | -1007.398 | 0.100 |
| tweaked_prompt_mid_temp | length_fit | 20 | 0.358 | 0.237 | 0.471 | 0.800 |
| tweaked_prompt_mid_temp | emb_ref_cos | 20 | -0.011 | -0.031 | 0.007 | 0.400 |
| tweaked_prompt_mid_temp | emb_paper_cos | 20 | -0.036 | -0.059 | -0.016 | 0.250 |
| tweaked_prompt_mid_temp | repeat5 | 20 | 0.110 | 0.046 | 0.197 | 0.850 |
| tweaked_prompt_mid_temp | structure_score | 20 | -0.160 | -0.300 | -0.030 | 0.250 |
| tweaked_prompt_mid_temp | hit_cap | 20 | 0.100 | 0.000 | 0.250 | 0.100 |
| very_big_prompt_high_temp | words | 40 | -1948.350 | -2022.602 | -1874.822 | 0.000 |
| very_big_prompt_high_temp | length_fit | 40 | 0.427 | 0.357 | 0.496 | 0.975 |
| very_big_prompt_high_temp | emb_ref_cos | 40 | 0.013 | 0.001 | 0.024 | 0.575 |
| very_big_prompt_high_temp | emb_paper_cos | 40 | -0.006 | -0.020 | 0.008 | 0.400 |
| very_big_prompt_high_temp | repeat5 | 40 | 0.006 | 0.003 | 0.009 | 0.500 |
| very_big_prompt_high_temp | structure_score | 40 | -0.130 | -0.200 | -0.060 | 0.200 |
| very_big_prompt_high_temp | hit_cap | 40 | -0.050 | -0.125 | 0.000 | 0.000 |
| very_big_prompt_very_high_temp | words | 40 | -1915.775 | -2006.734 | -1820.419 | 0.000 |
| very_big_prompt_very_high_temp | length_fit | 40 | 0.407 | 0.335 | 0.478 | 0.925 |
| very_big_prompt_very_high_temp | emb_ref_cos | 40 | -0.007 | -0.020 | 0.004 | 0.450 |
| very_big_prompt_very_high_temp | emb_paper_cos | 40 | -0.027 | -0.041 | -0.013 | 0.250 |
| very_big_prompt_very_high_temp | repeat5 | 40 | -0.001 | -0.001 | 0.000 | 0.150 |
| very_big_prompt_very_high_temp | structure_score | 40 | -0.125 | -0.230 | -0.020 | 0.375 |
| very_big_prompt_very_high_temp | hit_cap | 40 | -0.050 | -0.125 | 0.000 | 0.000 |

`mean_diff` is adapter minus base; `win_rate` is the share of papers where the adapter value is larger. For `repeat5` and `hit_cap` lower is better, so a negative difference is favourable.

## Phase 2 screening set (top 7 adapters by local rank)

| rank | run | n_papers | eligible_strict | composite_rank |
|---|---|---|---|---|
| 1 | gpt_prompt_high_temp | 10 | no | 1.250 |
| 2 | original_prompt_high_temp | 20 | yes | 3.500 |
| 3 | original_prompt_original_settings | 60 | no | 4.500 |
| 4 | very_big_prompt_high_temp | 40 | yes | 6.250 |
| 5 | tweaked_prompt_high_temp | 20 | yes | 6.500 |
| 6 | original_prompt_low_temp | 20 | no | 8.250 |
| 7 | relaxed_references_prompt_high_temp | 10 | yes | 8.250 |

These seven are the configs a judge would score next. Guardrail status is shown but does not remove a config, per the user's instruction.

## Confound summary

- Pairs compared: 105; identical: 0; single axis: 9; package effect: 96.
- Single-axis pairs (clean comparisons):
  - `a_bit_altered_prompt_60_papers` vs `new_prompt_with_constrains_60_papers`: `prompt_family`
  - `a_bit_altered_prompt_conservative_settings` vs `new_prompt_with_conservative_settings`: `prompt_family`
  - `gpt_prompt_high_temp` vs `gpt_prompt_low_temp`: `temperature`
  - `gpt_prompt_high_temp` vs `relaxed_references_prompt_high_temp`: `prompt_family`
  - `gpt_prompt_low_temp` vs `relaxed_references_prompt_low_temp`: `prompt_family`
  - `original_prompt_high_temp` vs `original_prompt_low_temp`: `temperature`
  - `original_prompt_high_temp` vs `tweaked_prompt_high_temp`: `prompt_family`
  - `relaxed_references_prompt_high_temp` vs `relaxed_references_prompt_low_temp`: `temperature`
  - `tweaked_prompt_high_temp` vs `tweaked_prompt_mid_temp`: `temperature`

## Limitations

- The 10 shared papers were already used during prompt development, so this is a dev-set comparison, not a held-out test.
- Sample sizes differ (10 to 100 papers) and two 100-paper runs are adapter-only, so cross-run comparisons are unequal.
- Embedding cosine is a proxy for topical similarity and grounding, not proof of factual correctness.
- ROUGE rewards word overlap and can reward copying or repetition; it is diagnostic only and never a tie-breaker.
- No human study was run. Any Phase 2 judge is an LLM proxy, not human reviewer judgement.
- Each paper has 1-4 human reviews, but the run CSVs ship only one selected reference. `emb_ref_cos` therefore depends on which human review was picked; the all-reference columns show how much.
- Adapter and base are ranked separately. The base leaderboard is useful for prompt choice without fine-tuning, not as a competitor to the adapter.
- The rate-based guardrail was added during Phase 1 after a 10-paper run showed that one flagged review is already 10% of the sample. The pre-registered count-rule result is kept in the `eligible` column, so the effect of the threshold is visible.
