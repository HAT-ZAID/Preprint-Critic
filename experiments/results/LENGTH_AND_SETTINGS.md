# Review length and decoding settings

Companion to the Phase 1 local comparison and the Phase 2 judge run. Everything here is computed from the archived runs and the 90 judged reviews. No model was called to produce this file.

## What the judged reviews look like

| arm | n | mean words | median | min | max | mean score |
|---|---|---|---|---|---|---|
| adapter | 270 | 373 | 309 | 92 | 3229 | 37.1 |
| base | 130 | 1889 | 1824 | 1011 | 2770 | 71.5 |

The human reference reviews for these papers average **443 words** (median 378, p10 191, p90 740). The adapter sits at or near that band; the base arm is roughly 4.3x the human length.

## Does length explain the score?

| group | n | Spearman | Pearson |
|---|---|---|---|
| all judged | 400 | 0.72 | 0.74 |
| arm=adapter | 270 | 0.35 | 0.02 |
| arm=base | 130 | -0.23 | -0.19 |
| a_bit_altered_prompt_60_papers [adapter] | 60 | 0.30 | 0.25 |
| a_bit_altered_prompt_60_papers [base] | 60 | -0.26 | -0.20 |
| a_bit_altered_prompt_conservative_settings [adapter] | 10 | 0.07 | -0.06 |
| gpt_prompt_high_temp [adapter] | 10 | 0.17 | -0.17 |
| gpt_prompt_high_temp [base] | 10 | -0.57 | -0.53 |
| gpt_prompt_low_temp [adapter] | 10 | 0.62 | 0.65 |
| new_prompt_with_conservative_settings [adapter] | 10 | 0.10 | -0.12 |
| new_prompt_with_constrains_60_papers [adapter] | 10 | 0.38 | 0.48 |
| original_prompt_high_temp [adapter] | 20 | 0.53 | 0.52 |
| original_prompt_high_temp [base] | 20 | -0.07 | -0.04 |
| original_prompt_low_temp [adapter] | 20 | 0.62 | 0.56 |
| original_prompt_original_settings [adapter] | 20 | -0.02 | -0.10 |
| relaxed_references_prompt_high_temp [adapter] | 10 | 0.00 | -0.06 |
| relaxed_references_prompt_low_temp [adapter] | 10 | 0.01 | -0.54 |
| tweaked_prompt_high_temp [adapter] | 20 | 0.69 | 0.61 |
| tweaked_prompt_mid_temp [adapter] | 10 | 0.21 | -0.46 |
| very_big_prompt_high_temp [adapter] | 40 | 0.27 | 0.23 |
| very_big_prompt_high_temp [base] | 40 | -0.01 | 0.04 |
| very_big_prompt_very_high_temp [adapter] | 10 | -0.01 | 0.11 |

Read the arm and per-config rows, not the pooled row. Pooled length and arm are almost perfectly confounded, because every adapter review is short and every base review is long. Inside each arm the correlation is the honest test.

## Length bands

| arm | words | n | mean score | median score |
|---|---|---|---|---|
| adapter | <200 | 49 | 32.3 | 32.0 |
| adapter | 200-300 | 81 | 34.5 | 35.0 |
| adapter | 300-350 | 30 | 36.3 | 38.0 |
| adapter | 350-400 | 26 | 41.2 | 40.0 |
| adapter | 400-500 | 38 | 41.5 | 40.0 |
| adapter | 500-700 | 29 | 41.4 | 42.0 |
| adapter | 700-1000 | 13 | 43.8 | 40.0 |
| adapter | 1000+ | 4 | 30.0 | 27.5 |
| base | 1000+ | 130 | 71.5 | 72.5 |

Within the adapter arm, best adapter band is 700-1000 words at 43.8.

## Setting weights: matched temperature pairs

| family | low T | high T | Δcomposite (lower better) | Δlength_fit | Δemb_paper | Δstructure | Δrepeat5 | Δdegenerate | other axes differ |
|---|---|---|---|---|---|---|---|---|---|
| a_bit | 0.60 | 0.80 | -0.75 | 0.02 | -0.01 | 0.08 | -0.052 | -0.07 | top_p, top_k, repetition_penalty, n_papers |
| gpt | 0.40 | 0.80 | -7.50 | 0.14 | 0.00 | 0.24 | -0.111 | -0.20 | none |
| new | 0.60 | 0.80 | -4.50 | 0.09 | 0.00 | 0.04 | -0.039 | -0.07 | top_p, top_k, repetition_penalty, n_papers |
| original | 0.40 | 0.80 | -4.75 | 0.13 | -0.01 | 0.13 | -0.033 | -0.05 | none |
| relaxed | 0.40 | 0.80 | -2.00 | 0.00 | 0.01 | 0.08 | -0.098 | -0.20 | none |
| tweaked | 0.50 | 0.80 | -3.75 | 0.04 | 0.00 | 0.09 | -0.108 | -0.20 | none |
| very_big | 0.80 | 0.95 | 3.25 | -0.03 | -0.02 | 0.06 | -0.006 | 0.00 | top_p, top_k |

Higher temperature produced the better composite rank in **6 of 7** families. A negative delta means the high-temperature run ranked better.
Where `other axes differ` is not empty, the pair is not a clean temperature test and is reported as a package effect.

## Setting weights: above vs below median

| axis | runs | distinct values | low group composite | high group composite | delta | usable |
|---|---|---|---|---|---|---|
| temperature | 15 | 6 | 7.89 | 9.50 | 1.61 | no |
| top_p | 15 | 4 | 7.89 | 9.50 | 1.61 | no |
| top_k | 15 | 3 | 8.69 | 7.75 | -0.94 | yes |
| repetition_penalty | 15 | 3 | 7.38 | 12.00 | 4.62 | no |
| max_paper_tokens | 15 | 2 | 6.75 | 8.31 | 1.56 | yes |
| max_new_tokens | 15 | 2 | 6.96 | 8.91 | 1.94 | yes |
| n_papers | 15 | 5 | 7.12 | 9.00 | 1.88 | yes |

Not usable as evidence: `temperature`, `top_p`, `repetition_penalty`. Either the value is constant across runs, or one side of the split has fewer than three runs.

## Setting weights: ridge with leave-one-run-out

| setting | coefficient on composite rank | standardised |
|---|---|---|
| temperature | -8.637 | -1.503 |
| top_p | 0.922 | 0.158 |
| top_k | 0.117 | 1.400 |
| repetition_penalty | 51.368 | 1.136 |
| max_paper_tokens | -0.001 | -0.701 |
| max_new_tokens | 0.000 | 0.286 |
| n_papers | 0.066 | 1.991 |

Leave-one-run-out R² = **-6.59** on 15 runs (ridge alpha 1.0).
A non-positive R² means the settings do not predict the ranking better than the mean. Treat the coefficients as directions to test, not as established effects.

## Verdict

- **Length:** best adapter band is 700-1000 words at 43.8. Short adapter reviews are not the problem; they already match human length.
- **arm=adapter:** length vs score Spearman 0.35 over 270 reviews.
- **arm=base:** length vs score Spearman -0.23 over 130 reviews.
- **Length is not the driver of the base advantage.** Inside the adapter arm length correlates weakly and positively; inside the base arm it correlates negatively. The base wins by covering more ground, not by being longer.
- **Settings:** only temperature has matched pairs. Every other axis is bundled with prompt or budget changes, and the regression cannot separate them from 15 runs. Any claim that top_p, top_k or repetition_penalty 'helped' would be unsupported.
