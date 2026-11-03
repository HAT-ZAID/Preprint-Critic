# Final configuration for the Kaggle run

**Run `very_big_prompt_high_temp` with the archived adapter, using the settings below.**

Selection rule: clearest mean usefulness (40.2 on 40 papers) and passes every gate.

## Settings to reproduce

- `prompt_file`: **D:\Preprint Critic\preprint_critic_review_experiments\very_big_prompt.txt**
- `prompt_chars`: **8673**
- `temperature`: **0.8**
- `top_p`: **0.9**
- `top_k`: **40**
- `repetition_penalty`: **1.15**
- `no_repeat_ngram_size`: **0**
- `max_paper_tokens`: **16050**
- `max_new_tokens`: **4096**

Prompt opening as archived:

```text
user_text = f"""You are an independent academic peer reviewer evaluating the research manuscript below.

Write a substantive, fair, constructive, technically informed review of the submitted manuscript. Your purpose is to evaluate the actual research, identify its meaningful strengths and weaknesses, assess how well its evidence supports its claims, and provide useful feedback.

GROUNDEDNESS IS THE GOVERNING PRINCIPLE OF THE ENTIRE REVIEW.

Every substantive judgment must be grounded in the supplied manuscript. Factual accuracy, faithful interpretation, and evidential defensibility take preced
```

## Candidate table

Judge score basis: **all judged papers**.

| config | judge overall | papers | pts / 1k out | paper cosine | length fit | words | gates |
|---|---|---|---|---|---|---|---|
| original_prompt_high_temp | 42.1 | 20 | 92.4 | 0.835 | 0.664 | 349 | fail (judge_meta=45%>25%) |
| very_big_prompt_high_temp | 40.2 | 40 | 81.2 | 0.832 | 0.609 | 374 | pass |
| a_bit_altered_prompt_60_papers | 38.6 | 60 | 76.2 | 0.823 | 0.613 | 389 | fail (judge_invented=27%>25%; judge_meta=37%>25%) |
| new_prompt_with_constrains_60_papers | 38.5 | 60 | 77.5 | 0.823 | 0.621 | 377 | fail (judge_invented=30%>25%; judge_meta=43%>25%) |
| tweaked_prompt_mid_temp | 37.3 | 10 | 44.6 | 0.824 | 0.565 | 587 | fail (degenerate_rate=0.200>0.05; cap_rate=0.100>0.05) |
| tweaked_prompt_high_temp | 37.0 | 20 | 77.1 | 0.829 | 0.603 | 357 | fail (judge_invented=45%>25%; judge_meta=45%>25%) |
| original_prompt_original_settings | 36.5 | 20 | 56.8 | 0.856 | 0.628 | 414 | fail (degenerate_rate=0.250>0.05; judge_degenerate=35%>25%) |
| gpt_prompt_high_temp | 35.5 | 10 | 79.4 | 0.859 | 0.652 | 342 | fail (meta_rate=0.100>0.05; judge_invented=50%>25%; judge_meta=40%>25%) |
| original_prompt_low_temp | 35.2 | 20 | 94.3 | 0.842 | 0.537 | 281 | fail (degenerate_rate=0.100>0.05) |
| a_bit_altered_prompt_conservative_settings | 35.0 | 20 | 51.1 | 0.832 | 0.589 | 448 | fail (degenerate_rate=0.070>0.05) |
| relaxed_references_prompt_low_temp | 34.4 | 10 | 35.7 | 0.832 | 0.536 | 591 | fail (degenerate_rate=0.200>0.05; cap_rate=0.100>0.05; judge_degenerate=30%>25%) |
| relaxed_references_prompt_high_temp | 33.9 | 10 | 87.1 | 0.838 | 0.540 | 300 | fail (judge_meta=60%>25%) |
| gpt_prompt_low_temp | 31.2 | 10 | 85.0 | 0.855 | 0.514 | 271 | fail (degenerate_rate=0.200>0.05; judge_degenerate=30%>25%) |
| new_prompt_with_conservative_settings | 30.1 | 20 | 88.1 | 0.818 | 0.532 | 364 | fail (degenerate_rate=0.070>0.05) |
| very_big_prompt_very_high_temp | 29.0 | 10 | 58.5 | 0.813 | 0.583 | 476 | fail (judge_invented=50%>25%; judge_meta=40%>25%) |

## Quality-first alternative

If judged usefulness is weighted above cost, the pick is `original_prompt_high_temp` at 42.1 judged points (349 words on average, prompt 1232 characters against 8673 for `very_big_prompt_high_temp`). The two are inside the 5-point noise band, so this is a trade-off, not a missed winner.

## Head-to-head record

- against gpt_prompt_high_temp (adapter): won 4, lost 6 of 10
- against original_prompt_high_temp (adapter): won 8, lost 12 of 20
- against original_prompt_original_settings (adapter): won 11, lost 9 of 20

## Evidence that disagrees

These configurations scored higher than the pick on a judged sample. Treat the differences as unresolved: the paired standard error on ten papers is about 4 points, so these are ties broken by the declared rule rather than wins.

- `original_prompt_high_temp` scores 42.1 on 20 judged papers (the pick scores 40.2 on the same or fewer)

## What this pick does not claim

- The judge is an LLM proxy, not a human review study. It is a consistent, rubric-fixed judge, but its absolute scores are not a peer-review standard.
- Configurations inside the noise band were separated by a paired comparison on the widest shared paper set, or by token efficiency when that comparison was also inside the band. Neither is a demonstrated quality difference.
- The fine-tuned adapter is still weaker than the untuned base model on judged usefulness. This pick is the best *adapter* configuration, which is what the Kaggle run needs; it is not a claim that fine-tuning beat the base model.

