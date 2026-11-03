# Findings

Fifteen generation configurations were compared on the same adapter, base model and test papers. No new training and no new review generation was done; every review already existed.

## Round 1: local metrics, no LLM

960 reviews across 15 runs were scored with lexical metrics and local `nomic-embed-text` embeddings. This ranked configurations on length fit, similarity to the human review, grounding in the paper and structure.

## Round 2: LLM judge

A rubric-fixed judge scored review usefulness out of 100, with flags for invented locations, filler, role-play and degenerate repetition. Coverage was widened until the two finalists were judged on every paper they generated, on both adapter and base arms (60 and 40 papers).

## What the comparison found

| Arm | Best judged usefulness | Notes |
|---|---|---|
| Adapter | 42.1 (20 papers) | fails the role-play gate: 9 of 20 reviews |
| Adapter (chosen) | 40.2 (40 papers) | only configuration passing both gate families |
| Base model | 73.7 (20 papers) | 73.6 and 72.2 on longer runs, 71.4 and 55.1 on others |

The adapter wins on efficiency, not judged quality. It produces 374 words instead of roughly 2,000, at 81.2 usefulness points per 1k generated tokens against 21.2, and it sits closer to the human review in embedding space (0.858 against 0.846). It loses on coverage and argument depth, and it loses every head-to-head against its own base model.

## Why the adapter writes short reviews

The training reviews are short. The median human review in the Re² subset is 345 words and the 75th percentile is 517. The adapter learned that house style faithfully, which is also why its coverage is thinner than the base model's multi-page output.

## Chosen configuration

See `comparisons/prompt.txt` and `comparisons/settings.json`. Gap table, per-run scores and the full protocol are in `experiments/results/`.
