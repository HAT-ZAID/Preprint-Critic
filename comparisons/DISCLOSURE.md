# Disclosure: how the evaluation was run

This file is the public record for the experiment comparison, kept open as a pull
request rather than merged, so the method, the limits, and the open questions stay
visible against the code that produced them.

## What was compared

Fifteen generation configurations of the same LoRA adapter on the same base model
(`Qwen/Qwen3-4B-Instruct-2507`), on the same archived papers. No training ran for
this comparison and no reviews were generated for it. Every review scored here
already existed.

## How it was scored

Round one used only local measures over the 960 archived reviews: length, repetition
and loop detection, structure and location references, embedding similarity to the
human review, and embedding grounding against the paper. Embeddings came from
`nomic-embed-text` running locally through Ollama, so round one cost no API calls.

Round two used a rubric-fixed LLM judge. Each call received the paper text and one
review, with the configuration name, arm, prompt and decoding settings hidden. The
judge returned a 0 to 100 usefulness score, seven subfield scores, and binary flags
for invented locations, generic filler, meta-commentary, and degenerate repetition.
Pairwise calls asked which of two reviews was more useful, with A/B order randomised
and a sample of repeats run with the sides swapped to check order sensitivity.

## What is not claimed

The judge is an LLM proxy, not a peer-review or human-subject study. Its absolute
scores are not a publication standard. Paper counts differ between configurations
because runs generated different numbers of papers, and the ten papers shared by
every run are development papers that were already used when the prompts were
chosen. Paired differences between the leading adapters fall inside the measured
noise band, so no adapter ordering here is statistically separable.

## What is deliberately excluded

The per-review metrics table (about 10 MB) and the raw judge log (about 0.9 MB) are
not published in this repository. They are regenerable from the scripts in
`experiments/` and the archived run files, and they are available on request. Their
summaries are committed under `experiments/results/`.

## Reproducing it

```
python -m experiments.analysis.load_runs
python -m experiments.analysis.analyze_phase1       # local metrics and ranking
python -m experiments.analysis.judge --mode all     # the judge round
python -m experiments.analysis.analyze_phase2       # judge scores and pairwise
python -m experiments.analysis.final_pick           # the shipped configuration
python -m experiments.analysis.verify_results       # artifact checks
```

## Open questions

- Reviews stop near 374 words, so coverage and argument depth stay thin.
- Whole-review judging may understate a section by section comparison, which has
  not been tested.
- Grounding and faithfulness at longer output lengths are unmeasured.
