---
base_model: Qwen/Qwen3-4B-Instruct-2507
library_name: peft
pipeline_tag: text-generation
language: en
license: apache-2.0
datasets:
  - Daoze/ReviewRebuttal
tags:
  - lora
  - peft
  - qwen3
  - peer-review
  - text-generation
  - scientific-papers
---

# Preprint Critic V3

## 1. Model summary

A LoRA adapter for `Qwen/Qwen3-4B-Instruct-2507` that generates one peer review from a paper's full text.

| | |
|---|---|
| Base model | `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0), 4,038,983,168 parameters |
| Adapter | LoRA, `r=8`, `lora_alpha=16`, `lora_dropout=0.05` |
| Target modules | q, k, v, o, gate, up, down projections |
| Adapter size | 504 tensors, all float32, 16,515,072 trainable parameters (0.4089% of the base), 63 MiB |
| Task | one peer review per paper, conditioned on an instruction brief followed by the paper text |
| Language | English |
| License | Apache-2.0, inherited from the base model |
| Adapter state | `inference_mode: true`; weights are byte-identical to `Checkpoint/checkpoint-350` |

The adapter is distributed unmerged. There is no merged checkpoint in the repository.

## 2. Intended use and out-of-scope use

**Intended use**

- Research on peer-review generation and on parameter-efficient adapters for long-context scientific text.
- Drafting a review that a human then checks line by line against the paper.
- A comparison point for other review-generation systems evaluated on Re².

**Out-of-scope use**

- Accept/reject recommendations, editorial decisions, or any decision about authors.
- Judging competence, integrity, or misconduct.
- Fact verification. The adapter cannot establish that a claim is true; it only generates text conditioned on the paper.
- Venues, languages, or paper types outside the Re² OpenReview sample. Behavior there was not measured.
- Papers longer than the prompt budget. Training capped paper text at 16,050 tokens; longer inputs are truncated or fall outside the trained distribution.
- Any workflow where a generated review is submitted without a human verifying every criticism against the paper.

## 3. Prompt format

The prompt is a single user message: a five-paragraph instruction brief, then `Paper:` and the paper
text. The brief asks for a substantive, fair review; requires claims to be grounded in the manuscript
and forbids invented results, citations, numbers, and section/table/figure references; asks for the
most consequential issues with actionable questions; names the dimensions to assess (novelty,
methodological validity, baseline adequacy, limitations, reproducibility); and asks the model to
avoid boilerplate, fixed templates, and unsupported verdicts. The notebook's `prompt_for` is the
source of truth; copy it verbatim rather than paraphrasing the brief.

The paper is context. Loss is masked to the review completion only, so the model is trained to
produce the review and not to reproduce the paper.

The prompt is rendered with the base tokenizer's chat template and `add_generation_prompt=True`; the
instruction brief plus `Paper:\n` plus the paper text is the user message. Padding token id is
151643, EOS is 151645, and generation is capped at 16,384 new tokens.

## 4. Training data and preprocessing

**Dataset.** `Daoze/ReviewRebuttal` (Re², Apache-2.0, [arXiv:2505.07920](https://arxiv.org/abs/2505.07920)): 2,000 OpenReview papers with human reviews.

**Filtering and split.** Reviews shorter than 100 characters were removed, leaving 7,040
paper-review rows. Papers were assigned 80/10/10 by `paper_id` (1,600 / 200 / 200), then each split
was shuffled and capped at 2,800 / 280 / 280 rows. The caps discard 3,680 of the 7,040 eligible
rows; the retained rows cover 1,439 / 163 / 172 papers, so 161 / 37 / 28 selected papers contribute
no row. In particular, the 280-row test cap, not the 100-character filter, is why 28 of the 200 test
papers are absent: every test paper has at least one review that passes the filter. A row-level
split would place reviews of the same paper on both sides of the boundary. Leakage check: zero of
the 172 retained test papers appear in train.

**Preprocessing.** `strip_refs_and_appendix` cut the text at the first references or bibliography
heading in 1,968 of 2,000 papers (32 had no matching heading), reducing paper text by 35.5% on
average. Because it truncates at that heading, anything after it was removed, including an appendix;
no standalone appendix heading was detected in the sample. Light text cleanup was applied. Paper text
was capped at 16,050 tokens, the review target at 1,750 tokens, for a total window of 18,432.

**Truncation.** Paper, review, and window truncation were counted separately, so a paper hitting its own cap is not confused with a sequence that overflowed the window.

| Split | Rows | Paper truncated at 16,050 | Review truncated at 1,750 | Window truncated at 18,432 |
|---|---:|---:|---:|---:|
| Train | 2,800 | 198 | 36 | 0 |
| Validation | 280 | 17 | 6 | 0 |
| Test | 280 | 27 | 2 | 0 |

Train row length (paper + review) percentiles, p10 / p50 / p90 / p100: 8,038 / 12,110 / 16,277 / 18,040 tokens. The longest row still fits the 18,432 window; the per-paper budget is what truncates, and the counts above are the rows affected.

**Review schema.** A census of all 7,040 rows found heterogeneous structure. These categories overlap, so they do not sum to 100%:

| Schema feature | Share of rows |
|---|---:|
| None of the five section keys | 60.6% |
| `main_review` | 23.4% |
| Full four-key template | 16.0% |
| `summary_of_the_paper` | 26.9% |

The five keys counted above are the four Re² sections — `summary_of_the_paper`, `strength_and_weaknesses`, `clarity,_quality,_novelty_and_reproducibility`, `summary_of_the_review` — plus `main_review`. Only 16.0% of rows carry all four Re² sections, and 60.6% carry none of the five, so the adapter sees a fully structured review in a minority of examples.

## 5. Training procedure

- 1 epoch, 350 optimizer steps.
- Per-device batch 1 with gradient accumulation 8, effective batch 8.
- Learning rate 2e-4 with linear decay and 0 warmup.
- bf16 compute and tf32.
- `adamw_torch_fused`.
- Base weights loaded in 4-bit NF4 with double quantization; bf16 compute; gradient checkpointing on.
- Liger kernel was installed and used. No benefit was measured, so this card claims no speedup from it.
- Single A100-SXM4-40GB. Peak 29.13 GiB allocated / 30.58 GiB reserved, measured after
  `reset_peak_memory_stats()` in the resumed session (steps 241–350); the first session's peak was
  not retained. Total 7.46e17 FLOPs.
- The run spanned two sessions and resumed from `checkpoint-240` via `get_last_checkpoint`; `load_best_model_at_end=True`.

Two loss numbers need care. The logged `train_loss` of 2.1123 covers only steps 241–350, so it is not an epoch average. The step-1 logged loss of 2.5437 is over 8 training rows and is not a validation number.

## 6. Evaluation

### Validation and test loss

Validation ran every 80 steps.

| Step | 80 | 160 | 240 | 320 | 350 |
|---|---:|---:|---:|---:|---:|
| Validation loss | 2.2040 | 2.1824 | 2.1725 | 2.1661 | 2.1653 |

Best = final = 2.1653 at step 350. Held-out test loss is 2.1518. Test is slightly below validation, but this is a single 280-row split; do not over-read that gap as a generalization result.

These are teacher-forced next-token losses on human reviews, with the paper as context. They measure how well the model reproduces a review under the training objective, not whether the review is correct, grounded, or useful. A matched base-model loss on the same splits was not measured.

### 10-sample pilot

This is a pilot, not a benchmark. It uses the same prompts and decoding settings with the adapter on and off, over 10 held-out papers.

| Measure | Base | Adapter | Human reference |
|---|---:|---:|---:|
| Mean words | 1,421 | 244 | 412 |
| ROUGE-L vs human reference (mean) | 0.123 | 0.175 | - |
| ROUGE-L range | 0.085–0.140 | 0.120–0.221 | - |
| Opens with `summary_of_the_paper:` | 0/10 | 6/10 | 5/10 |
| Full Re² four-section template | 0/10 | 6/10 | 3/10 |
| Section/Table/Figure citations (grounded) | 24 (20) | 3 (1) | 20 (16) |
| Mean 5-gram repetition | 0.2% | 6.5% (max 30%) | 0.6% |

"Grounded" means the cited section, table, or figure appears in the paper. The adapter emits the full four-section Re² template in 6 of 10 samples; the base model does so in none, and the human references in 3. All six adapter samples that use the full template open with `summary_of_the_paper:`.

The adapter learned the shape of the dataset's reviews, not yet the evidence. It wins on ROUGE-L in 9 of 10 papers and is better formatted than the base model's output, but it cites the paper less (3 citations, 1 grounded, against 24 and 20 for the base) and repeats paragraphs in one sample (30% 5-gram repetition). ROUGE-L rewards both length and template overlap, so the ROUGE-L gain is not evidence that the reviews are better. The pilot has 10 samples, no human evaluation, and no significance testing.

**Token-count caveat.** The `output_tokens` fields in the committed JSON were computed from padded batch width, so they repeat within a batch; they are not per-sample lengths. Re-tokenized lengths are in `comparison/metrics_v3a.json`, computed from the per-sample records in `comparison/comparison_v3a.json`.

## 7. Limitations and failure modes

- **Evidence, not just format.** The pilot's central failure is grounding: the adapter produced 3 Section/Table/Figure citations, 1 of them grounded. A review can be well formatted and still assert things the paper does not contain.
- **Length.** At 244 mean words the adapter is shorter than both the base model (1,421) and the human reference (412). The shorter output can omit required detail; ROUGE-L is length-sensitive and rewards template overlap, so the pilot's ROUGE-L gain does not establish better coverage.
- **Loss is not quality.** Validation and test loss are teacher-forced losses on human reviews. They do not measure factual accuracy, coverage, or usefulness, and no human evaluation was run.
- **No matched baseline loss.** The base model was not evaluated for loss on the same validation and test rows, so the loss numbers have no like-for-like before/after comparison.
- **Small pilot, one split.** The generation evidence is 10 papers with no significance testing. The test loss comes from a single 280-row split.
- **Truncation is real.** 198 of the 2,800 training rows and 27 of the 280 test rows hit the
  16,050-token paper cap; 36 training rows and 2 test rows hit the 1,750-token target cap. The effect
  of truncation on review quality was not measured.
- **Weak formatting prior.** 62.4% of the 2,800 training rows carry none of the five section keys
  and 15.4% carry all four Re² sections. Template-following is learned from a minority of the data:
  6 of 10 pilot outputs used the full four-section template, and four did not.
- **Domain and language.** Training and evaluation are English OpenReview papers in Re². Other venues, languages, and paper structures were not measured.
- **No calibration or abstention.** The adapter has no mechanism to decline when the paper is out of domain, truncated, or unreadable, and no uncertainty estimate is provided.
- **Repetition.** Mean 5-gram repetition is 6.5% for the adapter, 0.2% for the base model, and 0.6% for the human references. One adapter sample reaches 30% by repeating whole paragraphs between its strengths/weaknesses and clarity sections. Seven of ten adapter samples are at or below 3.2%.

## 8. Bias, risks, and responsible use

The training targets are human reviews from OpenReview. The adapter imitates the tone, length, formatting conventions, and judgment patterns of that reviewer population, including any venue-specific or demographic biases present in those reviews. No bias evaluation was run, so the direction and size of any inherited bias are unknown.

Generated criticism can be wrong, overly broad, or unsupported by the paper. The pilot's grounding count is the available evidence for this risk, and it is weak. A plausible-sounding review may still misrepresent the paper, and a reader may over-trust it because it is fluent and well formatted.

Responsible use:

- Keep a human in the loop who verifies every claim against the paper.
- Disclose model assistance where the venue or employer requires it, and follow the venue's policy on generated reviews.
- Do not use the adapter to produce review spam, to target authors, or as input to acceptance decisions.
- Do not treat the adapter's silence about a flaw as evidence that the flaw is absent.
- Check the licenses and terms that apply to the input paper and to the Re² dataset before redistributing outputs.

The adapter inherits the base model's Apache-2.0 license. The dataset is Apache-2.0. No privacy audit of paper text or reviews was performed.

## 9. How to use

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER = "rc_project_adapter_V3"

tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

INSTRUCTIONS = """You are an independent academic peer reviewer.

Write a substantive, fair, constructive review of the submitted
manuscript. Ground factual claims in the manuscript. Never invent
results, citations, numerical values, or section, table, or figure
references that are not present.

Prioritize the most consequential issues rather than listing generic
objections. For each major concern, identify the relevant claim,
method, or experiment; explain the evidence gap or reasoning behind
the concern and why it matters; and give an actionable question or
suggestion where appropriate.

Assess the contribution and novelty in the context of related work
actually discussed in the manuscript, methodological validity,
whether experiments support the conclusions, baseline adequacy,
limitations, and reproducibility where assessable.

Distinguish demonstrated problems from questions where the evidence
is insufficient. Avoid boilerplate praise, repeated summaries, and
criticisms that could apply to almost any paper. Use natural
peer-review prose. Use headings or bullets when helpful, but do not
force a fixed template or add a verdict unsupported by the evidence.
Be concise without sacrificing substantive, paper-specific analysis."""

paper = open("paper.txt", encoding="utf-8").read()
messages = [{"role": "user", "content": INSTRUCTIONS + "\n\nPaper:\n" + paper}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    output = model.generate(
        inputs,
        max_new_tokens=16384,
        do_sample=False,
        pad_token_id=151643,
        eos_token_id=151645,
    )

print(tokenizer.decode(output[0, inputs.shape[1]:], skip_special_tokens=True))
```

The snippet loads the base in bf16; it does not reproduce the 4-bit training configuration. It uses greedy decoding for determinism; the pilot sampled with temperature 0.7, top-p 0.8, top-k 20, and repetition penalty 1.1. `adapter_config.json` ships `inference_mode: true`, so a plain load freezes the LoRA weights. To continue training, pass `is_trainable=True` to `PeftModel.from_pretrained`. Truncate paper text to 16,050 tokens and put the instruction brief first; the notebook's `prompt_for` is the source of truth.

## 10. Reproduction notes

**In the repository**

| Path | Contents |
|---|---|
| `rc_project_adapter_V3/adapter_model.safetensors` | 63 MiB adapter, 504 float32 tensors |
| `rc_project_adapter_V3/adapter_config.json` | LoRA configuration and base model id |
| `rc_project_adapter_V3/trainer_state.json` | Loss and evaluation history for the 350-step run |
| `rc_project_adapter_V3/training_args.bin` | Resolved `TrainingArguments` |
| `rc_project_adapter_V3/tokenizer.json` | Tokenizer files |
| `comparison/comparison_v3a.json` | 10-sample paired pilot outputs |
| `comparison/baseline_results_v3a.json` | Base-model pilot outputs |
| `comparison/finetune_results_v3a.json` | Fine-tune pilot outputs |
| `comparison/metrics_v3a.json` | Re-tokenized lengths, ROUGE-L, format, grounding, repetition |
| `scripts/evaluate.py` | Regenerates the metrics file |
| `tests/` | CPU-only contract tests over the artifacts |
| `Notebooks/Preprint_Critic_V3.ipynb` | Training, evaluation, and pilot notebook |
| `requirements.txt` | Core training stack |
| `requirements-dev.txt` | Test-only dependencies |

**Gitignored, and why**

- `Checkpoint/` - the training checkpoint at `checkpoint-350`, including optimizer and scheduler state. It is large and is regenerated by retraining; the adapter is byte-identical to `checkpoint-350`.
- `rc_project_datasets_V3/` - tokenized and untokenized train/validation/test artifacts. These are large and are regenerated by the notebook's data cells.
- `comparison/extracted_paper_texts.json` - the 79 MB paper-text map used by the pilot. It is regenerable from `papers.zip` and is not part of the adapter artifact.

Because the run resumed from `checkpoint-240`, which is not in this checkout, a fresh clone cannot replay the second session as-is. Reproduction either restores that checkpoint or retrains from the start for the same 350 steps.

## 11. Citation

```bibtex
@misc{re2_2025,
  title         = {Re$^2$: A Consistency-ensured Dataset for Full-stage Peer Review and Multi-turn Rebuttal Discussions},
  year          = {2025},
  eprint        = {2505.07920},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2505.07920},
  note          = {Dataset: \url{https://huggingface.co/datasets/Daoze/ReviewRebuttal}, Apache-2.0}
}

@misc{qwen3_2025,
  title         = {Qwen3 Technical Report},
  author        = {{Qwen Team}},
  year          = {2025},
  eprint        = {2505.09388},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2505.09388}
}
```

## 12. Contact and provenance

This card was written from the run's `rc_project_adapter_V3/trainer_state.json` and the committed result files (`adapter_config.json`, `comparison/comparison_v3a.json`, `comparison/baseline_results_v3a.json`). Where a quantity was not measured, the card says so rather than supplying an estimate.

Contact: the repository maintainers.

Card date: 2026-10-23.
