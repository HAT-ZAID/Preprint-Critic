# Preprint Critic

A LoRA fine-tune of **Qwen3-4B-Instruct-2507** that writes peer reviews for conference
submissions from the paper's text.

Trained on [Re²](https://huggingface.co/datasets/Daoze/ReviewRebuttal) (`Daoze/ReviewRebuttal`,
Apache-2.0, [arXiv:2505.07920](https://arxiv.org/abs/2505.07920)) — 2,000 OpenReview papers paired
with their human reviews.

```
LoRA r=8 · Qwen3-4B-Instruct-2507 · peft 0.21.2 · transformers 5.19.0 · torch 2.11.0+cu130
16,515,072 trainable params (0.4089% of 4,038,983,168)
```

[Model card](MODEL_CARD.md) · [Offline metrics](comparison/metrics_v3a.json) · [Pilot outputs](comparison/comparison_v3a.json)

---

## What it does

The prompt is a five-paragraph instruction brief followed by the paper text (`Paper:\n{paper}`).
The brief asks for a substantive, fair review; requires claims to be grounded in the manuscript and
forbids invented results, citations, numbers, and section/table/figure references; asks for the most
consequential issues with actionable questions; names the dimensions to assess (novelty,
methodological validity, baseline adequacy, limitations, reproducibility); and asks the model to
avoid boilerplate, fixed templates, and unsupported verdicts. The target is the real human review,
and loss is masked to the completion only.

The instruction brief is the part I'd point at first: it encodes concrete failure modes as explicit
requirements, so the model is pushed toward *evidence* rather than generic praise.

## Approach

**Joining papers to reviews.** Re² ships paper text separately, inside a 12.4 GB `papers.zip`
(373,425 files). Indexed against the sampled ids at a 100% match rate, with a format census run
before choosing `.md` over `.tex` rather than assuming one.

**Splitting by paper, then capping rows.** Reviews expand from one paper into N review rows
(2,000 papers → 7,040 rows after the 100-character filter), so a row-level split would put two
reviews of the same paper on both sides of the boundary. Papers are assigned 80/10/10 by
`paper_id` (1,600 / 200 / 200); each split is then shuffled and capped at 2,800 / 280 / 280 rows.
The caps discard 3,680 of the 7,040 eligible rows, so the retained rows cover 1,439 / 163 / 172
papers; 161 / 37 / 28 selected papers contribute no retained row. Verified: **0 of 172** retained
test papers appear in train.

**Masking the prompt.** The prompt/token boundary is asserted before labels are masked, and every
row is asserted to keep at least one unmasked target token — a row can never silently train on
nothing. The median row supervises about **4%** of its tokens (503 of 12,109), so loss measures how
well the model reproduces a review, not how well it reads a paper.

**Measuring truncation instead of assuming it.** The paper and review caps are counted separately
from the window, so a paper hitting its budget is never mistaken for a sequence that overflowed:

| | paper cut at 16,050 | review cut at 1,750 | window cut at 18,432 |
|---|---:|---:|---:|
| train | 198 / 2,800 | 36 / 2,800 | 0 / 2,800 |
| val | 17 / 280 | 6 / 280 | 0 / 280 |
| test | 27 / 280 | 2 / 280 | 0 / 280 |

Train row length (paper + review) p10 / p50 / p90 / p100: **8,038 / 12,110 / 16,277 / 18,040**
tokens. The longest row still fits the 18,432 window, so the per-paper budget — not the window — is
what truncates.

## Results

| | |
|---|---|
| Validation loss (best, step 350) | **2.1653** |
| Held-out test loss | **2.1518** |
| Validation loss (first eval, step 80) | 2.2040 |
| Trainable params | 16,515,072 (0.4089%) |
| Optimizer steps | 350 (1 epoch, effective batch 8) |
| Peak VRAM (resumed session, steps 241–350) | 29.13 GiB allocated / 30.58 GiB reserved |
| Hardware | A100-SXM4-40GB |

Validation ran every 80 steps, which is what makes "converged" distinguishable from "stopped early":

| step | 80 | 160 | 240 | 320 | 350 |
|---|---|---|---|---|---|
| eval loss | 2.2040 | 2.1824 | 2.1725 | 2.1661 | **2.1653** |

Total movement is **0.039 nats**; the last two points differ by **0.0008**. Gradient norms sit in
0.26–0.45 after the first logged step. The test split lands slightly *below* validation (2.1518) —
one 280-row split, so I read that as noise rather than a result.

**On the step-1 loss.** LoRA initialises `lora_B` to zeros, so the adapter contributes nothing until
the first update and the model starts functionally identical to the base model. HF logs step 1
before that update, so **2.5437 approximates the base model's teacher-forced loss**. It averages the
8 micro-batches of the first optimizer step — 8 *training* rows, not the held-out split — so it is
the closest available baseline, not a like-for-like before/after.

**Training was resumed** across two sessions, so `train_loss: 2.1123` is the mean over steps
241–350, not the whole run. The full curve lives in `rc_project_adapter_V3/trainer_state.json`.

## Pilot: 10 held-out papers

The adapter versus the same base model with the adapter disabled, same prompts and decoding, 10
held-out papers. **This is a pilot, not a benchmark** — no human evaluation, no significance test.

| | Base | Adapter | Human ref |
|---|---:|---:|---:|
| Mean words | 1,421 | 244 | 412 |
| ROUGE-L vs reference (mean) | 0.123 | 0.175 | — |
| ROUGE-L range | 0.085–0.140 | 0.120–0.221 | — |
| Opens with `summary_of_the_paper:` | 0/10 | 6/10 | 5/10 |
| Full Re² four-section template | 0/10 | 6/10 | 3/10 |
| Section/Table/Figure refs (grounded) | 24 (20) | 3 (1) | 20 (16) |
| Mean 5-gram repetition | 0.2% | 6.5% (max 30%) | 0.6% |

The adapter wins on ROUGE-L in **9 of 10** papers and tracks the dataset's review format and length
far better than the base model, which writes ~1,400-word essays and ends 3 of 10 samples with a
canned verdict (for example, "*Reviewer recommendation: Major revision.*"). But it also cites the
paper *less* — 3 references with 1 grounded, against 24 with 20 — repeats whole paragraphs in one of
the ten samples (30% 5-gram repetition), and its 244 words are short against a 412-word human
reference.

The honest reading: **it learned the shape of the dataset's reviews before the evidence.** ROUGE-L
rewards length and template overlap, so this is not a quality claim. Numbers are re-computed
offline in [`comparison/metrics_v3a.json`](comparison/metrics_v3a.json).

## Layout

```
Notebooks/Preprint_Critic_V3.ipynb   the full pipeline, 48 cells, outputs recorded
rc_project_adapter_V3/               trained LoRA adapter (fp32 weights, 63 MiB)
  adapter_model.safetensors            504 tensors, 16,515,072 params
  adapter_config.json                  LoRA hyperparams + base model
  trainer_state.json                   350-step loss/eval history
  training_args.bin                    resolved TrainingArguments
comparison/                          10-sample pilot: paired outputs + metrics
MODEL_CARD.md                        model card
scripts/evaluate.py                  offline ROUGE-L / format / grounding metrics
tests/                               CPU-only contract tests over the artifacts
requirements.txt                     core stack
```

Not committed, but regenerable: `Checkpoint/` (optimizer states, 200 MB),
`rc_project_datasets_V3/` (tokenized splits, 766 MB), and the 79 MB extracted paper-text map. They
are gitignored; the notebook rebuilds them.

## Run it

```bash
pip install -r requirements-dev.txt   # CPU only, enough for the artifact tests
python -m pytest tests -q             # dataset tests skip unless rc_project_datasets_V3/ exists

pip install -r requirements.txt       # full training stack
```

Training needs a GPU and an HF token. Setup downloads `REVIEWS_train.json` (319 MB) and
**`papers.zip` (12.4 GB)** from the Hub, which is the bulk of the wall clock. The notebook is still
Colab-bound: Drive paths are hardcoded.

## Load the adapter

```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "rc_project_adapter_V3")
```

To keep training, pass `is_trainable=True` — `adapter_config.json` ships `inference_mode: true`,
so a plain load freezes the LoRA weights.

The prompt must match training. `prompt_for` in the notebook is the source of truth; the template is
applied with `add_generation_prompt=True`, which renders `assistant\n` with no thinking block on
this checkpoint. The notebook asserts that shape before training begins.

## Engineering decisions

**Why Instruct over the Thinking checkpoint.** The two Qwen3-4B checkpoints are architecturally
identical — 36 layers, hidden 2560, vocab 151,936, tied embeddings — so the swap is a
single-variable change: no memory term, KV cache or LoRA parameter count moves. Instruct leads on
open-ended generation benchmarks, which is what this task is.

**Why the paper budget is the binding constraint.** 16,050 tokens is what lets the model see
experiments, ablations and limitations rather than just the abstract. It is also what forces
`MAX_LENGTH` to 18,432 and the batch to 1, since the logits term scales linearly with sequence
length. Budget, window and batch are one decision.

**Resume and best-checkpoint selection.** The run spanned two Colab sessions. Training resumes from
`get_last_checkpoint` rather than a hardcoded path, and `load_best_model_at_end=True` means the
saved adapter is the best validation checkpoint — which here is also the final step.

## Open work

- **The committed pilot is 10 rows; the frozen notebook sets `BASELINE_N = 25`.** Set
  `BASELINE_N = 10` to reproduce the committed `comparison_v3a.json`,
  `baseline_results_v3a.json`, and `metrics_v3a.json`, or regenerate all three plus the 10-row
  assertions in `tests/test_results.py` for 25.
- **Token accounting in the committed JSONs is padded.** `output_tokens` and `hit_cap` are read from
  the padded batch width, so counts repeat within a batch (base 2,296 × 5 and 2,262 × 5; adapter
  393 / 414). Real lengths are base 1,648–2,295 and adapter 182–413. `comparison/metrics_v3a.json`
  re-tokenizes each output and reports the real numbers; the fix belongs in the generation cell.
- **Matched base-model validation loss** on the same 280 held-out rows. Forward passes only, and the
  cheapest way to replace the step-1 approximation above.
- **EmbedCos and a stronger groundedness check.** Chunk at ≤384 tokens, since the default sentence
  encoder truncates mid-review. Re² publishes ROUGE-L 17.92 / EmbedCos 0.730 for a LoRA-tuned
  LLaMA-3.1-8B on this data.
- **EOS survival is asserted nowhere.** The budget has room (16,050 + 1,750 + template < 18,432),
  but the notebook should assert it rather than leave it to arithmetic.
- **`REVIEWS_test.json` is unused.** 1,000 held-out papers exist upstream; adopting them needs
  deduping, since Re² ships no decontamination of its own.

## Credits

- Data: [Daoze/ReviewRebuttal](https://huggingface.co/datasets/Daoze/ReviewRebuttal) — Re², Apache-2.0, [arXiv:2505.07920](https://arxiv.org/abs/2505.07920). Paper text is converted from initial-submission PDFs by the dataset authors using commercial OCR.
- Base model: [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) (Apache-2.0), report [arXiv:2505.09388](https://arxiv.org/abs/2505.09388).
- Method: LoRA ([arXiv:2106.09685](https://arxiv.org/abs/2106.09685)) · QLoRA ([arXiv:2305.14314](https://arxiv.org/abs/2305.14314)).

MIT licensed (code). The adapter inherits the base model's Apache-2.0 terms.
