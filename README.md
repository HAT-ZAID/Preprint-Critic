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

---

## What it does

The prompt is the paper plus a five-point evidence checklist — main claims and assumptions · whether
the method and experiments actually support them · whether the baselines and comparisons are
sufficient · what limitations or alternative explanations weaken the conclusion · whether each
criticism can be justified from the paper. The target is the real human review, and loss is masked
to the completion only.

The checklist is the part I'd point at first: it encodes concrete failure modes as explicit checks,
so the model is pushed toward *evidence* rather than generic praise.

## Approach

**Joining papers to reviews.** Re² ships paper text separately, inside a 12.4 GB `papers.zip`
(373,425 files). Indexed against the sampled ids at a 100% match rate, with a format census run
before choosing `.md` over `.tex` rather than assuming one.

**Splitting by paper, not by row.** Reviews expand from one paper into N review rows (2,000 papers
→ 7,040 rows), so a row-level split would put two reviews of the same paper on both sides of the
boundary. The split is keyed on `paper_id` instead. Verified: `0 of 160` test papers appear in
train.

**Masking the prompt.** The prompt/token boundary is asserted before labels are masked, and every
row is asserted to keep at least one unmasked target token — a row can never silently train on
nothing.

**Measuring truncation instead of assuming it.** The paper and review caps are counted separately
from the window, so a paper hitting its budget is never mistaken for a sequence that overflowed:

| | paper cut at 14,800 | review cut at 1,550 | window cut at 16,384 |
|---|---:|---:|---:|
| train | 332 / 2,500 | 61 / 2,500 | 13 / 2,500 |
| val | 31 / 250 | 9 / 250 | 5 / 250 |
| test | 45 / 250 | 5 / 250 | 0 / 250 |

Sequence length p10 / p50 / p100 on train: **7,803 / 11,944 / 16,457**. The paper is input and only
the review is a target, so roughly **6% of the tokens in this run contribute to loss** — loss numbers
here measure how well the model reproduces a review, not how well it reads a paper.

## Results

| | |
|---|---|
| Validation loss (best, step 313) | **2.1685** |
| Held-out test loss | **2.1629** |
| Validation loss (first eval, step 50) | 2.2226 |
| Trainable params | 16,515,072 (0.4089%) |
| Optimizer steps | 313 (1 epoch, effective batch 8) |
| Peak VRAM | 30.34 GiB allocated / 34.38 GiB reserved |
| Hardware | A100-SXM4-40GB (39.5 GB) |

Validation ran every 50 steps, which is what makes "converged" distinguishable from "stopped early":

| step | 50 | 100 | 150 | 200 | 250 | 300 | 313 |
|---|---|---|---|---|---|---|---|
| eval loss | 2.2226 | 2.1971 | 2.1845 | 2.1783 | 2.1707 | 2.1687 | **2.1685** |

Total movement is **0.054 nats** and the last two points differ by **0.0002** — flat from roughly
step 250 on. Gradient norms sit in 0.24–0.58 after the first logged step (0.72), and the learning
rate decays linearly from 2e-4 to zero at step 313.

**On the step-1 loss.** LoRA initialises `lora_B` to zeros, so the adapter contributes nothing
until the first update and the model starts functionally identical to the base model. HF logs step 1
before that update, so **2.5983 approximates the base model's teacher-forced loss**. It averages the
8 micro-batches of the first optimizer step — 8 *training* rows, not the held-out split — so it is
the closest available baseline rather than a like-for-like before/after.

**Training was resumed** across two sessions, so `train_loss: 2.1470` is the mean over steps 110–313,
not the whole run. The full curve lives in `rc_project_adapter_V3/trainer_state.json`.

## Layout

```
Notebooks/Preprint_Critic_V3.ipynb    all 7 steps, 43 cells, outputs recorded
rc_project_adapter_V3/                trained LoRA adapter (fp32 weights, 63 MiB)
  adapter_model.safetensors            504 tensors, 16,515,072 params
  adapter_config.json                  LoRA hyperparams + base model
  tokenizer.json                       11 MB, same vocab as the base model's
  chat_template.jinja                  Qwen3 Instruct template
  training_args.bin                    resolved TrainingArguments
  trainer_state.json                   full loss/eval history for this run
requirements.txt                      pinned stack
```

`data/` holds the extracted paper text (181 MB) and the joined sample list. Both are gitignored and
regenerable — rerun notebook cells 10–14 to rebuild them. Adapter weights are fp32 and would halve
in bf16.

## Run it

```bash
pip install -r requirements.txt
```

Needs a GPU and an HF token. Setup downloads `REVIEWS_train.json` (319 MB) and **`papers.zip`
(12.4 GB)** from the Hub, which is the bulk of the wall clock.

## Load the adapter

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
model = PeftModel.from_pretrained(base, "rc_project_adapter_V3")
```

To keep training, pass `is_trainable=True` — `adapter_config.json` ships `inference_mode: true`,
so a plain load freezes the LoRA weights.

The prompt must match training. `prompt_for` in cell 27 is the source of truth; the template is
applied with `add_generation_prompt=True`, which renders `assistant\n` with no thinking block on
this checkpoint. Notebook cell 23 asserts that shape before training begins.

## Engineering decisions

**Why Instruct over the Thinking checkpoint.** The two Qwen3-4B checkpoints are architecturally
identical — 36 layers, hidden 2560, vocab 151,936, tied embeddings — so the swap is a
single-variable change: no memory term, KV cache or LoRA parameter count moves. Instruct leads on
open-ended generation benchmarks, which is what this task is. One line of training code depends on
the swap — with no `<think>` prefill, the completion must not open with a closing tag.

**Why the paper budget is the binding constraint.** 14,800 tokens is what lets the model see
experiments, ablations and limitations rather than just the abstract. It is also what forces
`MAX_LENGTH` to 16,384 and the batch to 1, since the logits term scales linearly with sequence
length. Budget, window and batch are one decision.

**What this run does not measure.** Generation quality — no generations are in the repo yet.

## Open work

- **Paired generation eval, next up.** Cells 38–40 cover 100 held-out review rows with
  per-paper deterministic seeds. Cell 38 needs the adapter detached (`disable_adapter()` or
  `merge_and_unload()`) before it produces a real baseline.
- **Metrics.** EmbedCos, chunked at ≤384 tokens since the default sentence encoder truncates
  mid-review, plus a groundedness check — does every Section/Table/URL/number in an output appear
  in the paper. Re² publishes ROUGE-L 17.92 / EmbedCos 0.730 for a LoRA-tuned LLaMA-3.1-8B here.
- **Matched base-model validation loss** on the same 250 held-out rows. Forward passes only, and
  the cheapest way to replace the step-1 approximation above.
- **`REVIEWS_test.json` is unused.** 1,000 held-out papers exist upstream; adopting them needs
  deduping, since Re² ships no decontamination of its own.
- **Model card is still the HF template** — needs filling in before publishing the adapter.
- **Notebook is Colab-bound.** Drive paths are hardcoded and the resume path points at a
  checkpoint a fresh clone will not have.

## Credits

- Data: [Daoze/ReviewRebuttal](https://huggingface.co/datasets/Daoze/ReviewRebuttal) — Re², Apache-2.0, [arXiv:2505.07920](https://arxiv.org/abs/2505.07920). Paper text is converted from initial-submission PDFs by the dataset authors using commercial OCR.
- Base model: [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) (Apache-2.0), report [arXiv:2505.09388](https://arxiv.org/abs/2505.09388).
- Method: LoRA ([arXiv:2106.09685](https://arxiv.org/abs/2106.09685)) · QLoRA ([arXiv:2305.14314](https://arxiv.org/abs/2305.14314)).

MIT licensed (code). The adapter inherits the base model's Apache-2.0 terms.