# Preprint Critic

A LoRA fine-tune of **Qwen3-4B-Thinking-2507** that writes peer reviews for conference
submissions from the paper's text.

Trained on [Re²](https://huggingface.co/datasets/Daoze/ReviewRebuttal) (`Daoze/ReviewRebuttal`,
Apache-2.0, [arXiv:2505.07920](https://arxiv.org/abs/2505.07920)) — 2,000 OpenReview papers paired
with their human reviews.

```
LoRA r=8 · Qwen3-4B-Thinking-2507 · peft 0.21.2 · transformers 5.19.0 · torch 2.11.0+cu130
16,515,072 trainable params (0.4089% of 4,038,983,168)
```

---

## What it does

The prompt is the paper plus a four-point evidence checklist — main claims and assumptions ·
whether the method and experiments actually support them · whether the baselines and comparisons
are sufficient · what limitations or alternative explanations weaken the conclusion. The target is
the real human review, and loss is masked to the completion only.

The checklist is the part I'd point at first: it encodes four concrete failure modes as explicit
checks, so the model is pushed toward *evidence* rather than generic praise. Two behaviours the
fine-tune produced, both visible in `comparison.json`:

- **It picks up the dataset's review form.** Outputs open with `summary_of_the_paper:` and carry
  `strength_and_weaknesses:` → `Strengths:` / `Weaknesses:`, reproducing the dataset's
  comma-and-underscore key `clarity,_quality,_novelty_and_reproducibility`.
- **It becomes direct.** The base model reasons for 4–6k characters before reviewing; the
  fine-tuned model answers immediately, which is what you want in a drafting tool.

## Approach

**Joining papers to reviews.** Re² ships reviews and paper sources separately — paper text is not
a column you can read. It lives in a 12.4 GB `papers.zip` (373,425 files). The notebook downloads
it, indexes it against the sampled paper ids (2,000/2,000 matched), and checks which formats exist
before choosing one: every paper had both `.tex` and `.md`, so `.md` was preferred.

**Splitting by paper, not by row.** Reviews expand from one paper into N review rows (2,000
papers → 7,040 rows). Splitting on rows would put two reviews of the same paper on both sides of
the boundary, so the split is keyed on `paper_id`: shuffle the ids, assign 80/10/10, then filter
rows by their paper's partition. Verified: `0 of 160` test papers appear in train.

**Masking the prompt.** The prompt/token boundary is asserted before labels are masked
(`assert prompt_ids[:prompt_length] == encoded['input_ids'][:prompt_length]`), and every row is
asserted to keep at least one unmasked target token, so a row can never silently train on nothing.

**Budgeting the context.** Papers are capped at 6,500 tokens and reviews at 1,500, inside an 8,192
window. Length percentiles are printed per split and `truncated = 0/2500` with p100 = 8,131.

## Results

| | |
|---|---|
| Train loss (final logged, step 70) | **2.2351** |
| Train loss (step 1, ≈ base model) | 2.8531 |
| Validation loss | 2.2496 |
| Test loss | 2.2227 |
| Trainable params | 16,515,072 (0.4089%) |
| Sequence length p10 / p50 / p100 | 6,833 / 7,129 / 8,131 |
| Truncated sequences | 0 / 2,500 |
| Hardware | A100 80GB-class card (79.3 GB reported) |

| step | 1 | 10 | 20 | 30 | 40 | 50 | 60 | 70 |
|---|---|---|---|---|---|---|---|---|
| loss | 2.853 | 2.467 | 2.272 | 2.259 | 2.220 | 2.250 | 2.221 | **2.235** |
| grad norm | 0.805 | 0.369 | 0.196 | 0.154 | 0.163 | 0.139 | 0.179 | 0.218 |

Gradient norms stay in 0.14–0.22 and the learning rate decays linearly 2e-4 → 2.9e-6. Final train
loss sits between validation and test, so there's no overfitting gap — expected from 1 epoch over
2,500 rows at rank 8.

**About the step-1 number, since it's the closest thing to a baseline here.** LoRA initialises
`lora_B` to zeros, so the adapter contributes nothing until the first update and the adapted model
starts out functionally identical to the base model. HF logs step 1 before that update, so **2.8531
is a genuine measurement of the base model's teacher-forced loss** — a drop of 0.618 nats. Two
limits worth stating: it averages the 12 micro-batches of the first optimizer step, so it covers
~36 *training* rows rather than the held-out split, and the validation/test losses above are on
different data. A matched base-model validation loss was never computed, so this is not a
like-for-like before/after. It's the closest available baseline, not a clean one.

Two more notes on where these numbers come from:

- The Colab session collapsed mid-run and training resumed from `checkpoint-70`, which was already
  at the final step. The notebook's `train()` therefore did zero work and printed
  `train_loss: 0.0` — that `0.0` is a resume artifact, not a result. The curve above is recovered
  from `checkpoint-70`'s `trainer_state.json` and matches the log table the notebook itself
  rendered (`2.235146`). Wall-clock training time wasn't preserved, so it's not quoted here.
- Hardware is what the run reported (79.3 GB, an A100 80GB-class card). The notebook still carries
  some T4-era constants and metadata inherited from the tutorial it was adapted from, so the batch
  size and token budget are conservative rather than tuned.

**Evaluation.** `comparison.json` holds 10 held-out papers: base model, fine-tuned model, and the
human review side by side. It's qualitative — no generation metric is computed yet, and the
baseline arm is truncated by a shared `max_new_tokens`, so treat it as a look rather than a
benchmark. Re²'s own paper reports ROUGE-L 17.92 and EmbedCos 0.730 for LoRA-tuned LLaMA-3.1-8B on
this dataset, which is the comparison to grow into.

## Layout

```
Preprint Critic.ipynb                 all 7 steps, 39 cells, outputs recorded
rc_project_adapter_V2/
  rc_project_adapter/                 trained LoRA adapter (66 MB, fp32)
    adapter_model.safetensors         the weights (504 tensors, 16,515,072 params)
    adapter_config.json               LoRA hyperparams + base model
    tokenizer.json                    11 MB (same vocab as the base model's, not the same bytes)
    chat_template.jinja               Qwen3 template
    README.md                         model card
comparison.json                       10 held-out papers: base vs fine-tuned vs human
finetune_results.json                 fine-tuned generations only
baseline_results.json                 base generations only
requirements.txt                      pinned stack
```

The tokenizer is committed for convenience. It carries the same vocabulary as the base model's but
is not byte-identical to it — four bytes differ, in `trim_offsets` / `use_regex` /
`add_prefix_space` decoder settings — so load from the base model if you want the exact upstream
config. Adapter weights are fp32 and would halve in bf16.

## Run it

```bash
pip install -r requirements.txt
```

Needs a GPU and an HF token. Setup downloads `REVIEWS_train.json` (319 MB) and **`papers.zip`
(12.4 GB)** from the Hub, which is the bulk of the wall clock. Smaller cards should work by
lowering `per_device_train_batch_size`; that's untested.

## Load the adapter

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Thinking-2507")
model = PeftModel.from_pretrained(base, "rc_project_adapter_V2/rc_project_adapter")
```

To keep training, pass `is_trainable=True` — `adapter_config.json` ships `inference_mode: true`,
so a plain load freezes the LoRA weights.

The prompt must match training exactly, typos included (`prompt_for` in cell 24 — it reads
"peer reiviwer"). The template is applied with `add_generation_prompt=True` and no
`enable_thinking` argument; this model variant is thinking-only and its template has no such
switch.

## Limitations and next steps

- **Add a matched baseline.** LoRA is a no-op at initialization, so step 1 (2.8531) already
  approximates the base model's loss — but only on ~36 training rows. A base-model validation loss
  on the same 250 held-out rows is forward passes only, roughly 4 minutes at the speed this run
  already achieved. That is the single highest-value thing missing from this repo.
- **Give the model the whole paper.** Papers were capped at 6,500 tokens while Re² papers run
  6,000–16,000, so the tail — experiments, ablations, limitations — is exactly what the checklist
  asks about and exactly what the model can't see. Raising the budget is the first thing I'd
  change.
- **Add real metrics.** Currently 10 hand-read samples. Re² publishes EmbedCos and BERTScore for
  review generation on this data, so a comparable number is available rather than invented.
- **Score the base model too.** Step 1 approximates base loss on a small training batch, but no
  base-model validation loss was computed, so there's no matched before/after delta yet.
- **Re-run generation with a real token budget.** `max_new_tokens=1500` left all 10 base outputs
  ending mid-sentence; 8192 would make the comparison fair.
- **Use the held-out split upstream.** Re² ships `REVIEWS_test.json` (1,000 papers) that this run
  didn't touch; it also needs deduping against Re², which has no decontamination of its own.
- **Use more of the data, and validate while training.** 2,500 of ~5,600 available rows, one epoch,
  one seed, and `eval_steps` above `max_steps` meant no validation ran during training.
- **Free the GPU.** My best estimate is that the bulk of the memory went to the fp32 logits tensor
  at `vocab_size 151,936` rather than the 4-bit weights — that's the usual suspect at 8k context,
  and `liger-kernel`'s fused linear cross-entropy is the documented fix. This run recorded no
  `torch.cuda.max_memory_allocated()` call, so it is an inference, not a measurement.
- **Make it runnable off Colab.** Drive paths are hardcoded in nine cells and the resume path
  points at a checkpoint a fresh clone won't have.
- **Then:** fill in the model card, publish the adapter to the Hub, and add tests around the
  prompt-boundary assertion and the reference-stripping regex.

## Evaluation methodology

`comparison.json` holds 10 held-out papers with three texts each: base-model generation,
fine-tuned generation, and the human review. Everything below is computed from that file with
deterministic string analysis — no GPU, no scoring model, nothing to install.

| measured | how | base | fine-tuned | human ref |
|---|---|---:|---:|---:|
| opens with `summary_of_the_paper:` | literal heading | 0/10 | **10/10** | 1/10 |
| uses the dataset's `clarity,_quality,…` key | literal heading | 0/10 | 4/10 | 1/10 |
| cites a Section/Table/Figure | regex, needs a number | 9/10 | **0/10** | 7/10 |
| contains a numeric result | decimal or percentage | 7/10 | 0/10 | 5/10 |
| carries a citation marker (`[n]`/URL/year) | regex | 1/10 | **3/10** | 1/10 |
| 5-gram repeated ≥3× | non-overlapping chunks | 0/10 | **7/10** | 0/10 |
| ends on a sentence boundary | last character | **0/10** | 9/10 | 9/10 |

**Read the last row first.** All 10 base generations stop mid-sentence and 3 are raw reasoning
rather than reviews, because `max_new_tokens=1500` was shared with a thinking-only model. The base
arm is therefore not a valid control — base-vs-fine differences in this table measure the
generation budget at least as much as the fine-tune. The rows computed on complete fine-tuned
output (schema adoption, repetition) are the cleaner ones.

An earlier version of this table reported "10-gram repeated ≥3×: fine 10/10". That was an artifact
of counting overlapping n-gram windows: one repeated passage matched ~10 times and looked like ten
repeats. With non-overlapping chunks, **no arm repeats a 10-gram three times** — the effect is
real at 5-grams, where 7/10 fine-tuned outputs repeat a phrase three or more times against 0/10
for both the base model and the human reviewers.

What they do establish: the adapter reliably learned the review **form** — every fine-tuned output
opens with the dataset's `summary_of_the_paper:` heading, against 0/10 for the base model and 1/10
for human reviewers. What it did *not* learn is the review's **substance**: it cites nothing, and
it repeats phrases far more than either the base model or a human reviewer. The likely cause is
the 6,500-token paper budget, but that is a hypothesis this repo cannot test — see Limitations.

Note these are n=10 regex proxies, not judgments, and they are directional only.

Not yet computed, and the obvious next step: ROUGE-L and EmbedCos against Re²'s published
numbers (17.92 / 0.730 for LoRA-tuned LLaMA-3.1-8B on this dataset). Both need no retraining,
only the metric libraries.

## Credits

- Data: [Daoze/ReviewRebuttal](https://huggingface.co/datasets/Daoze/ReviewRebuttal) — Re², Apache-2.0, [arXiv:2505.07920](https://arxiv.org/abs/2505.07920). Paper text is converted from initial-submission PDFs by the dataset authors using commercial OCR.
- Base model: [Qwen/Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) (Apache-2.0), report [arXiv:2505.09388](https://arxiv.org/abs/2505.09388).
- Method: LoRA ([arXiv:2106.09685](https://arxiv.org/abs/2106.09685)) · QLoRA ([arXiv:2305.14314](https://arxiv.org/abs/2305.14314)).

MIT licensed (code). The adapter inherits the base model's Apache-2.0 terms.