# Preprint Critic

A LoRA fine-tune of **Qwen3-4B-Thinking-2507** that writes evidence-grounded peer reviews for
conference submissions from the paper's text.

Trained on [Re²](https://huggingface.co/datasets/Daoze/ReviewRebuttal) (`Daoze/ReviewRebuttal`,
Apache-2.0, [arXiv:2505.07920](https://arxiv.org/abs/2505.07920)) — 2,000 OpenReview papers paired
with their human reviews.

```
LoRA r=8 · Qwen3-4B-Thinking-2507 · peft 0.21.2 · transformers 5.19.0 · torch 2.11.0+cu130
16,515,072 trainable params (0.4089% of 4,038,983,168)
```

---

## What it does

The prompt is the paper plus a four-point evidence checklist (main claims · do the experiments
support them · are the baselines sufficient · what limitations weaken the conclusion). The target
is the real human review. Loss is masked to the completion only.

The adapter does two things that are worth stating plainly, because neither was the intent and
both are measurable:

- **It learns the review form.** Every output opens with `summary_of_the_paper:` and carries
  `strength_and_weaknesses:` → `Strengths:` / `Weaknesses:`, reproducing the dataset's
  comma-and-underscore key `clarity,_quality,_novelty_and_reproducibility` verbatim.
- **It stops reasoning.** The base model emits 4,489–6,422 characters of reasoning before
  reviewing; the fine-tuned model emits **none** on 10 of 10 held-out papers. This is trained
  behaviour — the completion is prefixed with an unmasked blank line immediately after the
  template's `assistant\n<think>\n` prefill, so the model learned to open its thinking block and
  never close it.

## What it does *not* do, stated up front

The headline finding from my own review of the outputs, reproducible from `comparison.json`:

| | fine-tuned | base | human refs |
|---|---:|---:|---:|
| outputs citing a Section / Table / Figure | **0 / 10** | 9 / 10 | 7 / 10 |
| outputs with a ≥3× repeated 40-gram | 8 / 10 | 2 / 10 | 0 / 10 |
| outputs with a fabricated citation or URL | 2 / 10 | 0 / 10 | 0 / 10 |

Loss fell from 2.85 to 2.24 while evidence citation fell to zero. Teacher-forced loss cannot see
fabrication or repetition, because under teacher forcing the model never has to predict its own
drift — so the loss numbers below are **not** evidence of review quality, and the only quality
evaluation in this repo is 10 hand-read samples.

The likely mechanical cause is truncation: papers were cut to a 6,500-token budget, while Re²
papers run 6,000–16,000 tokens and the model config declares 262,144 positions. The model was
being asked to critique experiments it could not see. That is the first thing I would fix.

## Approach

**Joining papers to reviews.** Re² ships reviews and LaTeX sources separately — paper text is not
a column you can read — and paper text lives in a 12.4 GB `papers.zip` (373,425 files, 92.7% of the
13.4 GB dataset repo). The notebook downloads it, indexes it against the sampled paper ids
(2,000/2,000 matched, 100%), and checks which format exists before choosing: every paper had both
`.tex` and `.md`, so `.md` was preferred.

**Splitting by paper, not by row.** Reviews expand from 1 paper → N review rows (2,000 papers →
7,040 rows). Splitting on rows would put two reviews of the same paper on both sides of the
boundary, so the split is keyed on `paper_id`: shuffle the ids, assign 80/10/10, then filter rows
by their paper's partition. Verified `0 of 160` test papers appear in train.

**Masking the prompt.** The prompt/token boundary is asserted before labels are masked —
`assert prompt_ids[:prompt_length] == encoded['input_ids'][:prompt_length]` — and every row is
asserted to keep at least one unmasked target token. Length percentiles are printed per split;
`truncated = 0/2500` with p100 = 8,131 against an 8,192 window.

## Results

| | |
|---|---|
| Train loss (final logged, step 70) | **2.2351** |
| Validation loss | 2.2496 |
| Test loss | 2.2227 |
| Trainable params | 16,515,072 (0.4089%) |
| Sequence length p10 / p50 / p100 | 6,833 / 7,129 / 8,131 |
| Truncated sequences | 0 / 2,500 |
| Hardware | A100 80GB-class card (reported 79.3 GB) |

**Where the train-loss number comes from, disclosed in full:** the Colab session collapsed
mid-run and training was resumed from `checkpoint-70`, which was already at the final step — so
the notebook's own `train()` did zero work and printed `train_loss: 0.0`, `train_runtime: 0.0068`.
`0.0` is not a training result. The real curve lives in `checkpoint-70`'s `trainer_state.json`
and is reproduced in the notebook's own rendered log table (`2.235146`):

| step | 1 | 10 | 20 | 30 | 40 | 50 | 60 | 70 |
|---|---|---|---|---|---|---|---|---|
| loss | 2.853 | 2.467 | 2.272 | 2.259 | 2.220 | 2.250 | 2.221 | **2.235** |
| grad norm | 0.805 | 0.369 | 0.196 | 0.154 | 0.163 | 0.139 | 0.179 | 0.218 |

Gradients are stable and LR decays linearly 2e-4 → 2.9e-6. Final train loss sits between val and
test — no overfitting gap, which is expected from 1 epoch over 2,500 rows at rank 8. **Wall-clock
training time is unrecoverable** (`trainer_state.json` does not store it and the cell that printed
it died), so I have not invented one. The only surviving timing is the post-hoc eval pass:
250 val rows in 4 min 22 s at batch 3.

**A note on the old README's claim that nobody had published a baseline:** that was wrong, and I
checked. Re² itself evaluates review generation, reporting for LoRA-tuned LLaMA-3.1-8B on this
dataset **ROUGE-L 17.92, EmbedCos 0.730** (zero-shot 16.29 / 0.460). Their setup was lr 1e-4,
cosine, 1 epoch on 4×A100. So a comparable number is available — I have not computed it yet, which
is the main gap below.

## Evaluation status

The only generation evaluation is `comparison.json`: 10 held-out papers, base vs fine-tuned vs
the human review, sampled (`temperature 0.6, top_p 0.95, top_k 20, max_new_tokens 1500`), batch
size 1, no metric computed. Two known defects in it, both reproducible from the file:

- **The baseline arm is not a clean control.** `max_new_tokens = 1500` is shared with a
  thinking-only model whose traces run 4–6k characters. 3 of 10 base outputs never produced a
  review at all (they are raw reasoning, 6,627 / 7,630 / 7,068 chars, ending mid-word), and all 10
  end without terminal punctuation.
- **The ten samples share one sampling stream** — the seed is re-set inside the generation
  function before every `model.generate`, so they are not ten independent draws.

No base-model val/test loss was ever computed, so there is no before/after delta.

## Limitations

- **Truncated input.** Papers cut to 6,500 tokens of a median ~9,600, always at the tail —
  experiments, ablations and limitations. The most likely cause of the fabrication and the 0/10
  citation rate.
- **No quantitative evaluation.** 10 hand-read samples. No EmbedCos, no BERTScore, no judge, no
  human eval — despite baselines existing in the source paper.
- **No baseline loss**, so no delta.
- **The base model is thinking-only** and its generations are systematically truncated by the
  shared token budget, so base-vs-fine is not a like-for-like comparison of weights.
- **Half the data unused**: 2,500 of ~5,600 available train rows, 1 epoch, one seed, 70 optimizer
  steps, no validation during training (`eval_steps: 500` vs `max_steps: 70`, so it never fired).
- **Not reproducible from a clean clone.** `from google.colab import drive` is the first import and
  `/content/drive/MyDrive/` is hardcoded in seven cells; `resume_from_checkpoint` points at a
  literal `checkpoint-70` that a fresh clone does not have.
- **Adapter model card is an unfilled template**, and `inference_mode: true` means you must pass
  `is_trainable=True` to `PeftModel.from_pretrained` before continuing training.
- **The prompt contains typos** (`reiviwer`, `seperate`). They are baked into the weights — the
  inference prompt must reproduce them exactly.

## Next steps, in priority order

1. **Stop truncating the paper** — budget the full 6k–16k tokens. Fixes the evidence channel.
2. **Add the missing baseline loss**, then EmbedCos + BERTScore against Re²'s published numbers.
   The fabrication/repetition/citation rates in the table above are computable today and are
   currently the most informative diagnostics in the repo.
3. **Use the benchmarks that already exist** — AAAR-1.0 PaperWeakness (generation, automatic
   metric, eval code shipped) or `REVIEWS_test.json` (1,000 papers, already held out upstream) —
   after deduplicating against Re², which has no decontamination of its own.
4. **Fix the two-arm design**: compare against a non-reasoning base so only weights differ, or put
   reasoning in the targets.
5. **Then throughput.** ~65 of 79.3 GB is the fp32 logits tensor at `vocab_size 151,936`, not the
   weights; `liger-kernel`'s fused linear cross-entropy is the documented fix, after which the
   micro-batch can go up with accumulation down.

## Layout

```
Preprint Critic.ipynb                 all 7 steps, 39 cells, outputs recorded
rc_project_adapter_V2/
  rc_project_adapter/                 trained LoRA adapter (66 MB, fp32)
    adapter_model.safetensors         the weights (504 tensors, 16,515,072 params)
    adapter_config.json               LoRA hyperparams + base model
    tokenizer.json                    11 MB (identical to the base model's)
    chat_template.jinja               Qwen3 template; add_generation_prompt prefills <think>
    README.md                         model card — currently an unfilled template
comparison.json                       10 held-out papers: base vs fine-tuned vs human
finetune_results.json                 fine-tuned generations only
baseline_results.json                 base generations only
requirements.txt                      pinned stack
```

The tokenizer is committed for convenience but is byte-equivalent to
`AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Thinking-2507")` — load from the base model if you
prefer. Adapter weights are stored fp32 and would halve in bf16.

## Run it

```bash
pip install -r requirements.txt
```

Needs a GPU and an HF token. Setup downloads `REVIEWS_train.json` (319 MB) plus **`papers.zip`
(12.4 GB)** from the Hub — that download is the bulk of the wall clock. This was run on Colab
against an A100 80GB-class card; the notebook still carries T4-era constants and metadata from the
tutorial it was adapted from, so treat the batch size and token budget as tuned-for-a-T4
conservatism rather than a measured optimum. A smaller card should work by lowering
`per_device_train_batch_size`, but that is untested.

## Load the adapter

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Thinking-2507")
model = PeftModel.from_pretrained(base, "rc_project_adapter_V2/rc_project_adapter")
```

To keep training, pass `is_trainable=True` — `adapter_config.json` ships `inference_mode: true`,
so a plain load freezes the LoRA weights and `trainer.train()` would silently do nothing.

The prompt must match training exactly, typos included. In the notebook it is
`prompt_for(example)` in cell 24; the template is applied with `add_generation_prompt=True` and
**no** `enable_thinking` argument — this model card states it "supports only thinking mode", and
its chat template has no such conditional.

## Credits

- Data: [Daoze/ReviewRebuttal](https://huggingface.co/datasets/Daoze/ReviewRebuttal) — Re², Apache-2.0, [arXiv:2505.07920](https://arxiv.org/abs/2505.07920). Paper text is converted from initial-submission PDFs via a commercial OCR tool by the dataset authors.
- Base model: [Qwen/Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) (Apache-2.0), technical report [arXiv:2505.09388](https://arxiv.org/abs/2505.09388).
- Method: LoRA ([arXiv:2106.09685](https://arxiv.org/abs/2106.09685)) · QLoRA ([arXiv:2305.14314](https://arxiv.org/abs/2305.14314)).

MIT licensed (code). The adapter inherits the base model's Apache-2.0 terms.