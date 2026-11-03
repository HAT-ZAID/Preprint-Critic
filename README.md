# Preprint Critic

A 4B-parameter model, fine-tuned to turn a full research paper into a structured peer review.

The base model is `Qwen/Qwen3-4B-Instruct-2507`, adapted with LoRA on the Re² peer review dataset. Given a paper, it returns one review of roughly 370 words, organised as a summary, a list of strengths, a list of weaknesses and a short verdict.

Trained on 2,800 human reviews drawn from 1,439 OpenReview papers. Dataset: [`Daoze/ReviewRebuttal`](https://huggingface.co/datasets/Daoze/ReviewRebuttal) (Apache-2.0, [arXiv:2505.07920](https://arxiv.org/abs/2505.07920)).

```
LoRA r=8 · Qwen3-4B-Instruct-2507 · peft 0.21.2 · transformers 5.19.0 · torch 2.11.0+cu130
16,515,072 trainable params (0.4089% of 4,038,983,168)
```

[Model card](MODEL_CARD.md) · [Training metrics](comparison/metrics_v3a.json) · [Base vs adapter comparison](comparison/comparison_v3a.json) · [Evaluation notes](comparisons/FINDINGS.md)

## What it does

One paper in, one review out.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "rc_project_adapter_V3")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
```

The prompt is a single instruction block followed by the paper text, kept under 16,050 tokens. The full prompt ships in [`comparisons/prompt.txt`](comparisons/prompt.txt) and the decoding settings in [`comparisons/settings.json`](comparisons/settings.json).

Typical output is 300 to 400 words. For reference, the median human review in this dataset is 345 words, with the 75th percentile at 517.

## Fine-tuning

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen3-4B-Instruct-2507` |
| Adapter | LoRA, r=8, alpha=16, dropout 0.05 |
| Trainable parameters | 16.5M (0.41%) |
| Epochs / steps | 1 epoch, 350 steps |
| Effective batch size | 8 |
| Learning rate | 2e-4 |
| Hardware | A100-SXM4-40GB, 29.1 GiB peak |
| Final validation loss | 2.1653 (step 350) |
| Test loss | 2.1518 |

The split is by paper, not by row, so no test paper appears in training. Raw rows: 7,040. After length caps: 2,800 train, 280 validation, 280 test, covering 1,439, 163 and 172 papers. All 172 test papers are unseen.

Loss is computed on the review only. The paper text is context, and the median training row supervises about 4% of its tokens.

## What improved

Adapter against the same base model, same prompt, same decoding settings, on the shared ten test papers.

| Measure | Fine-tuned | Base |
|---|---|---|
| Review length (words) | 374 | 2,322 |
| ROUGE-L against the human review | 0.153 | 0.086 |
| Embedding similarity to the human review | 0.858 | 0.846 |
| Usefulness per 1k generated tokens | 81.2 | 21.2 |
| Generated tokens per review | 495 | 3,366 |

ROUGE-L is the lexical overlap with the human review. Embedding similarity is cosine distance between review embeddings from a local sentence embedding model. Neither is a quality score on its own; both are reported because they move in the same direction as the human-style measures.

Read plainly:

- It writes reviews that look structurally like the human ones: summary first, then strengths, then weaknesses, then a verdict.
- It costs about a fifth of the generated tokens per review.
- It stays on topic. Across all fifteen configurations tested, none collapsed into repetition except the runs that deliberately used low temperature with a wide prompt.
- On length it lands near the human median instead of writing a 2,000-word essay.

## Run it

```bash
pip install -r requirements.txt
python -m preprint_critic.review --paper paper.txt --out review.md
```

Notebook: `Notebooks/Preprint_Critic_V3.ipynb`, which is the training run and is kept for reference.

## Future works

- Reviews stop near 374 words, so coverage and argument depth stay thin.
- Judged usefulness is 40.2 against 71.4 for the base model, and the base wins every head-to-head (both sides were scored as one whole review; a section by section comparison might favour the adapter, which has not been tested).
- Occasional reviewer role-play, such as "I have not read that far yet".
- Longer reviews with the same grounding discipline.
- Everything else is tracked in open issues and pull requests.

## Credits

Dataset: Re² (`Daoze/ReviewRebuttal`), Apache-2.0. Base model: Qwen3-4B-Instruct-2507. Trained and evaluated on Kaggle and locally.
