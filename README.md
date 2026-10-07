# Preprint Critic

A LoRA fine-tune of **Qwen3-1.7B** to write peer reviews for conference submissions. Give it a paper's text, get back a structured review.

Trained on **ReviewRebuttal** — 500 ICLR papers paired with their human reviews.

```python
LoRA · Qwen3-1.7B · peft 0.21.2 · transformers 5.19.0 · PyTorch 2.11.0+cu130
```

---

## What it does

The prompt is a paper plus *"write a constructive peer review for this paper."* The target is the real human review, truncated and cleaned. Review structure is learned from the data, not hard-coded — output looks like `### Review Summary …`.

---

## Approach

**Joining papers to reviews.** ReviewRebuttal ships reviews and LaTeX sources separately, and the paper text isn't a column you can read directly. The notebook pulls `papers.zip` from the Hub (373,425 files), matches all 500 paper IDs, and extracts text per paper.

**Trimming the input.** Raw papers average ~45.5k characters — roughly 11k tokens before the prompt. References and appendices get regex-stripped (LaTeX `\bibliography{`, `\printbibliography`, `thebibliography`, numbered `## References` headings, `\appendix`), cutting 25.5% and dropping the average to ~33.9k chars.

**Building the example.** Paper text is compacted to 16k chars, wrapped in a chat template with `enable_thinking=False`, and the human review becomes the completion. Loss is masked to the completion only — `DataCollatorForSeq2Seq` with `label_pad_token_id=-100` pads without contributing to the loss.

**Splitting by paper, not by row.** Reviews expand into `(paper, review)` rows — 1,584 from 500 papers. Splitting on rows would put reviews of the same paper on both sides, so the split is keyed on `paper_id`. Verified: 0 of 50 test papers appear in train.

---

## Results

| | |
|---|---|
| Train loss | 2.670 |
| Validation loss | 2.535 |
| Test loss | 2.607 |
| Train runtime | 1,372s on a T4 (1 epoch, 0.875 samples/s) |
| Base model footprint | 1.24 GB in 4-bit |
| Adapter size | 34 MB |

LoRA config: `r=8`, `alpha=16`, `dropout=0.05`, targeting all attention and MLP projections (`q/k/v/o_proj`, `gate/up/down_proj`).

> Loss numbers, not ROUGE. Nobody has published a ROUGE baseline for paper-text → review on this dataset, so I'd be inventing a comparison. Generation is qualitatively coherent — summaries and structured sections that track the actual paper.

---

## Layout

```
Preprint Critic.ipynb           all 7 steps, 35 cells, outputs recorded
rc_project_adapter_v1/          trained LoRA adapter (34 MB)
  adapter_model.safetensors     the weights
  adapter_config.json           LoRA hyperparams
  tokenizer.json / chat_template.jinja
requirements.txt                pinned stack
```

---

## Run it

```bash
pip install -r requirements.txt
```

Needs a GPU — this was trained on a T4 with 4-bit quantisation. Then open `Preprint Critic.ipynb` and run top to bottom; the data-prep cells will re-download `papers.zip` from the Hub (needs `HF_TOKEN` for higher rate limits).

To load the adapter instead of retraining:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B")
model = PeftModel.from_pretrained(base, "rc_project_adapter_v1")
```

---

## Things I'd fix

- **Reproducibility is thin.** One seed (42), one run, no hyperparameter search. The notebook is a record of what happened, not a tuned result.
- **The dataset is small** — 500 papers, 1,584 review rows, and I capped training at 1,200 rows to fit a T4. There's more available; I didn't use it.
- **`adapter_config.json` has `inference_mode: true`**, which is correct for the published artifact but means you need to flip it to `False` before continuing training.
- **Title came back `null`** on at least one held-out paper, so the title field isn't always populated upstream.

---

## Notes

- Data: [`Daoze/ReviewRebuttal`](https://huggingface.co/datasets/Daoze/ReviewRebuttal). Base model: [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B).
- The notebook was run on Colab (`/content/...` paths); a few cells assume that environment.
- MIT licensed.
