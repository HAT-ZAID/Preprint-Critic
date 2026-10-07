---
base_model: Qwen/Qwen3-1.7B
library_name: peft
pipeline_tag: text-generation
license: mit
tags:
- base_model:adapter:Qwen/Qwen3-1.7B
- lora
- transformers
---

# Preprint Critic — LoRA adapter

LoRA fine-tune of [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) to write conference peer reviews from a paper's text. Trained on [Daoze/ReviewRebuttal](https://huggingface.co/datasets/Daoze/ReviewRebuttal) — 500 ICLR papers with their human reviews.

Full training details, results, and methodology are in the [repository README](https://github.com/HAT-ZAID/Preprint-Critic).

## Quick start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = PeftModel.from_pretrained(base, "rc_project_adapter_v1")

messages = [{"role": "user", "content": "write a constructive peer review for this paper.\n\n" + PAPER_TEXT}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=512)[0][inputs.input_ids.shape[1]:]))
```

Requires a GPU; the base model was loaded in 4-bit during training.

## Model details

| | |
|---|---|
| Base model | Qwen/Qwen3-1.7B |
| Method | LoRA (`r=8`, `alpha=16`, `dropout=0.05`) |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Task type | `CAUSAL_LM` |
| Adapter size | 34 MB |
| Max sequence length | 6144 |
| PEFT version | 0.21.2 |

## Training data

`Daoze/ReviewRebuttal` — 500 ICLR papers joined to their reviews. Paper text came from the dataset's separate `papers.zip` LaTeX archive (373,425 files, all 500 IDs matched); references and appendices were regex-stripped before prompting, cutting input length 25.5%.

## Results

Loss only — no published baseline exists for paper-text → review on this dataset.

| Split | Loss |
|---|---|
| Train | 2.670 |
| Validation | 2.535 |
| Test | 2.607 |

Splits are keyed on `paper_id`, so no paper appears in both train and test (verified: 0 of 50 test papers overlap). Trained 1 epoch on a T4, 1,372s.

## Limitations

- Single seed, single run, no hyperparameter search — the config is what fit a T4, not a tuned optimum.
- Training capped at 1,200 rows of the 1,584 available.
- No ROUGE or similar metric; quality assessment here is qualitative.
- `adapter_config.json` ships with `inference_mode: true`; set it to `False` before further training.

## Citation

No paper. Cite the base model, the dataset, and this repository.

- Qwen3-1.7B — https://huggingface.co/Qwen/Qwen3-1.7B
- ReviewRebuttal — https://huggingface.co/datasets/Daoze/ReviewRebuttal

## Authors

Z Hat — https://github.com/HAT-ZAID
