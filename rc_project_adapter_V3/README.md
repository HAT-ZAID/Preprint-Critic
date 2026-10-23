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
---

# Preprint Critic V3 adapter

This is a LoRA adapter for `Qwen/Qwen3-4B-Instruct-2507`: `r=8`, `lora_alpha=16`, `lora_dropout=0.05`, applied to the q, k, v, o, gate, up, and down projections. It has 16,515,072 trainable parameters (0.4089% of the 4,038,983,168-parameter base) and is 63 MiB. It was trained for 1 epoch on the 2,800-row training split drawn from Re²'s 7,040 paper-review rows (`Daoze/ReviewRebuttal`), with the paper text plus a five-point evidence checklist as the prompt and loss masked to the review completion. It generates one peer review per paper.

Validation loss is 2.1653 and held-out test loss is 2.1518 on a single 280-row split. In a 10-paper pilot the adapter is much shorter than the base model (244 vs 1,421 mean words), wins on ROUGE-L in 9 of 10 papers (mean 0.175 vs 0.123), and opens with the dataset's `summary_of_the_paper:` header more often (6/10 vs 0/10). But it cites the paper less (3 Section/Table/Figure citations, 1 grounded, vs 24 citations, 20 grounded for the base) and repeats paragraphs in one sample (30% 5-gram repetition). It learned the shape of the dataset's reviews, not yet the evidence. The pilot has no human evaluation and no significance testing.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507", torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base, "rc_project_adapter_V3")
# inference_mode: true freezes the LoRA weights; pass is_trainable=True to train further.
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

paper = open("paper.txt", encoding="utf-8").read()
CHECKLIST = """Assess the paper against these five points:
1. What are the main claims and assumptions?
2. Do the method and experiments support them?
3. Are the baselines and comparisons sufficient?
4. Which limitations or alternative explanations weaken the conclusion?
5. Can each criticism be justified from the paper?"""
prompt = paper + "\n\n" + CHECKLIST
inputs = tok.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    out = model.generate(
        inputs, max_new_tokens=16384, do_sample=False,
        pad_token_id=151643, eos_token_id=151645,
    )
print(tok.decode(out[0, inputs.shape[1]:], skip_special_tokens=True))
```

Full model card, including training details, limitations, and citation: [../MODEL_CARD.md](../MODEL_CARD.md)
