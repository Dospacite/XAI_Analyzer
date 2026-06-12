---
base_model: Qwen/Qwen3-4B
library_name: peft
pipeline_tag: text-generation
tags:
  - base_model:adapter:Qwen/Qwen3-4B
  - lora
  - phishing-detection
  - explainability
  - cybersecurity
---

# Explainable Phishing Detector — qwen3_4b_explainable_seed42_20260609_072223

Fine-tuned **Qwen/Qwen3-4B** with QLoRA to classify webpages as phishing or benign
**and** generate structured, evidence-grounded explanations for each verdict.

## Model Details

| Field | Value |
|-------|-------|
| Base model | `Qwen/Qwen3-4B` |
| PEFT method | QLoRA (4-bit NF4, double quant) |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Max sequence length | 3072 |
| Training epochs (actual) | 3.00 |
| Early stopping patience | 2 evals |
| Best eval loss | 0.011818 |
| Best global step | 3213 |

## Training Data

- **Source:** `50k.jsonl` (147,524-row phishing/benign webpage feature dataset)
- **Balance:** Strict 50/50 — 34248 train / 7446 val / 14962 test
- **Input:** URL, hostname, page title, structured numerical features (126 feature groups)
- **Target:** Structured explanation (feature lists + rationale) followed by verdict token
- **Leakage control:** `families` metadata excluded from model input

## Output Format

```
Phishing risk factors:
- <feature_id>

Benign mitigating factors:
- <feature_id>

Reasoning:
<1-3 evidence-grounded sentences>

Verdict: phishing  # or: Verdict: benign
```

## Evaluation Results (Fine-tuned, 14962 held-out examples)

### Classification
| Metric | Value |
|--------|-------|
| Accuracy | 0.8907 |
| Precision (phishing) | 0.8771 |
| Recall (phishing) | 0.9062 |
| F1 | 0.8914 |
| AUC-PR | N/A |

### Explanation Quality
| Metric | Value |
|--------|-------|
| Feature-grounding F1 (combined) | 0.8316 |
| ROUGE-L | 0.8419 |
| BLEU | 0.7461 |
| Mean faithfulness score | 0.9800 |
| Taxonomy compliance | 1.0000 |
| Contradiction rate | 0.2389 |
| Parse failure rate | 167/7000 |
| Template rate | 0.9873 |

## Known Limitations

- Recall on phishing is below precision — the model may miss novel phishing patterns.
- Evaluated on a single dataset distribution; OOD performance may differ.
- 4-bit quantisation may introduce minor accuracy degradation vs full-precision inference.
- The 50/50 class balance does not reflect real-world phishing base rates.
- `families` metadata excluded to prevent data leakage.

## Feature Taxonomy

The model produces feature IDs from a defined taxonomy of 36 identifiers
(e.g., `credential.password_input_present`, `brand.title_domain_mismatch`, `redirect.cross_domain`).

## Pipeline Usage

```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch, json
from pathlib import Path

contract = json.loads((Path("/content/drive/MyDrive/qwen3_explainable_runs/qwen3_4b_explainable_seed42_20260607_141527/adapter") / "deployment_contract.json").read_text())
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_use_double_quant=True,
                          bnb_4bit_compute_dtype=torch.bfloat16)
base  = AutoModelForCausalLM.from_pretrained(contract["base_model_id"],
                                              quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(base, contract["adapter_dir"])
tok   = AutoTokenizer.from_pretrained(contract["tokenizer_dir"])
```

## Hugging Face Inference Endpoint

The repository includes `handler.py`, a custom Inference Endpoint handler that
loads `Qwen/Qwen3-4B` and applies this adapter. It accepts `inputs.messages` and
returns a `generated_text` field. Deploy the repository as a Custom task, then
send the system and model-specific user prompt documented in
`deployment_contract.json`.

## Framework

- PEFT / Transformers / BitsAndBytes / Hugging Face Trainer
- Trained on Google Colab A100
