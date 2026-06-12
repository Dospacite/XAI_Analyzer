---
base_model: unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit
library_name: peft
pipeline_tag: text-generation
tags:
  - base_model:adapter:unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit
  - lora
  - phishing-detection
  - explainability
  - cybersecurity
  - sft
  - unsloth
---

# CyberGuard Phishing Detector - DeepSeek R1 Qwen 7B

LoRA adapter fine-tuned from
`unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit` for explainable webpage
phishing analysis.

## Input

The training prompt groups 126 numerical webpage features into URL, page
structure, forms, links, resources, security indicators, and redirects. See
`deepseek_phishing_v3-3.ipynb` for the exact training and inference contract.

## Output

```text
### PHISHING SIGNALS:
<suspicious evidence>

### BENIGN SIGNALS:
<mitigating evidence>

### REASONING:
<2-3 sentence analysis>

### VERDICT: PHISHING
```

## Inference Endpoint

This repository includes `handler.py`, a Hugging Face custom Inference Endpoint
handler that loads the quantized base model and applies this adapter. Requests
accept chat messages:

```json
{
  "inputs": {
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "### TARGET:\nURL: ..."}
    ]
  },
  "parameters": {
    "max_new_tokens": 320,
    "temperature": 0.1,
    "do_sample": true
  }
}
```

## Limitations

- This is a PEFT adapter and requires its base model at runtime.
- The detector depends on accurate feature extraction.
- A verdict should be reviewed alongside the cited evidence.
- Novel phishing kits and out-of-distribution websites may be misclassified.

