---
base_model: meta-llama/Llama-3.1-8B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
  - base_model:adapter:meta-llama/Llama-3.1-8B-Instruct
  - lora
  - phishing-detection
  - explainability
  - cybersecurity
  - sft
---

# Explainable Phishing Detector - Llama 3.1 8B

PEFT LoRA adapter fine-tuned from `meta-llama/Llama-3.1-8B-Instruct` to classify
webpages as phishing or benign and return feature-grounded explanations.

## Input

The model was trained on structured webpage data derived from URL, HTML, form,
link, resource, JavaScript, title, and redirect features. It does not consume a
URL alone. Callers must extract the trained feature vector and build the prompt
format used in `grad_FINAL.ipynb`.

## Output

```text
Phishing risk factors:
- <feature_id>

Benign mitigating factors:
- <feature_id>

Reasoning:
<1-3 evidence-grounded sentences>

Verdict: phishing
```

The final verdict is either `phishing` or `benign`.

## Inference Endpoint

This repository includes `handler.py`, a Hugging Face custom Inference Endpoint
handler that loads the Llama base model and this adapter. Send:

```json
{
  "inputs": {
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "Analyze this webpage:\n\n..."}
    ]
  },
  "parameters": {"max_new_tokens": 256, "do_sample": false}
}
```

The Llama base model is gated. The endpoint account must have accepted its
license and the deployment needs enough GPU memory for the base model.

## Limitations

- This is an adapter, not a standalone merged model.
- Predictions are decision support, not a substitute for manual investigation.
- Performance outside the training distribution has not been established.
- Missing or incorrectly extracted features can materially change the verdict.

