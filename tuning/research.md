# Research: LLM Fine-Tuning for Explainable Phishing Website Classification

Date: 2026-05-08

## Executive Summary

Fine-tuning is useful for phishing website classification when the system must make fast, repeatable decisions over a known input schema. Prompting alone is attractive for prototyping, but the phishing literature consistently points toward fine-tuned or domain-adapted models for higher recall, better behavior under realistic low-phishing base rates, and more predictable outputs.

For explainability, the strongest design is not "ask the model why after it predicts." A better design is to train the model to emit a structured verdict with grounded evidence: URL indicators, HTML/form indicators, brand/domain mismatch, screenshot/logo cues, credential-taking intent, and missing or ambiguous evidence. Explanations should be evaluated separately from label accuracy because a model can be right for the wrong reason.

Recommended direction:

1. Fine-tune a compact open model with LoRA/QLoRA for structured phishing website analysis.
2. Use multimodal or multi-input records: URL, normalized HTML text, DOM/form features, screenshot-derived brand/logo signals, and optional reputation metadata.
3. Train on JSON outputs containing `label`, `confidence`, `target_brand`, `intention`, `evidence`, and `explanation`.
4. Evaluate by time/FQDN/campaign splits, not random splits only.
5. Measure explanation faithfulness with evidence alignment checks, not just human readability.

## What The Literature Says

### Fine-tuning vs prompting

Trad and Chehab compare prompt engineering and fine-tuning for phishing URL detection. Their setup fine-tunes pretrained LLMs with a classification head for phishing vs legitimate URLs, while prompt engineering uses grouped zero-shot prompts. They report that fine-tuned models outperform prompt-engineered models, including under lower phishing base-rate scenarios closer to production traffic. Source: [Prompt Engineering or Fine-Tuning? A Case Study on Phishing Detection with Large Language Models](https://www.mdpi.com/2504-4990/6/1/18).

Takeaway: use prompting for baselines and data-labeling assistance, but use fine-tuning for the production classifier if enough labeled examples are available.

### Website phishing needs more than URLs

URL-only detection is cheap and useful, but modern phishing often uses compromised domains, redirect chains, cloaking, copied login pages, and brand impersonation. The strongest website detectors use multiple views of the page.

PhishLLM, from USENIX Security 2024, frames phishing detection as reference-based brand-domain consistency without a manually maintained brand list. It uses LLM knowledge plus search validation to infer intended brand/domain relationships and credential-taking intent, reporting recall gains over Phishpedia and PhishIntention. Source: [USENIX PhishLLM paper page](https://www.usenix.org/conference/usenixsecurity24/presentation/liu-ruofan).

KnowPhish builds a large multimodal brand knowledge base and uses an LLM-based approach to extract brand information from webpage text, improving reference-based phishing detection beyond logo-only methods. Source: [USENIX KnowPhish paper page](https://www.usenix.org/conference/usenixsecurity24/presentation/li-yuexin).

Phishpedia is a useful non-LLM baseline because it is explicitly explainable: it detects logos in screenshots, matches them to target brands, and flags domain-brand mismatch. Source: [Phishpedia project page](https://sites.google.com/view/phishpedia-site/home).

Takeaway: for explainable classification, include brand impersonation and credential-taking evidence, not only lexical URL features.

### Lightweight models can be enough

PhishLang uses MobileBERT in a fully client-side browser extension, combining URL and source-code models. It emphasizes privacy, local operation, and zero-day/evasive phishing detection without external blocklists. Source: [PhishLang arXiv](https://arxiv.org/abs/2408.05667).

Small-language-model work in 2026 argues that cost, latency, and privacy make smaller models attractive for phishing website detection, while proprietary LLMs remain useful as teachers, evaluators, or high-confidence fallback systems. Source: [Small Language Models for Phishing Website Detection](https://www.mdpi.com/2624-800X/6/2/48).

Takeaway: do not assume a large general LLM is required. A compact encoder or 1B-8B instruction model fine-tuned on high-quality examples may be more deployable.

### Explanations require explicit training and evaluation

PhishDebate uses specialized agents for URL structure, HTML composition, semantic content, and brand impersonation, then combines their reasoning through a moderator/judge flow. It reports high recall and emphasizes interpretability. Source: [PhishDebate arXiv](https://arxiv.org/abs/2506.15656).

PhishIntentionLLM focuses on intent labels such as credential theft, financial fraud, malware distribution, and personal information harvesting using screenshot-based multi-agent RAG. Source: [PhishIntentionLLM arXiv](https://arxiv.org/abs/2507.15419).

SCAMNET is adjacent but highly relevant: it uses a two-step LoRA process for fraudulent shopping websites. First it fine-tunes for classification, then it fine-tunes on curated explanation data. This is a practical recipe for preserving detection ability while improving explanation quality. Source: [SCAMNET AAAI 2025 PDF](https://adamdoupe.com/publications/scamnet-aaai2025.pdf).

Kuikel, Piplai, and Aggarwal evaluate phishing classification and explainability with fine-tuned BERT/Llama/Wizard-style models and measure prediction-explanation alignment with SHAP-based consistency. They find that better accuracy and better explanation faithfulness do not necessarily come from the same model. Source: [Evaluating LLMs for Phishing Detection, Self-Consistency, Faithfulness, and Explainability](https://huggingface.co/papers/2506.13746).

Takeaway: train explanations as first-class outputs and evaluate whether the cited evidence actually supports the verdict.

## Candidate Datasets

Use datasets that preserve URL, HTML, screenshots, and collection time where possible.

| Dataset | Useful contents | Notes |
| --- | --- | --- |
| WikiPhish | 87,563 legitimate pages and 23,043 phishing pages with URLs, HTML, and screenshots | Strong candidate for multimodal training/eval; access by request. Source: [WikiPhish](https://www.hornetsecurity.com/en/wikiphish/) |
| LNU-Phish | 23,364 websites with URL, raw HTML, screenshot, DNS records, and features | Useful because it stores page state at collection time. Source: [LNU-Phish](https://lnu-phish.github.io/) |
| PhreshPhish | Large phishing dataset and benchmark designed to reduce leakage and unrealistic base-rate artifacts | Important for realistic evaluation methodology. Source: [PhreshPhish arXiv](https://arxiv.org/abs/2507.10854) |
| Phishpedia data | Logo labels, screenshots, HTML, and discovered phishing pages | Useful for brand/logo explanation baselines. Source: [Phishpedia](https://sites.google.com/view/phishpedia-site/home) |
| LegitPhish | 101,219 labeled URLs with engineered lexical/structural features | Good URL-only baseline; not enough by itself for rich website explanations. Source: [LegitPhish](https://pmc.ncbi.nlm.nih.gov/articles/PMC12538017/) |

## Recommended Training Schema

Each training example should preserve raw-ish artifacts, extracted features, and a structured target.

Input fields:

```json
{
  "url": "https://example.test/login",
  "final_url": "https://example.test/login",
  "html_text": "visible text and selected attributes only",
  "dom_summary": {
    "forms": 1,
    "password_inputs": 1,
    "external_scripts": 4,
    "iframe_count": 0
  },
  "screenshot_caption": "Rendered page appears to imitate Example Bank login",
  "logo_or_brand_candidates": ["Example Bank"],
  "domain_features": {
    "registered_domain": "example.test",
    "age_days": 3,
    "has_punycode": false
  }
}
```

Target output:

```json
{
  "label": "phishing",
  "confidence": 0.91,
  "target_brand": "Example Bank",
  "intention": "credential_theft",
  "evidence": [
    {
      "type": "brand_domain_mismatch",
      "value": "Page claims Example Bank but registered domain is unrelated."
    },
    {
      "type": "credential_collection",
      "value": "Login form requests username and password."
    }
  ],
  "benign_counterevidence": [],
  "explanation": "The page appears to impersonate Example Bank and collects credentials on an unrelated young domain."
}
```

Use a constrained label set:

- `label`: `phishing`, `benign`, `suspicious`, `insufficient_evidence`
- `intention`: `credential_theft`, `financial_fraud`, `personal_information_harvesting`, `malware_distribution`, `brand_impersonation`, `unknown`
- `evidence.type`: `url_lexical`, `domain_reputation`, `brand_domain_mismatch`, `credential_collection`, `html_obfuscation`, `redirect_behavior`, `visual_impersonation`, `security_claim_abuse`, `benign_context`, `missing_signal`

## Fine-Tuning Approach

### Baseline stack

1. Build classical baselines first: logistic regression or gradient boosting over URL/DOM features; MobileBERT/RoBERTa encoder over URL plus visible text.
2. Add a generative explanation model only after the classifier baseline is measurable.
3. Compare against zero-shot and few-shot prompting with the same schema.

### Main LLM approach

Use LoRA or QLoRA rather than full fine-tuning unless there is a strong reason to update all weights. LoRA freezes pretrained weights and trains low-rank adapters; QLoRA reduces memory by training adapters through a frozen 4-bit quantized model. Sources: [LoRA arXiv](https://arxiv.org/abs/2106.09685), [QLoRA arXiv](https://arxiv.org/abs/2305.14314).

Recommended model choices:

- Encoder classifier: DeBERTa/RoBERTa/MobileBERT for fast URL + text classification.
- Small generative model: Llama/Qwen/Mistral-class 1B-8B instruction model with LoRA for structured JSON verdicts.
- Multimodal model: VLM only if screenshots are central and enough screenshot-labeled examples exist. Otherwise, use a separate screenshot/logo model and feed its extracted signals to the LLM.

Recommended staged training:

1. Stage A: supervised fine-tuning for correct structured labels without long explanations.
2. Stage B: supervised fine-tuning on curated evidence/explanation examples.
3. Stage C: preference tuning or rejection sampling to penalize unsupported explanations, schema violations, and overconfident answers.

This mirrors the practical lesson from SCAMNET: separate task adaptation from explanation adaptation.

## Evaluation Plan

Classification metrics:

- Precision, recall, F1, PR-AUC, ROC-AUC.
- False-positive rate at fixed recall, because benign-site blocking is operationally expensive.
- Recall for zero-day or newly collected pages.
- Calibration: expected calibration error and confidence reliability curves.

Explainability metrics:

- Evidence precision: percentage of cited evidence items that are actually present in the input.
- Evidence coverage: whether the explanation mentions the main decisive signals.
- Prediction-explanation consistency: explanation should support the predicted label.
- Counterfactual sensitivity: removing the cited evidence should reduce phishing confidence.
- Human analyst usefulness: time-to-triage and analyst agreement.

Splitting strategy:

- Split by FQDN/domain, brand, campaign, and collection time.
- Include realistic base-rate tests where phishing is rare.
- Deduplicate near-identical pages, templates, and URLs.
- Keep a recently collected holdout set to measure drift.

PhreshPhish is especially relevant here because it calls out leakage, unrealistic base rates, and over-optimistic benchmarking in existing phishing datasets. Source: [PhreshPhish arXiv](https://arxiv.org/abs/2507.10854).

## Security And Data Handling Risks

Phishing webpages are adversarial inputs. Treat webpage content as untrusted data.

- Do not let HTML text directly instruct the model. Wrap it as quoted evidence or parsed fields.
- Strip scripts where possible; collect dynamic behavior in a sandbox.
- Avoid opening live phishing pages outside an isolated crawler.
- Store screenshots and HTML safely because they may contain victim tokens, PII, or active payload references.
- Log model evidence and raw features for audit, but redact secrets and personal data.
- Use temporal retraining because phishing campaigns drift quickly.

## Practical Recommendation

For this project, the most defensible initial target is a text-plus-feature fine-tuned model:

1. Inputs: URL, final URL, visible text, selected HTML attributes, DOM/form summary, screenshot/logo-derived brand candidates.
2. Model: compact instruction model fine-tuned with LoRA/QLoRA to emit strict JSON.
3. Baseline: MobileBERT/RoBERTa classifier over URL + visible text.
4. Explanation: evidence-grounded short explanation, not free-form chain-of-thought.
5. Evaluation: time/domain/brand split, realistic base-rate testing, and evidence faithfulness checks.

This balances accuracy, explainability, privacy, and deployability better than a large prompted model or a URL-only classifier.

## Source List

- Trad, F.; Chehab, A. "Prompt Engineering or Fine-Tuning? A Case Study on Phishing Detection with Large Language Models." 2024. https://www.mdpi.com/2504-4990/6/1/18
- Liu et al. "Less Defined Knowledge and More True Alarms: Reference-based Phishing Detection without a Pre-defined Reference List." USENIX Security 2024. https://www.usenix.org/conference/usenixsecurity24/presentation/liu-ruofan
- Li et al. "KnowPhish: Large Language Models Meet Multimodal Knowledge Graphs for Enhancing Reference-Based Phishing Detection." USENIX Security 2024. https://www.usenix.org/conference/usenixsecurity24/presentation/li-yuexin
- Lin et al. "Phishpedia: A Hybrid Deep Learning Based Approach to Visually Identify Phishing Webpages." https://sites.google.com/view/phishpedia-site/home
- Roy and Nilizadeh. "PhishLang: A Real-Time, Fully Client-Side Phishing Detection Framework Using MobileBERT." https://arxiv.org/abs/2408.05667
- Li et al. "PhishDebate: An LLM-Based Multi-Agent Framework for Phishing Website Detection." https://arxiv.org/abs/2506.15656
- Li et al. "PhishIntentionLLM: Uncovering Phishing Website Intentions through Multi-Agent Retrieval-Augmented Generation." https://arxiv.org/abs/2507.15419
- Kuikel, Piplai, Aggarwal. "Evaluating Large Language Models for Phishing Detection, Self-Consistency, Faithfulness, and Explainability." https://huggingface.co/papers/2506.13746
- Dalton et al. "PhreshPhish: A Real-World, High-Quality, Large-Scale Phishing Website Dataset and Benchmark." https://arxiv.org/abs/2507.10854
- Loiseau et al. "WikiPhish: A Diverse Wikipedia-Based Dataset for Phishing Website Detection." https://www.hornetsecurity.com/en/wikiphish/
- LNU-Phish dataset. https://lnu-phish.github.io/
- LegitPhish dataset. https://pmc.ncbi.nlm.nih.gov/articles/PMC12538017/
- Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models." https://arxiv.org/abs/2106.09685
- Dettmers et al. "QLoRA: Efficient Finetuning of Quantized LLMs." https://arxiv.org/abs/2305.14314
- SCAMNET AAAI 2025 PDF. https://adamdoupe.com/publications/scamnet-aaai2025.pdf
