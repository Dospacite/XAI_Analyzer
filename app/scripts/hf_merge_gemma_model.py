# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate>=1.7.0",
#   "huggingface-hub>=0.32.0",
#   "peft==0.19.1",
#   "safetensors>=0.5.3",
#   "torch>=2.6.0",
#   "transformers>=4.53.0",
# ]
# ///
from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge the Gemma phishing PEFT adapter and upload standalone weights.")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--title", required=True)
    return parser.parse_args()


def model_card(args: argparse.Namespace) -> str:
    return f"""---
base_model:
- {args.base}
library_name: transformers
pipeline_tag: text-generation
tags:
- merged
- gemma-4
- phishing
- cybersecurity
- explainable-ai
---

# {args.title}

Standalone model produced by merging the LoRA adapter
[`{args.adapter}`](https://huggingface.co/{args.adapter}) into
[`{args.base}`](https://huggingface.co/{args.base}).

Expected assistant output is compact JSON with `label`, `confidence`, and
`explanation`.
"""


def normalize_tokenizer_config(output: Path) -> None:
    path = output / "tokenizer_config.json"
    if not path.exists():
        return
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    extra = data.get("extra_special_tokens")
    if isinstance(extra, list):
        data["extra_special_tokens"] = {
            ("video_token" if value == "<|video|>" else f"extra_token_{index}"): value
            for index, value in enumerate(extra)
        }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    token = os.environ["HF_TOKEN"]
    if not torch.cuda.is_available():
        raise RuntimeError("This merge job requires a CUDA GPU")

    print(f"Loading tokenizer from adapter {args.adapter}")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, token=token, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model {args.base}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        token=token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    print(f"Applying adapter {args.adapter}")
    peft_model = PeftModel.from_pretrained(base, args.adapter, token=token, is_trainable=False)
    merged = peft_model.merge_and_unload(safe_merge=True, progressbar=True)
    merged.eval()

    with tempfile.TemporaryDirectory(prefix="gemma-merged-") as output_dir:
        output = Path(output_dir)
        print(f"Saving merged model to {output}")
        merged.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")
        tokenizer.save_pretrained(output)
        normalize_tokenizer_config(output)
        (output / "README.md").write_text(model_card(args), encoding="utf-8")
        (output / "merge_metadata.json").write_text(
            json.dumps(
                {
                    "adapter": args.adapter,
                    "base_model": args.base,
                    "dtype": "bfloat16",
                    "standalone": True,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        del peft_model
        del base
        gc.collect()
        torch.cuda.empty_cache()

        print(f"Uploading standalone model to {args.destination}")
        api = HfApi(token=token)
        api.create_repo(args.destination, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(
            repo_id=args.destination,
            repo_type="model",
            folder_path=output,
            commit_message=f"Upload merged standalone model from {args.adapter}",
        )

    info = HfApi(token=token).model_info(args.destination, files_metadata=True)
    weight_files = [
        sibling.rfilename
        for sibling in info.siblings
        if sibling.rfilename.endswith((".safetensors", ".bin"))
    ]
    if not weight_files:
        raise RuntimeError("Upload completed without model weight files")
    print(json.dumps({"repo": args.destination, "weights": weight_files}, indent=2))


if __name__ == "__main__":
    main()
