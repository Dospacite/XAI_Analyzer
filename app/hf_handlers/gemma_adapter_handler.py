from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class EndpointHandler:
    def __init__(self, path: str = ""):
        adapter_path = Path(path)
        config = json.loads((adapter_path / "adapter_config.json").read_text())
        base_model = config["base_model_name_or_path"]

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.bfloat16

        base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()
        self.base_model = base_model

    def __call__(self, data: dict[str, Any]) -> dict[str, str]:
        inputs = data.get("inputs", "")
        parameters = dict(data.get("parameters") or {})
        if isinstance(inputs, dict) and isinstance(inputs.get("messages"), list):
            messages = inputs["messages"]
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            prompt = str(inputs)

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=int(parameters.pop("max_input_tokens", 4096)),
        ).to(self.model.device)
        input_length = encoded["input_ids"].shape[1]
        temperature = float(parameters.pop("temperature", 0))
        do_sample = bool(parameters.pop("do_sample", temperature > 0))
        generation_kwargs = {
            "max_new_tokens": int(parameters.pop("max_new_tokens", 256)),
            "repetition_penalty": float(parameters.pop("repetition_penalty", 1.1)),
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = max(temperature, 0.01)
            generation_kwargs["top_p"] = float(parameters.pop("top_p", 0.9))

        with torch.inference_mode():
            output = self.model.generate(**encoded, **generation_kwargs)
        generated = self.tokenizer.decode(output[0, input_length:], skip_special_tokens=True).strip()
        return {"generated_text": generated, "base_model": self.base_model}
