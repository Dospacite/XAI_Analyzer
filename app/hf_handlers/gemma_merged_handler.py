from __future__ import annotations

import importlib
import sys
from typing import Any

import torch


def _load_transformers_classes():
    for module_name in list(sys.modules):
        if module_name == "transformers" or module_name.startswith("transformers."):
            del sys.modules[module_name]
    importlib.invalidate_caches()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return AutoModelForCausalLM, AutoTokenizer


class EndpointHandler:
    def __init__(self, path: str = ""):
        AutoModelForCausalLM, AutoTokenizer = _load_transformers_classes()
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, padding_side="left")
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "attn_implementation": "sdpa",
        }
        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        self.model.eval()

    def __call__(self, data: dict[str, Any]) -> dict[str, str]:
        inputs = data.get("inputs", "")
        parameters = dict(data.get("parameters") or {})
        if isinstance(inputs, dict) and isinstance(inputs.get("messages"), list):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    inputs["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt = self.tokenizer.apply_chat_template(
                    inputs["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            prompt = str(inputs)

        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        max_input_tokens = int(parameters.pop("max_input_tokens", 4096))
        if encoded["input_ids"].shape[1] > max_input_tokens:
            encoded["input_ids"] = encoded["input_ids"][:, -max_input_tokens:]
            encoded["attention_mask"] = encoded["attention_mask"][:, -max_input_tokens:]
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
        return {"generated_text": generated}
