from pathlib import Path

from setuptools import setup


def patch_hf_toolkit() -> None:
    target = Path("/app/huggingface_inference_toolkit/utils.py")
    if not target.exists():
        return
    original = "from transformers.file_utils import is_tf_available, is_torch_available"
    replacement = (
        "try:\n"
        "    from transformers.file_utils import is_tf_available, is_torch_available\n"
        "except ImportError:\n"
        "    def is_tf_available():\n"
        "        return False\n"
        "    def is_torch_available():\n"
        "        return True"
    )
    text = target.read_text()
    if original in text and replacement not in text:
        target.write_text(text.replace(original, replacement))


patch_hf_toolkit()

setup(name="hf-endpoint-compat", version="0.2.0", py_modules=["sitecustomize"])
