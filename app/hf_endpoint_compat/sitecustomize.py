try:
    import transformers.file_utils as file_utils

    if not hasattr(file_utils, "is_tf_available"):
        file_utils.is_tf_available = lambda: False
    if not hasattr(file_utils, "is_torch_available"):
        file_utils.is_torch_available = lambda: True
    if not hasattr(file_utils, "is_flax_available"):
        file_utils.is_flax_available = lambda: False
    if not hasattr(file_utils, "is_tokenizers_available"):
        file_utils.is_tokenizers_available = lambda: True
except Exception:
    pass
