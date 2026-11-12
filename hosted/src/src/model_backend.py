"""Shared loading and prompt formatting for the hosted app and TAMU trainer."""
from pathlib import Path

QWEN = "Qwen/Qwen2.5-3B-Instruct"
MINISTRAL = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
REWRITE_PREFIX = (
    "Rewrite the following text in clear, natural language. Preserve its meaning, "
    "facts, names and numbers. Do not invent details or add commentary. "
    "Return only the rewritten text.\n\n"
)


def training_messages(prompt):
    return [{"role": "user", "content": prompt}]


def load_base(model_id, revision=None, **kwargs):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if model_id == MINISTRAL:
        from transformers import Mistral3ForConditionalGeneration
        model_class = Mistral3ForConditionalGeneration
    else:
        model_class = AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = model_class.from_pretrained(model_id, revision=revision, **kwargs)
    return tokenizer, model


def attach_adapter(model, model_id, source, revision=None, base_revision=None, token=None):
    from peft import PeftConfig, PeftModel

    if Path(source).is_absolute() and not Path(source).is_dir():
        raise ValueError("Local adapter folder does not exist.")
    config = PeftConfig.from_pretrained(source, revision=revision, token=token)
    if config.base_model_name_or_path != model_id:
        raise ValueError("The adapter was trained on a different base model.")
    if config.revision and base_revision and config.revision != base_revision:
        raise ValueError("The configured base revision does not match the adapter.")
    return PeftModel.from_pretrained(model, source, config=config, revision=revision,
                                     token=token, is_trainable=False, torch_device="cpu")
