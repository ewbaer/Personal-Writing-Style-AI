"""Shared loading and prompt formatting for the hosted app and TAMU trainer."""
from pathlib import Path

QWEN = "Qwen/Qwen2.5-3B-Instruct"
MINISTRAL = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
REWRITE_PREFIX = (
    "Lightly revise the draft below so it reads like a college student answering a "
    "class discussion or journal prompt. Keep the writer's main ideas, facts, "
    "point of view, and natural rhythm. Leave wording alone when it already sounds natural.\n\n"
    "Use common words and natural contractions. First-person phrases like 'I think,' "
    "'I noticed,' and 'what stood out to me' are fine where they fit, but vary how "
    "paragraphs begin. If several short sentences feel choppy, connect related ideas "
    "with 'and,' 'but,' 'because,' or 'so.' Keep some variety in sentence length.\n\n"
    "Keep each paragraph focused. When discussing a book, explain what happened and "
    "why it stood out to the writer. Skip formal transitions like 'Furthermore' and "
    "'This demonstrates.' Avoid repeating 'really' or 'not really.' Don't turn every "
    "point into a life lesson.\n\n"
    "Keep any personal connections grounded in details already in the draft. Preserve "
    "quoted words exactly and never invent quotes or experiences. If the draft answers "
    "multiple questions, keep each answer focused on its own point.\n\n"
    "Keep it casual and appropriate for class. Don't add a title, summary, or "
    "commentary. Return only the revised draft.\n\n"
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
