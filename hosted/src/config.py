"""Shared limits and model settings for both inference engines."""
from dataclasses import dataclass
import os
from .model_backend import QWEN


def positive_int(name, default):
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f'{name} must be positive.')
    return value


@dataclass(frozen=True)
class Settings:
    model_id: str = os.environ.get('WRITING_MODEL_ID', QWEN)
    base_revision: str | None = os.environ.get('WRITING_MODEL_REVISION')
    adapter: str | None = os.environ.get('WRITING_ADAPTER') or None
    adapter_revision: str | None = os.environ.get('WRITING_ADAPTER_REVISION')
    experiment_prompt: bool = os.environ.get('WRITING_EXPERIMENT_PROMPT', '0') == '1' or bool(os.environ.get('WRITING_ADAPTER'))
    max_words: int = 1000
    max_chars: int = 20000
    max_prompt_tokens: int = 7000
    max_examples: int = 3
    max_new_tokens: int = positive_int('WRITING_MAX_NEW_TOKENS', 1200)
    queue_size: int = 30
    detector_min_words: int = 50
    detector_max_chars: int = 30000
    sentence_min_words: int = 8
    sentence_threshold: float = 0.5
    detector_retries: int = 1
    export_retention_hours: int = 24


SETTINGS = Settings()


@dataclass(frozen=True)
class ModelProfile:
    key: str
    label: str
    model_id: str
    base_revision: str | None = None
    adapter: str | None = None
    adapter_revision: str | None = None
    experiment_prompt: bool = True


def model_profiles():
    from .model_backend import MINISTRAL
    arm2 = os.environ.get("WRITING_ARM2_ADAPTER")
    yield ModelProfile("qwen", "Qwen · ARM 1", SETTINGS.model_id, SETTINGS.base_revision,
                       SETTINGS.adapter, SETTINGS.adapter_revision, SETTINGS.experiment_prompt or bool(arm2))
    if arm2:
        yield ModelProfile("arm2", "Ministral · ARM 2", MINISTRAL,
                           os.environ.get("WRITING_ARM2_MODEL_REVISION"), arm2,
                           os.environ.get("WRITING_ARM2_ADAPTER_REVISION"))
