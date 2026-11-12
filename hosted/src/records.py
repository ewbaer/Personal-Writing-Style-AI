"""Explicit per-run state. Model weights never hold request history."""
from datetime import datetime, timezone
from uuid import uuid4
from .config import SETTINGS
from .prompts import PROMPT_VERSION

SCHEMA_VERSION = 1


def now():
    return datetime.now(timezone.utc).isoformat()


def new_run(text, method, profile=None):
    model = profile or SETTINGS
    return dict(schema_version=SCHEMA_VERSION, run_id=uuid4().hex,
                status='validating', started_at=now(), updated_at=now(),
                original=text, method=method, rewrite='', model=model.model_id, model_key=profile.key if profile else 'qwen',
                prompt_version="sft-pairs-v1" if model.experiment_prompt else PROMPT_VERSION, adapter=None, errors=[],
                completed_steps=[], retry_counts={}, ai_detection={},
                settings={'max_new_tokens': SETTINGS.max_new_tokens, 'do_sample': False,
                          'max_prompt_tokens': SETTINGS.max_prompt_tokens})


def update_run(record, status):
    record['status'] = status
    record['updated_at'] = now()


def error_record(record, step, error):
    # No draft text or exception messages in server logs or error records.
    record['errors'].append({'step': step, 'type': type(error).__name__, 'at': now()})
