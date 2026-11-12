"""Input checks shared by workflow and inference boundaries."""
from .config import SETTINGS


def validate_text(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Paste some text first.')
    text = text.strip()
    if len(text) > SETTINGS.max_chars or len(text.split()) > SETTINGS.max_words:
        raise ValueError(f'Use {SETTINGS.max_words:,} words or fewer (and at most {SETTINGS.max_chars:,} characters).')
    return text


def validate_method(method):
    if method not in ('basic', 'examples'):
        raise ValueError('Choose General rewrite or My examples.')
    return method
