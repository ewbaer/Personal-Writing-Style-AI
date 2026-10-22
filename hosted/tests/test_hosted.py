"""Local checks for the hosted app. Run from the hf-space folder:

    python tests/make_tiny_model.py /tmp/tiny-qwen
    WRITING_MODEL_ID=/tmp/tiny-qwen python -m pytest tests -q

With WRITING_MODEL_ID pointing at the tiny random model, these tests exercise the
real transformers/Gradio code paths on CPU. They do not prove anything about
Qwen2.5-3B output quality or about ZeroGPU itself.
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("WRITING_MAX_NEW_TOKENS", "40")

import app  # noqa: E402  (loads the model named by WRITING_MODEL_ID)
from src.hosted_engine import gpu_seconds, validate_text  # noqa: E402
from src.rewrite_engine import build_messages, load_examples  # noqa: E402

PASSAGE = (" ".join(["This sentence is here so the detector has enough words to score the passage."] * 8))


def test_model_loaded():
    assert app.engine.model is not None, app.engine.load_error


def test_validation_rejects_empty_and_oversized():
    for bad in ["", "   ", None, 7]:
        with pytest.raises(ValueError):
            validate_text(bad)
    with pytest.raises(ValueError):
        validate_text("word " * 1001)
    with pytest.raises(ValueError):
        validate_text("x" * 20001)
    assert validate_text("  hello  ") == "hello"


def test_examples_exclude_exact_match_and_cap_at_three():
    rows = load_examples(ROOT / "data/splits/train.csv")
    assert len(rows) == 6
    messages, ids = build_messages(rows[0]["AI Version"], "examples", rows)
    assert rows[0]["ID"] not in ids and len(ids) == 3
    assert messages[0]["role"] == "system" and messages[-1]["role"] == "user"
    with pytest.raises(ValueError):
        build_messages("text", "nonsense", rows)


def test_gpu_duration_bounds():
    assert gpu_seconds("", "basic") == 30
    assert gpu_seconds("word " * 1000, "examples") <= 120
    assert 30 <= gpu_seconds("word " * 250, "examples") <= 120


def test_detector_scores_and_highlights():
    data = app.detector.explain(PASSAGE)
    assert data["available"] and 0 <= data["score"] <= 100
    assert data["sentence_count"] == 8
    segments = app.build_highlights(PASSAGE, data)
    assert "".join(s for s, _ in segments) == PASSAGE
    labels = {label for _, label in segments}
    assert labels <= {None, "AI-like"}
    short = app.detector.explain("Too short.")
    assert not short["available"]


def test_streaming_generation_path():
    text, tokens, prompt_tokens, ids = app.engine.prepare("Please rewrite this short note about my class project.", "examples")
    assert prompt_tokens > 0 and len(ids) == 3
    partials, final = [], None
    for item in app.engine.stream(tokens):
        if isinstance(item, dict):
            final = item
        else:
            partials.append(item)
    assert final is not None and final["rewrite"]
    assert partials and partials[-1].strip() == final["rewrite"]
    assert final["output_tokens"] <= 40
    assert final["first_token_seconds"] is not None
    assert final["truncated"] in (True, False)


def test_rewrite_handler_end_to_end():
    updates = list(app.on_rewrite(PASSAGE, "basic", True))
    assert updates[0]["status"].startswith("Waiting")
    last = updates[-1]
    assert last["result"] is not None and last["result"]["rewrite"] == last["rewritten"]
    assert "ai_detection" in last["result"]
    assert last["cache"]["text"] == last["rewritten"]


def test_rewrite_handler_rejects_bad_input():
    import gradio as gr
    with pytest.raises(gr.Error):
        list(app.on_rewrite("", "basic", False))
    with pytest.raises(gr.Error):
        list(app.on_rewrite("word " * 1001, "basic", False))


def test_check_edit_and_export(tmp_path):
    line, details, highlights, cache = app.on_check(PASSAGE, True)
    assert "flagged" in line.lower() and cache["text"] == PASSAGE and highlights
    stale = app.on_edit(PASSAGE + " more", cache)
    assert stale[0] == app.EDITED_SIGNALS
    fresh = app.on_edit(PASSAGE, cache)
    assert not isinstance(fresh[0], str)  # gr.update: nothing changes when text matches
    result = {"rewrite": PASSAGE, "original": "x", "model": "m"}
    path = app.on_export(result, PASSAGE + " edit", cache)
    assert Path(path).exists() and '"edited_rewrite"' in Path(path).read_text()
    import gradio as gr
    with pytest.raises(gr.Error):
        app.on_export(None, "", None)


def test_friendly_messages():
    assert "GPU time" in app.friendly("<p>ZeroGPU quota exceeded</p>")
    assert "memory" in app.friendly("CUDA out of memory")
    assert app.friendly("boom").startswith("Could not rewrite")


def test_gradio_app_builds_and_serves():
    import urllib.request
    app.demo.queue(max_size=30, default_concurrency_limit=1)
    _, url, _ = app.demo.launch(theme=app.THEME, css=app.CSS, js=app.THEME_JS,
                                prevent_thread_lock=True, quiet=True, server_port=7865)
    try:
        page = urllib.request.urlopen(url, timeout=20).read().decode()
        assert "Writing desk" in page
        config = urllib.request.urlopen(url + "config", timeout=20).read().decode()
        assert "Rewrite" in config and "Check edited output" in config
    finally:
        app.demo.close()
