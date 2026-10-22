"""Writing desk on Hugging Face ZeroGPU. Gradio version of the local Flask app.

Start locally for a UI check:  python app.py
The @spaces.GPU decorator does nothing outside a ZeroGPU Space, so the same
file runs on a normal GPU or (slowly) on CPU.
"""
import html
import json
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import spaces  # Must be imported before torch so ZeroGPU can patch CUDA calls.
import gradio as gr

from src.ai_detector import AIDetector
from src.hosted_engine import MAX_NEW_TOKENS, HostedEngine, gpu_seconds, validate_text
from src.rewrite_engine import MODEL_ID

# Load once at startup and place on CUDA at module level (ZeroGPU guidance).
engine = HostedEngine()
engine.load()
detector = AIDetector()
EXPORT_DIR = Path(tempfile.gettempdir()) / "writing-desk-exports"
EXPORT_DIR.mkdir(exist_ok=True)
ON_SPACE = bool(os.environ.get("SPACE_ID"))

METHOD_CHOICES = [("My examples", "examples"), ("General rewrite", "basic")]
HINTS = {"examples": "Uses up to 3 examples from Ethan's writing.",
         "basic": "Rewrites with general instructions, without examples."}
IDLE_SIGNALS = "Rewrite text to see writing signals."
EDITED_SIGNALS = "Output edited · check again to update."


# ---------------------------------------------------------------- helpers
def count_words(text):
    text = (text or "").strip()
    n = len(text.split()) if text else 0
    return f"{n} word" if n == 1 else f"{n} words"


def safe_score(text):
    try:
        return detector.explain(text)
    except Exception as error:
        return {"available": False, "message": f"Detector unavailable: {type(error).__name__}."}


def signals_line(data):
    if not data or not data.get("available"):
        return (data or {}).get("message") or "Detector unavailable."
    return (f"Text flagged as AI-like: {data['flagged_percent']:.1f}% · "
            f"{data['flagged_count']} of {data['sentence_count']} sentences")


def signals_details(data):
    if not data or not data.get("available"):
        return ""
    return (f"Whole-text classifier score: {data['score']:.1f} / 100. "
            f"Flagged words: {data['flagged_words']} of {data['total_words']}. "
            f"Unassessed words: {data['unassessed_words']}. Sentences under 8 words are "
            "left unassessed. Sentence cutoff: 50 / 100. The percentage measures words in "
            "flagged sentences, not verified AI authorship. Sentence scoring has not been validated.")


def build_highlights(text, data):
    """Render the revision as HTML with flagged sentences wrapped in <mark>."""
    text = text or ""
    if not text.strip():
        return '<div class="hl-box hl-empty">Your revision will appear here.</div>'
    if not data or not data.get("available"):
        return f'<div class="hl-box">{html.escape(text)}</div>'
    parts, cursor = [], 0
    for sentence in data.get("sentences", []):
        start = text.find(sentence["text"], cursor)
        if start < 0:
            continue
        parts.append(html.escape(text[cursor:start]))
        piece = html.escape(sentence["text"])
        parts.append(f"<mark>{piece}</mark>" if sentence["highlight"] else piece)
        cursor = start + len(sentence["text"])
    parts.append(html.escape(text[cursor:]))
    return '<div class="hl-box">' + "".join(parts) + "</div>"


def show_editor():
    """Revision textarea visible, highlight view hidden."""
    return gr.update(visible=True), gr.update(visible=False)


def show_highlights(text, data):
    return gr.update(visible=False), gr.update(value=build_highlights(text, data), visible=True)


def friendly(error):
    message = re.sub(r"<[^>]+>", " ", str(error))
    message = " ".join(message.split())
    low = message.lower()
    if "quota" in low or "gpu task aborted" in low or "zerogpu" in low:
        return ("No free GPU time left for you right now. Hugging Face gives signed-in "
                "visitors more daily ZeroGPU minutes than anonymous visitors, so sign in or "
                "try again later. Details: " + message)
    if "out of memory" in low:
        return "The GPU ran out of memory. Try a shorter draft."
    if "not loaded" in low:
        return "The rewrite model did not load on this Space. Check the Space logs. " + message
    return "Could not rewrite: " + message


def run_details(result):
    if not result:
        return "No rewrite in this session."
    ttft = result.get("first_token_seconds")
    ttft_text = f"{ttft}s to first token · " if ttft else ""
    return (f"{result['model']}  \n{result['gpu']} · {ttft_text}{result['seconds']}s generation  \n"
            f"{len(result['example_ids'])} examples · {result['prompt_tokens']} input tokens · "
            f"{result['output_tokens']} output tokens · {result['precision']}")


# ---------------------------------------------------------------- GPU work
@spaces.GPU(duration=gpu_seconds)
def generate(text, method):
    """Runs on the ZeroGPU GPU. Yields partial text, then the final result dict."""
    text, tokens, prompt_tokens, example_ids = engine.prepare(text, method)
    for item in engine.stream(tokens):
        if isinstance(item, dict):
            item.update(original=text, method=method, example_ids=example_ids,
                        prompt_tokens=prompt_tokens)
        yield item


# ---------------------------------------------------------------- handlers
def on_rewrite(text, method, want_highlight):
    """Generator: streams the revision, then attaches detector results."""
    blank = dict(rewritten=gr.update(value="", visible=True), signals=IDLE_SIGNALS,
                 highlights=gr.update(visible=False), details="",
                 run="Rewrite in progress.", result=None, cache=None)
    try:
        clean = validate_text(text)
        if method == "examples":
            # Fail before taking a GPU slot if the examples are unusable.
            from src.rewrite_engine import build_messages, load_examples
            from src.hosted_engine import examples_path
            build_messages(clean, method, load_examples(examples_path()))
        if engine.model is None:
            raise RuntimeError("The rewrite model is not loaded. " + (engine.load_error or ""))
    except ValueError as error:
        raise gr.Error(str(error))
    except Exception as error:
        yield {**blank, "status": friendly(error), "run": "The last rewrite did not finish.",
               "signals": "Rewrite failed; no detection result."}
        return

    yield {**blank, "status": "Waiting for a GPU… the first request after a restart can take longer.",
           "signals": "Waiting for the rewrite…"}
    result = None
    try:
        for item in generate(clean, method):
            if isinstance(item, dict):
                result = item
            else:
                yield {**blank, "rewritten": gr.update(value=item, visible=True), "status": "Writing…",
                       "signals": "Waiting for the rewrite…"}
    except gr.Error as error:
        yield {**blank, "status": friendly(error), "run": "The last rewrite did not finish.",
               "signals": "Rewrite failed; no detection result."}
        return
    except Exception as error:
        yield {**blank, "status": friendly(error), "run": "The last rewrite did not finish.",
               "signals": "Rewrite failed; no detection result."}
        return
    if result is None:
        yield {**blank, "status": "Could not rewrite: the model returned nothing.",
               "run": "The last rewrite did not finish."}
        return

    yield {**blank, "rewritten": gr.update(value=result["rewrite"], visible=True),
           "status": "Checking writing signals…",
           "signals": "Checking writing signals…", "run": run_details(result)}
    result["ai_detection"] = {"original": safe_score(result["original"]),
                              "rewrite": safe_score(result["rewrite"])}
    cached = result["ai_detection"]["rewrite"]
    warning = " Output hit its length limit; check the ending." if result["truncated"] else ""
    editor, view = (show_highlights(result["rewrite"], cached) if want_highlight
                    else show_editor())
    yield {"rewritten": gr.update(value=result["rewrite"], visible=editor["visible"]),
           "highlights": view, "status": f"Revision ready.{warning}",
           "signals": signals_line(cached), "details": signals_details(cached),
           "run": run_details(result), "result": result,
           "cache": {"text": result["rewrite"], "data": cached}}


def on_check(edited, want_highlight):
    """Re-run the detector on the edited revision. No GPU, no Qwen."""
    if not (edited or "").strip():
        return IDLE_SIGNALS, "", gr.update(), None
    data = safe_score(edited)
    view = gr.update(value=build_highlights(edited, data), visible=True) if want_highlight else gr.update()
    return signals_line(data), signals_details(data), view, {"text": edited, "data": data}


def on_toggle_highlight(want_highlight, edited, cache):
    """ON: replace the editor with the highlighted view. OFF: back to the editor."""
    if not want_highlight:
        editor, view = show_editor()
        return editor, view, gr.update(), gr.update()
    if cache and cache.get("text") == edited:
        editor, view = show_highlights(edited, cache["data"])
        return editor, view, gr.update(), gr.update()
    if not (edited or "").strip():
        editor, view = show_highlights("", None)
        return editor, view, gr.update(), gr.update()
    line, details, view, _ = on_check(edited, True)
    return gr.update(visible=False), view, line, details


def on_edit(edited, cache):
    """Typing in the revision makes the last detector result stale."""
    if cache and cache.get("text") == edited:
        return gr.update(), gr.update()
    line = EDITED_SIGNALS if (edited or "").strip() else IDLE_SIGNALS
    return line, ""


def on_clear():
    return ("", gr.update(value="", visible=True), gr.update(visible=False), False,
            "Paste a draft to begin.", IDLE_SIGNALS, "",
            "No rewrite in this session.", None, None, gr.update(interactive=False))


def on_export(result, edited, cache):
    if not result:
        raise gr.Error("Rewrite something first.")
    edited_detection = cache["data"] if cache and cache.get("text") == edited else None
    saved = {**result, "edited_rewrite": edited, "edited_ai_detection": edited_detection,
             "saved_at": datetime.now(timezone.utc).isoformat()}
    path = EXPORT_DIR / f"rewrite-{int(time.time() * 1000)}.json"
    path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------- interface
CSS = """
/* Gradio component wrappers default to width:100%. Explicit grids prevent
   those wrappers from overflowing into the neighboring editor. */
.gradio-container { width: 100% !important; max-width: 1520px !important;
  margin: 0 auto !important; padding: 0 32px 28px !important;
  font-family: "Segoe UI", Arial, sans-serif; }
.topbar, .panel-head, .panel-foot, .action-bar, .voice-control,
.desk-nav, .status-row { display: grid !important; }
.topbar > *, .panel-head > *, .panel-foot > *, .action-bar > *,
.voice-control > *, .desk-nav > *, .status-row > * {
  width: auto !important; min-width: 0 !important; max-width: 100% !important;
  box-sizing: border-box !important; }
.topbar { grid-template-columns: minmax(0,1fr) auto; align-items: center !important;
  min-height: 84px; gap: 20px !important; margin-bottom: 12px; }
.desk-nav { grid-template-columns: 40px 80px; gap: 6px !important; align-items: center !important; }
.desk-nav button { height: 36px !important; min-height: 36px !important;
  border: 0 !important; background: transparent !important; box-shadow: none !important;
  color: var(--body-text-color) !important; padding: 6px !important; }
.desk-icon { font-size: 22px !important; }
.desk-settings { font-size: 12px !important; }
.identity { display: flex; align-items: center; gap: 14px; }
.mark { display: grid; place-items: center; width: 32px; height: 32px;
  background: var(--body-text-color); color: var(--body-background-fill);
  border-radius: 3px; font: italic 27px/1 Georgia, serif; }
.desk-title { margin: 0; font: normal 28px/1.2 Georgia, serif; letter-spacing: -.8px;
  color: var(--body-text-color); }
.period { color: #a69579; }
.editors { display: grid !important; grid-template-columns: minmax(0,1fr) minmax(0,1fr);
  gap: 36px !important; align-items: start !important; }
.editors > * { width: 100% !important; min-width: 0 !important; max-width: 100% !important; }
.panel-head { grid-template-columns: minmax(0,1fr) 60px; align-items: center !important;
  gap: 12px !important; min-height: 44px; margin-bottom: 4px; }
.panel-title { margin: 0; font: normal 26px/1.3 Georgia, serif; color: var(--body-text-color); }
.quiet { border: 0 !important; background: transparent !important; box-shadow: none !important;
  color: var(--body-text-color-subdued) !important; font-size: 12px !important;
  padding: 8px !important; min-height: 32px !important; }
.editors textarea, .hl-box { box-sizing: border-box !important;
  width: 100% !important; height: 420px !important; min-height: 420px !important; max-height: 420px !important;
  overflow-y: auto !important; overflow-x: hidden !important; resize: none !important;
  padding: 28px 30px !important; border: 0 !important; border-radius: 8px !important;
  background: var(--input-background-fill) !important; color: var(--body-text-color) !important;
  font: 16px/1.8 "Segoe UI", Arial, sans-serif !important;
  box-shadow: none !important; scrollbar-width: thin; }
.editors textarea:focus { outline: 1px solid var(--input-border-color-focus) !important; outline-offset: -1px; }
.hl-box { white-space: pre-wrap; overflow-wrap: anywhere; }
.hl-box mark { background: color-mix(in srgb, #bd984e 30%, transparent); color: inherit; border-radius: 2px; }
.hl-empty { color: var(--body-text-color-subdued) !important; }
.panel-foot { grid-template-columns: minmax(0,1fr) auto; gap: 12px !important;
  align-items: center !important; min-height: 30px; }
.count, .hint, .small, .status, .signal-line, .keyboard-note, .revision-helper,
.count *, .hint *, .small *, .status *, .signal-line *, .keyboard-note *, .revision-helper * {
  font: 12px/1.6 "Segoe UI", Arial, sans-serif !important;
  color: var(--body-text-color-subdued); }
.count p, .status p, .keyboard-note p { margin: 0 !important; }
.count.right { text-align: right; }
.action-bar { grid-template-columns: minmax(0,1fr) 138px; gap: 14px !important;
  align-items: center !important; margin-top: 12px; }
.voice-control { grid-template-columns: 76px minmax(0,200px); align-items: center !important;
  gap: 10px !important; }
.voice-label { white-space: nowrap; font: 12px/1.5 "Segoe UI", Arial, sans-serif;
  color: var(--body-text-color-subdued); }
#writing-style { width: 100% !important; max-width: 200px !important; }
.voice-control input { font-size: 13px !important; min-width: 0 !important; }
.voice-control .wrap-inner { min-height: 42px !important; border: 1px solid var(--input-border-color) !important;
  border-radius: 6px !important; background: var(--input-background-fill) !important;
  transition: border-color 120ms ease, background-color 120ms ease; }
#rewrite-button { width: 100% !important; min-width: 0 !important; min-height: 42px !important;
  border-radius: 6px !important; padding: 10px 16px !important; font-size: 13px !important; }
.hint { margin-top: 6px; }
.revision-helper { margin-top: 2px; }
.signals { box-sizing: border-box; padding: 15px !important; margin-top: 12px;
  border: 1px solid var(--input-border-color) !important; border-radius: 8px;
  gap: 7px !important; background: transparent !important; }
.signals-title { margin: 0; font: normal 18px/1.3 Georgia, serif; color: var(--body-text-color); }
.signal-line, .signal-line * { color: var(--body-text-color) !important; }
.signal-line p { margin: 0 !important; }
.signal-toolbar { display: grid !important; grid-template-columns: repeat(2,minmax(0,1fr));
  gap: 8px !important; margin-top: 4px; }
.signal-toolbar > button { width: 100% !important; min-width: 0 !important;
  height: 36px !important; min-height: 36px !important; border-radius: 6px !important;
  padding: 8px 10px !important; font-size: 12px !important; }
.status-row { grid-template-columns: minmax(0,1fr) auto; gap: 16px !important; margin-top: 18px; }
.keyboard-note { text-align: right; }
#score-details { margin-top: 10px; width: 100% !important; min-width: 0 !important;
  overflow: hidden !important; border: 0 !important; background: transparent !important; }
#score-details p, #score-details span { font-size: 12px !important; overflow-wrap: anywhere; }
#score-details [data-testid="accordion-content"] { min-width: 0; overflow-wrap: anywhere; }
#score-details .label-wrap { padding: 8px 0 !important; }
.export-button { width: auto !important; }
/* Theme-aware polish: identical editor metrics, readable metadata and clear controls. */
.editors textarea, .hl-box {
  scrollbar-color: var(--body-text-color-subdued) var(--input-background-fill);
  scrollbar-gutter: stable; }
.editors textarea::-webkit-scrollbar, .hl-box::-webkit-scrollbar { width: 8px; height: 8px; }
.editors textarea::-webkit-scrollbar-track, .hl-box::-webkit-scrollbar-track {
  background: var(--input-background-fill); border-radius: 8px; }
.editors textarea::-webkit-scrollbar-thumb, .hl-box::-webkit-scrollbar-thumb {
  background: var(--body-text-color-subdued); border: 2px solid var(--input-background-fill); border-radius: 8px; }
.editors textarea::placeholder { color: var(--body-text-color-subdued) !important; opacity: 1; }
.count, .hint, .status, .revision-helper, .keyboard-note,
.count *, .hint *, .status *, .revision-helper *, .keyboard-note * {
  color: var(--body-text-color-subdued) !important; }
.quiet:not(:disabled) { color: var(--body-text-color-subdued) !important; cursor: pointer; }
.quiet:hover:not(:disabled), .desk-nav button:hover:not(:disabled) {
  background: var(--button-secondary-background-fill-hover) !important;
  color: var(--body-text-color) !important; }
#writing-style .wrap-inner:hover { border-color: var(--body-text-color-subdued) !important; }
#writing-style input { color: var(--body-text-color) !important; cursor: pointer !important; }
#writing-style:focus-within .wrap-inner { border-color: var(--input-border-color-focus) !important; }
.signal-toolbar > button { border: 1px solid var(--input-border-color) !important;
  cursor: pointer; transition: background-color 120ms ease, border-color 120ms ease; }
.signal-toolbar > button.secondary:not(:disabled) {
  color: var(--button-secondary-text-color) !important;
  background: var(--button-secondary-background-fill) !important; }
.signal-toolbar > button.secondary:hover:not(:disabled) {
  background: var(--button-secondary-background-fill-hover) !important;
  border-color: var(--body-text-color-subdued) !important; }
.signal-toolbar > button.primary { border-color: var(--button-primary-background-fill) !important; }
button:disabled { cursor: not-allowed !important; }
.quiet:focus-visible, .desk-nav button:focus-visible, .signal-toolbar > button:focus-visible,
#rewrite-button:focus-visible, #score-details .label-wrap:focus-visible {
  outline: 2px solid var(--body-text-color-subdued) !important; outline-offset: 2px; }
@media (max-width: 420px) {
 .editors textarea, .hl-box { padding: 20px !important; }
}
footer { opacity: .5; }
@media (max-width: 900px) {
 .editors { gap: 24px !important; }
 .action-bar { grid-template-columns: minmax(0,1fr); }
 .voice-control { grid-template-columns: 76px minmax(0,200px); }
 #rewrite-button { justify-self: end; width: 138px !important; }
}
@media (max-width: 700px) {
 .gradio-container { padding: 0 18px 24px !important; }
 .editors { grid-template-columns: minmax(0,1fr); gap: 28px !important; }
 .editors textarea, .hl-box { height: 340px !important; min-height: 340px !important; max-height: 340px !important; }
 .action-bar { grid-template-columns: minmax(0,1fr) 128px; }
 #rewrite-button { width: 100% !important; }
 .keyboard-note { display: none !important; }
}
@media (max-width: 420px) {
 .action-bar, .signal-toolbar { grid-template-columns: minmax(0,1fr); }
 #rewrite-button { width: 100% !important; }
 .desk-title { font-size: 24px; }
}
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.stone, neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.Font("Segoe UI"), gr.themes.Font("Arial"), gr.themes.Font("sans-serif")],
).set(
    body_background_fill="#f7f5f0", body_background_fill_dark="#1c1c1b",
    body_text_color="#292825", body_text_color_dark="#ece7de",
    body_text_color_subdued="#625f58", body_text_color_subdued_dark="#bbb7ae",
    block_background_fill="#f7f5f0", block_background_fill_dark="#1c1c1b",
    block_border_width="0px",
    input_background_fill="#efede7", input_background_fill_dark="#252522",
    input_border_width="0px",
    input_border_color="#d9d5cb", input_border_color_dark="#45443e",
    input_border_color_focus="#a39580", input_border_color_focus_dark="#a89a80",
    button_primary_background_fill="#292825", button_primary_background_fill_dark="#ece7de",
    button_primary_text_color="#f7f5f0", button_primary_text_color_dark="#1c1c1b",
    button_primary_background_fill_hover="#3f3d39", button_primary_background_fill_hover_dark="#d6d1c7",
    button_secondary_background_fill="#e9e6de", button_secondary_background_fill_dark="#30302c",
    button_secondary_background_fill_hover="#dfdbd1", button_secondary_background_fill_hover_dark="#3a3a34",
    button_secondary_text_color="#292825", button_secondary_text_color_dark="#ece7de",
    border_color_primary="#d9d5cb", border_color_primary_dark="#45443e",
    layout_gap="10px", block_padding="0px", form_gap_width="0px",
)

THEME_JS = """
() => {
  const apply = (dark) => {
    document.body.classList.toggle('dark', dark);
    document.documentElement.classList.toggle('dark', dark);
    document.querySelectorAll('gradio-app, .gradio-container').forEach(el => el.classList.toggle('dark', dark));
    document.documentElement.style.colorScheme = dark ? 'dark' : 'light';
  };
  let saved = null;
  try { saved = localStorage.getItem('writing-desk-theme'); } catch (_) {}
  if (saved === 'dark' || saved === 'light') apply(saved === 'dark');
  else if (!window.location.search.includes('__theme')) apply(true);
  window.writingDeskToggle = () => {
    const dark = !document.body.classList.contains('dark');
    apply(dark);
    try { localStorage.setItem('writing-desk-theme', dark ? 'dark' : 'light'); } catch (_) {}
  };
  window.writingDeskSettings = () => {
    const acc = document.querySelector('#score-details button, #score-details .label-wrap');
    if (acc) { acc.click(); acc.scrollIntoView({behavior: 'smooth', block: 'center'}); }
  };
  document.addEventListener('keydown', (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
      const button = document.querySelector('#rewrite-button button, button#rewrite-button');
      if (button && !button.disabled) { event.preventDefault(); button.click(); }
    }
  });
}
"""

HEADER_HTML = """
<div class="identity"><span class="mark" aria-hidden="true">w</span>
<h1 class="desk-title">Writing desk<span class="period">.</span></h1></div>
"""
NAV_HTML = """
<div class="nav-right">
<button class="icon-button" type="button" title="Switch light / dark" aria-label="Switch light / dark" onclick="window.writingDeskToggle && window.writingDeskToggle()">
<svg class="sun-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5"/></svg>
<svg class="moon-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><path d="M20.5 14A8.5 8.5 0 0 1 10 3.5 8.5 8.5 0 1 0 20.5 14Z"/></svg></button>
<button class="icon-button" type="button" title="Settings" aria-label="Open settings" onclick="window.writingDeskSettings && window.writingDeskSettings()">
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><path d="M4 7h16M4 17h16"/><circle cx="9" cy="7" r="3" fill="var(--body-background-fill)"/><circle cx="15" cy="17" r="3" fill="var(--body-background-fill)"/></svg></button>
</div>
"""
COPY_ICON = ('<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.3" aria-hidden="true" '
             'style="width:16px;height:16px;vertical-align:-3px;margin-right:6px"><rect x="7" y="7" width="10" height="10" rx="1.5"/>'
             '<path d="M12 7V3H3v9h4"/></svg>')
MODEL_INFO = (f"**Model** {MODEL_ID}  \n**Method** Prompting with optional writing examples  \n"
              "**Training** No DPO adapter loaded  \n"
              f"**Processing** Hugging Face ZeroGPU, up to {MAX_NEW_TOKENS} new tokens, greedy decoding  \n"
              "**Detector** Local TF-IDF + logistic regression classifier (experimental, CPU)")

with gr.Blocks(title="Writing desk", fill_width=True) as demo:
    result_state = gr.State(None)
    cache_state = gr.State(None)
    highlight_state = gr.State(False)

    with gr.Row(elem_classes="topbar"):
        gr.HTML(HEADER_HTML, css_template=CSS, apply_default_css=False)
        with gr.Row(scale=0, elem_classes="desk-nav"):
            theme_button = gr.Button("◐", scale=0, min_width=40, elem_classes="desk-icon")
            settings_button = gr.Button("Settings", scale=0, min_width=80, elem_classes="desk-settings")

    with gr.Row(elem_classes="editors", equal_height=False):
        # Original ---------------------------------------------------------
        with gr.Column():
            with gr.Row(elem_classes="panel-head"):
                gr.HTML('<div class="panel-title">Original</div>', css_template=CSS, apply_default_css=False)
                clear_button = gr.Button("Clear", size="sm", scale=0, min_width=60, elem_classes="quiet")
            original = gr.Textbox(lines=14, max_lines=14, max_length=20000, show_label=False,
                                  container=False, placeholder="Paste your draft here…", autofocus=True)
            with gr.Row(elem_classes="panel-foot"):
                in_count = gr.Markdown("0 words", elem_classes="count")
                gr.Markdown("1,000 word limit", elem_classes="count right")
            with gr.Row(elem_classes="action-bar"):
                with gr.Row(elem_classes="voice-control", scale=0):
                    gr.HTML('<span class="voice-label">Writing style</span>', min_width=90, css_template=CSS, apply_default_css=False)
                    method = gr.Dropdown(METHOD_CHOICES, value="examples", show_label=False,
                                         container=False, scale=0, min_width=0,
                                         filterable=False, allow_custom_value=False,
                                         elem_id="writing-style")
                rewrite_button = gr.Button("Rewrite ↗", variant="primary", scale=0, min_width=138,
                                           elem_id="rewrite-button", elem_classes="primary-btn")
            hint = gr.Markdown(HINTS["examples"], elem_classes="hint")
            with gr.Accordion("Score details", open=False, elem_id="score-details"):
                details = gr.Markdown("No detector result yet.")
                gr.Markdown(MODEL_INFO)
                gr.Markdown("**Last rewrite**")
                run_box = gr.Markdown("No rewrite in this session.")
                export_button = gr.DownloadButton("Download experiment data (.json)", size="sm",
                                                  interactive=False, elem_classes="export-button")
                gr.Markdown("Check names, facts and meaning before using a rewrite. Export includes "
                            "the input, model output, your edits and run settings. Editing a revision "
                            "does not retrain the model.", elem_classes="small")
        # Revision ---------------------------------------------------------
        with gr.Column():
            with gr.Row(elem_classes="panel-head"):
                gr.HTML('<div class="panel-title">Revision</div>', css_template=CSS, apply_default_css=False)
                copy_button = gr.Button("Copy", size="sm", scale=0, min_width=60, elem_classes="quiet",
                                        elem_id="copy-button", interactive=False)
            rewritten = gr.Textbox(lines=14, max_lines=14, show_label=False, container=False,
                                   interactive=True, placeholder="Your revision will appear here.",
                                   elem_id="revision-box")
            highlights = gr.HTML(build_highlights("", None), visible=False, container=False, css_template=CSS, apply_default_css=False)
            with gr.Row(elem_classes="panel-foot"):
                out_count = gr.Markdown("0 words", elem_classes="count")
                gr.Markdown("Editable", elem_classes="count right")
            gr.Markdown("Review and edit anything that doesn't sound like you", elem_classes="revision-helper")
            with gr.Column(elem_classes="signals"):
                gr.HTML('<h3 class="signals-title">Writing signals</h3>', css_template=CSS, apply_default_css=False)
                signals = gr.Markdown(IDLE_SIGNALS, elem_classes="signal-line")
                with gr.Row(elem_classes="signal-toolbar"):
                    check_button = gr.Button("Check edited output", size="sm", scale=0, min_width=150, interactive=False)
                    highlight_button = gr.Button("Highlight sentences", size="sm", scale=0, min_width=150, interactive=False)

    with gr.Row(elem_classes="status-row"):
        status = gr.Markdown("Paste a draft to begin.", elem_classes="status")
        gr.Markdown("Ctrl / ⌘ + Enter to rewrite", elem_classes="keyboard-note")

    # Native Gradio buttons keep theme controls styled and keyboard accessible.
    theme_button.click(fn=None, js="() => window.writingDeskToggle?.()", queue=False)
    settings_button.click(fn=None, js="() => window.writingDeskSettings?.()", queue=False)
    rewritten.change(
        lambda text: tuple(gr.update(interactive=bool((text or '').strip())) for _ in range(3)),
        rewritten, [check_button, highlight_button, copy_button], queue=False,
        show_progress="hidden")

    # Wiring -------------------------------------------------------------
    rewrite_outputs = {"rewritten": rewritten, "status": status, "signals": signals,
                       "highlights": highlights, "details": details, "run": run_box,
                       "result": result_state, "cache": cache_state}

    def rewrite_adapter(text, method_value, want_highlight):
        for update in on_rewrite(text, method_value, want_highlight):
            out = [update.get(k, gr.update()) for k in rewrite_outputs]
            out.append(gr.update(interactive=update.get("result") is not None)
                       if "result" in update else gr.update())
            yield tuple(out)

    rewrite_button.click(
        rewrite_adapter, [original, method, highlight_state],
        list(rewrite_outputs.values()) + [export_button],
        concurrency_limit=1, concurrency_id="gpu", show_progress="hidden")

    original.change(count_words, original, in_count, show_progress="hidden")
    rewritten.change(count_words, rewritten, out_count, show_progress="hidden")
    rewritten.input(on_edit, [rewritten, cache_state], [signals, details], show_progress="hidden")
    method.change(lambda m: HINTS[m], method, hint, show_progress="hidden")
    check_button.click(on_check, [rewritten, highlight_state],
                       [signals, details, highlights, cache_state], concurrency_limit=2,
                       concurrency_id="cpu", show_progress="hidden")

    def toggle_highlight(active, edited, cache):
        active = not active
        editor, view, line, det = on_toggle_highlight(active, edited, cache)
        button = gr.update(variant="primary" if active else "secondary")
        return active, button, editor, view, line, det

    highlight_button.click(toggle_highlight, [highlight_state, rewritten, cache_state],
                           [highlight_state, highlight_button, rewritten, highlights, signals, details],
                           concurrency_limit=2, concurrency_id="cpu", show_progress="hidden")

    def clear_all():
        values = on_clear()  # original, rewritten, highlights, toggle, status, signals, details, run, result, cache, export
        return values[:3] + (False, gr.update(variant="secondary")) + values[4:]

    clear_button.click(clear_all, None,
                       [original, rewritten, highlights, highlight_state, highlight_button, status,
                        signals, details, run_box, result_state, cache_state, export_button])
    copy_button.click(lambda: "Rewrite copied.", None, status, show_progress="hidden",
                      js="() => { const t = document.querySelector('#revision-box textarea');"
                         " if (t && navigator.clipboard) navigator.clipboard.writeText(t.value); return []; }")
    export_button.click(on_export, [result_state, rewritten, cache_state], export_button)

if __name__ == "__main__":
    demo.queue(max_size=30, default_concurrency_limit=1)
    demo.launch(theme=THEME, css=CSS, js=THEME_JS, show_error=True, footer_links=[],
                server_name="0.0.0.0" if ON_SPACE else "127.0.0.1",
                server_port=int(os.environ.get("PORT", "7860")))
