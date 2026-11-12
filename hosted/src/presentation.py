"""Format results for the UI; no model calls or file writes."""
import html
import re
import gradio as gr

IDLE_SIGNALS = "Rewrite text to see writing signals."
EDITED_SIGNALS = "Output edited · check again to update."

def count_words(text):
    text = (text or "").strip()
    n = len(text.split()) if text else 0
    return f"{n} word" if n == 1 else f"{n} words"

def signals_line(data):
    if not data or not data.get("available"):
        return (data or {}).get("message") or "Detector unavailable."
    return f"Whole-text classifier score: {data['score']:.1f} / 100"

def signals_details(data):
    if not data or not data.get("available"):
        return ""
    return ("The whole-text score is a classifier estimate, not a percentage of AI authorship. "
            f"Experimental sentence check: {data['flagged_count']} of {data['sentence_count']} sentences flagged. "
            f"Words in flagged sentences: {data['flagged_percent']:.1f}%. "
            f"Flagged words: {data['flagged_words']} of {data['total_words']}. "
            f"Unassessed words: {data['unassessed_words']}. Sentences under 8 words are "
            "left unassessed. Sentence cutoff: 50 / 100. Sentence scoring has not been validated.")

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

