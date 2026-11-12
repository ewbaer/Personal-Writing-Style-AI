"""Build and wire the interface using injected services."""
import gradio as gr
import os
from .config import SETTINGS
from .paths import ROOT
from .ui_assets import CSS, THEME_JS, THEME, HEADER_HTML
from .presentation import *

METHOD_CHOICES = [("My examples", "examples"), ("General rewrite", "basic")]
HINTS = {"examples": "Uses up to 3 examples from Ethan's writing.",
         "basic": "Rewrites with general instructions, without examples."}


def build_app(controller):
    model_routes = getattr(controller, "model_routes", {"qwen": controller})
    model_choices = getattr(controller, "model_choices", [("Qwen · ARM 1", "qwen")])
    experiment_prompt = getattr(controller, "experiment_prompt", SETTINGS.experiment_prompt)
    on_check = controller.on_check
    on_edit = controller.on_edit
    on_clear = controller.on_clear
    on_export = controller.on_export
    has_examples = (ROOT / "data/splits/train.csv").is_file() or bool(os.environ.get("WRITING_EXAMPLES_REPO"))
    choices = METHOD_CHOICES if has_examples and not experiment_prompt else METHOD_CHOICES[1:]
    default_method = choices[0][1]
    MODEL_INFO = (f"**Method** {'Shared experiment prompt' if experiment_prompt else 'Prompting with optional writing examples'}  \n"
                  f"**Processing** ZeroGPU, up to {SETTINGS.max_new_tokens} new tokens  \n"
                  "**Detector** TF-IDF + logistic regression on CPU")
    with gr.Blocks(title="Writing desk", fill_width=True, delete_cache=(3600, SETTINGS.export_retention_hours*3600)) as demo:
        result_state = gr.State(None)
        cache_state = gr.State(None)

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
                model_choice = gr.Dropdown(model_choices, value="qwen", label="Model",
                                           filterable=False, allow_custom_value=False,
                                           elem_id="rewrite-model")
                with gr.Row(elem_classes="action-bar"):
                    with gr.Row(elem_classes="voice-control", scale=0):
                        gr.HTML('<span class="voice-label">Writing style</span>', min_width=90, css_template=CSS, apply_default_css=False)
                        method = gr.Dropdown(choices, value=default_method, show_label=False,
                                             container=False, scale=0, min_width=0,
                                             filterable=False, allow_custom_value=False,
                                             elem_id="writing-style")
                    rewrite_button = gr.Button("Rewrite ↗", variant="primary", scale=0, min_width=138,
                                               elem_id="rewrite-button", elem_classes="primary-btn")
                    cancel_button = gr.Button("Cancel", scale=0, min_width=80, elem_classes="quiet")
                hint = gr.Markdown(HINTS[default_method], elem_classes="hint")
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

        with gr.Row(elem_classes="status-row"):
            status = gr.Markdown("Paste a draft to begin.", elem_classes="status")
            gr.Markdown("Ctrl / ⌘ + Enter to rewrite", elem_classes="keyboard-note")

        # Native Gradio buttons keep theme controls styled and keyboard accessible.
        theme_button.click(fn=None, js="() => window.writingDeskToggle?.()", queue=False)
        settings_button.click(fn=None, js="() => window.writingDeskSettings?.()", queue=False)
        rewritten.change(
            lambda text: tuple(gr.update(interactive=bool((text or '').strip())) for _ in range(2)),
            rewritten, [check_button, copy_button], queue=False,
            show_progress="hidden")

        # Wiring -------------------------------------------------------------
        rewrite_outputs = {"rewritten": rewritten, "status": status, "signals": signals,
                           "highlights": highlights, "details": details, "run": run_box,
                           "result": result_state, "cache": cache_state}

        def rewrite_adapter(text, method_value, model_key):
            route = model_routes.get(model_key)
            if route is None:
                raise gr.Error("Choose an available model.")
            for update in route.on_rewrite(text, method_value, False):
                out = [update.get(k, gr.update()) for k in rewrite_outputs]
                out.append(gr.update(interactive=bool(update.get("result", {}).get("rewrite")) and update["result"].get("status") in ("completed", "completed_without_detection"))
                           if "result" in update else gr.update())
                yield tuple(out)

        rewrite_event = rewrite_button.click(
            rewrite_adapter, [original, method, model_choice],
            list(rewrite_outputs.values()) + [export_button],
            concurrency_limit=1, concurrency_id="gpu", show_progress="hidden", api_name="rewrite_adapter")

        def cancel_rewrite(result):
            return controller.on_cancel(result)

        cancel_button.click(cancel_rewrite, result_state,
                            [status, result_state, export_button],
                            cancels=[rewrite_event], queue=False, show_progress="hidden")

        original.change(count_words, original, in_count, show_progress="hidden")
        rewritten.change(count_words, rewritten, out_count, show_progress="hidden")
        rewritten.input(on_edit, [rewritten, cache_state], [signals, details], show_progress="hidden")
        method.change(lambda m: HINTS[m], method, hint, show_progress="hidden")
        def check_edited_output(text):
            return on_check(text, False)

        check_button.click(check_edited_output, [rewritten],
                           [signals, details, highlights, cache_state], concurrency_limit=2,
                           concurrency_id="cpu", show_progress="hidden", api_name="on_check")

        def clear_all():
            values = on_clear()  # original, rewritten, highlights, toggle, status, signals, details, run, result, cache, export
            return values[:3] + values[4:]

        clear_button.click(clear_all, None,
                           [original, rewritten, highlights, status,
                            signals, details, run_box, result_state, cache_state, export_button],
                           cancels=[rewrite_event], queue=False)
        copy_button.click(lambda: "Rewrite copied.", None, status, show_progress="hidden",
                          js="() => { const t = document.querySelector('#revision-box textarea');"
                             " if (t && navigator.clipboard) navigator.clipboard.writeText(t.value); return []; }")
        export_button.click(on_export, [result_state, rewritten, cache_state], export_button)

    return demo

