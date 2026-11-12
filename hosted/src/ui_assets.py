"""Theme and page identity. Editor styling lives in web/."""
import gradio as gr
from .paths import ROOT
CSS = (ROOT/"web/style.css").read_text(encoding="utf-8")
THEME_JS = (ROOT/"web/theme.js").read_text(encoding="utf-8")
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

HEADER_HTML = """
<div class="identity"><span class="mark" aria-hidden="true">w</span>
<h1 class="desk-title">Writing desk<span class="period">.</span></h1></div>
"""
