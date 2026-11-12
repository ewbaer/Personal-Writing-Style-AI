"""Hugging Face entry point. Build models once, then inject services into the UI."""
import os
import spaces  # Must precede any torch import on ZeroGPU.
from src.config import SETTINGS
from src.runtime import production_controller
from src.ui import build_app
from src.ui_assets import THEME, CSS, THEME_JS

controller = production_controller()
demo = build_app(controller)

if __name__ == '__main__':
    demo.queue(max_size=SETTINGS.queue_size, default_concurrency_limit=1)
    demo.launch(theme=THEME, css=CSS, js=THEME_JS, show_error=False, footer_links=[],
                server_name='0.0.0.0' if os.environ.get('SPACE_ID') else '127.0.0.1',
                server_port=int(os.environ.get('PORT', '7860')))
