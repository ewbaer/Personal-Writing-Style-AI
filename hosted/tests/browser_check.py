"""Drive the running app in a headless browser. Start the app first:

    WRITING_MODEL_ID=/tmp/tiny-qwen WRITING_MAX_NEW_TOKENS=40 python app.py

then:  python tests/browser_check.py <screenshot_dir>
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

shots = Path(sys.argv[1]); shots.mkdir(parents=True, exist_ok=True)
PASSAGE = " ".join(["I spent the weekend fixing a bug in our class project and learned a lot about debugging."] * 7)

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.goto("http://127.0.0.1:7860", wait_until="domcontentloaded")
    page.wait_for_selector("textarea", timeout=60000)
    assert page.get_by_text("Writing desk").first.is_visible()
    boxes = page.locator("textarea")
    original, revision = boxes.nth(0), boxes.nth(1)

    # Empty input shows a clear error, not a crash.
    page.get_by_role("button", name="Rewrite").click()
    page.wait_for_timeout(1500)
    page.screenshot(path=str(shots / "01-empty-error.png"))
    body = page.inner_text("body")
    assert "Paste some text first" in body, body[-500:]

    # Oversized input is rejected before any generation.
    original.fill("word " * 1001)
    page.get_by_role("button", name="Rewrite").click()
    page.wait_for_timeout(1500)
    assert "1,000 words or fewer" in page.inner_text("body")

    # Real rewrite through the Gradio queue (tiny random model, nonsense output).
    original.fill(PASSAGE)
    page.wait_for_timeout(1500)
    assert "119 words" in page.inner_text("body")
    page.get_by_role("button", name="Rewrite").click()
    page.wait_for_function("() => document.body.innerText.includes('Revision ready')", timeout=120000)
    page.screenshot(path=str(shots / "02-revision-ready.png"))
    text = revision.input_value()
    assert text.strip(), "revision box is empty"
    page.get_by_text("Workspace settings").click()
    page.wait_for_timeout(800)
    body = page.inner_text("body")
    assert "s generation" in body and "to first token" in body, body[-800:]
    page.get_by_text("Workspace settings").click()

    # Detector on the edited output and highlights.
    revision.fill(PASSAGE[:-1])
    revision.press("End")
    revision.type(PASSAGE[-1])  # a real keystroke, like a user editing
    page.wait_for_timeout(1200)
    assert "Output edited" in page.inner_text("body")
    page.get_by_role("button", name="Check edited output").click()
    page.wait_for_function("() => document.body.innerText.includes('flagged as AI-like')", timeout=30000)
    page.get_by_text("Highlight sentences").click()
    page.wait_for_timeout(1500)
    page.screenshot(path=str(shots / "03-highlights.png"))
    assert page.locator(".highlight-box").is_visible()

    # Theme toggle and settings panel.
    page.get_by_role("button", name="Light / dark").click()
    page.wait_for_timeout(500)
    page.get_by_text("Workspace settings").click()
    page.wait_for_timeout(800)
    page.screenshot(path=str(shots / "04-settings-other-theme.png"), full_page=True)
    assert "Download experiment data" in page.inner_text("body")
    print("browser check passed")
    browser.close()
