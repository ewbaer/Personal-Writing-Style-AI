---
title: Writing desk
emoji: ✍️
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 6.30.0
python_version: 3.12
app_file: app.py
pinned: false
short_description: Rewrite a draft in Ethan's style (Qwen2.5-3B, ZeroGPU)
models:
  - Qwen/Qwen2.5-3B-Instruct
tags:
  - text-rewriting
  - writing-style
  - zerogpu
---

# Writing desk

Rewrite a draft with Qwen2.5-3B-Instruct, edit the revision, and check writing signals with a trained TF-IDF and logistic regression classifier. The app supports optional example prompting and light/dark themes.

[Open the app](https://huggingface.co/spaces/ewbaer/personal-writing-style-ai)

## Current implementation

- Gradio UI with original/revision editors, streaming, word counts, highlights, copying and JSON exports.
- One queued rewrite at a time on Hugging Face ZeroGPU. Multiple visitors have separate session state.
- A Cancel button and cancellation when Clear is selected. Generation stops cooperatively at a token boundary.
- A unique run ID, prompt version, settings, timestamps, completed steps and safe error records for each rewrite.
- Detector checks run on CPU. A failed check keeps the rewrite available for manual rechecking.
- Explicit JSON downloads include model output and user edits separately. Editing does not retrain the model.

The rewrite model still uses prompting. The desktop SFT adapter is not loaded and DPO training has not started in this Space. The detector score is an experimental classifier output, not proof of authorship. Sentence scores and passage scores can disagree.

## Code structure

| Location | Purpose |
| --- | --- |
| `app.py` | Creates production services and launches the UI. |
| `src/ui.py`, `src/ui_assets.py`, `web/` | Layout, theme, controls and styles. |
| `src/controller.py`, `src/presentation.py` | Gradio updates and display formatting. |
| `src/workflow.py`, `src/records.py` | Ordered processing, run state and retry rules. |
| `src/contracts.py` | Rewrite, detector and export service interfaces. |
| `src/config.py`, `src/validation.py`, `src/prompts.py` | Shared limits, validation and versioned prompts. |
| `src/runtime.py`, `src/hosted_engine.py` | Startup model loading and ZeroGPU streaming. |
| `src/ai_detector.py`, `models/ai_detector.joblib` | Saved CPU detector and sentence checks. |
| `src/exports.py` | UUID-named user-controlled downloads and expiry cleanup. |
| `tests/` | Workflow, detector, UI and tiny-model checks. |

[Architecture checks and request flow](docs/ARCHITECTURE.md) · [Test report](docs/TEST_REPORT.md) · [Upload instructions](docs/UPLOAD.md)

## Limits and state

Input must contain at most 1,000 words and 20,000 characters. Prompts including examples must fit within 7,000 tokens. Rewrites allow 1,200 new tokens by default. Cut-off output is marked. Detector passages need at least 50 words; fragments under eight words are unassessed.

Run records live in browser-session state, not a history database. Restarting or refreshing can lose that state. Export results to keep them. Export files are temporary and cleaned up after their retention period; download links are not a protected account history. Do not share private export links. Stop is cooperative; an in-flight operation may finish before generation ends.

## Examples and settings

The public example file is `data/splits/train.csv`. No API key is needed for the included model and examples. For private examples, remove that CSV and configure `WRITING_EXAMPLES_REPO` and the read-only `HF_TOKEN` secret in Space settings.

- `WRITING_MODEL_ID`: defaults to `Qwen/Qwen2.5-3B-Instruct`.
- `WRITING_MAX_NEW_TOKENS`: defaults to `1200`.

The model loads once at startup in full precision appropriate to the device. Local execution uses your device, not the hosted GPU. Keep the hosted environment separate from the desktop training environment.

## Tests

Install the Space dependencies, Gradio 6.30.0 and pytest in a separate environment, then run:

```shell
python -m pytest tests/test_hosted.py -q
```

Those tests inject small services and use the real saved detector; they do not download Qwen. `tests/make_tiny_model.py` creates a random tiny model for a real Transformers streaming check. Its meaningless output must not be used to judge rewriting quality. See the test report for what was actually run.
