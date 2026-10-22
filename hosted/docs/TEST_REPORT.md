# What was tested, and what was not

Date: 2026-10-20. Environment: Linux container, 2 CPUs, 7 GB RAM, no GPU, no
access to huggingface.co. Python 3.13, torch 2.13.0, transformers 5.20.2,
Gradio 6.30.0, spaces 0.51.3, scikit-learn 1.7.2, joblib 1.5.2.

## Actually tested here

Because the real Qwen weights could not be downloaded, a tiny Qwen2-architecture
model with random weights and a 1,200-token tokenizer was built
(`tests/make_tiny_model.py`). It produces nonsense text, but it runs through the
exact same transformers and Gradio code as the real model, so these checks are
real code-path tests, not mocks.

Automated tests (`tests/test_hosted.py`, 11 passed):

- The supplied `models/ai_detector.joblib` loads with the pinned
  scikit-learn/joblib versions and returns a score, sentence spans, flagged
  percentage and the under-50-words message. No detector was retrained or
  replaced.
- The supplied `data/splits/train.csv` loads with 6 rows. "My examples" picks
  3 examples and excludes an exact match of the input, same as the local app.
- Validation rejects empty, non-string, 1,001-word and 20,001-character inputs.
- Chat template, tokenization, streaming generation in a background thread,
  time-to-first-token measurement, output token count, truncation flag and
  empty-output error all run on the installed library versions.
- The full rewrite handler streams partial text, then attaches detector results
  for original and rewrite, and the export JSON includes the edited text.
- Stale-result handling: editing the revision marks signals as outdated;
  unchanged text does not.
- The Gradio app starts with the custom theme, CSS and theme-toggle script and
  serves its page and config.

Headless browser test (`tests/browser_check.py`, Chromium, passed):

- Empty input shows "Paste some text first." Oversized input shows the
  1,000-word message. Neither reaches generation.
- Word counts update while typing.
- Rewrite streams text into the editable Revision box and ends with "Revision
  ready", including the "hit its length limit" warning when truncated.
- "Check edited output" updates the signals line; "Highlight sentences" shows
  the flagged sentence list; light/dark toggle works; Workspace settings shows
  model, first-token and generation timings, and the JSON download button.
- With two other requests ahead, the page shows "queue: 2/2" with an estimate.

Queue test (`tests/queue_check.py`, through the Gradio API): three requests
sent at the same time all finished, one at a time (5.0 s, 9.5 s, 13.5 s), with
the later ones reporting a queued state.

## Not tested, must be verified on Hugging Face

- Qwen2.5-3B-Instruct output quality and speed on ZeroGPU. All timing here was
  from a random tiny model on CPU and says nothing about real speed.
- Whether `@spaces.GPU` on a generator function streams correctly on the
  current ZeroGPU runtime. The pattern follows the official docs (model placed
  on CUDA at startup, decorated function, generator supported by the spaces
  package), but it has not run on real ZeroGPU hardware.
- The daily quota message. `friendly()` turns ZeroGPU quota errors into a plain
  sentence, but the exact error text Hugging Face sends was not observed.
- Build time, cold start time and model download time on the Space.
- Real multi-user load with classmates. The queue was tested with 3 API clients
  on one machine, not with 5 to 15 people on the hosted Space.
- Windows laptop behaviour of the hosted app (it needs more than 6 GB VRAM in
  bf16, so it is not expected to run on the RTX 4050 laptop).

## Known differences from the local Flask app

- Highlights appear as a separate list under the editor (Gradio's
  HighlightedText) rather than as a transparent overlay inside the textarea.
  The textarea stays the only editor.
- Light/dark is a small button at the top right instead of an icon; the choice
  is saved in the browser.
- Max new tokens is 1,200 instead of 1,600 to keep GPU time per request down.
- Multiple requests queue instead of being rejected with "A rewrite is already
  running".
- Output streams in, so time to first token is measured and shown.
- Model loads at startup, not on first rewrite, as ZeroGPU recommends.
