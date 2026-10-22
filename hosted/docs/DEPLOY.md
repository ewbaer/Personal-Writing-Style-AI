# Putting the Writing desk on Hugging Face ZeroGPU

This folder is a complete Space. Your local Flask app is untouched; this is a
separate copy built for Hugging Face.

## Before you start

Checked against the ZeroGPU documentation on 2026-10-20:

- ZeroGPU only works with the Gradio SDK. That is why this is a Gradio app.
- Free personal accounts can host up to two ZeroGPU Spaces if the account is
  older than 30 days with a verified email.
- Each visitor gets a daily GPU budget: about 2 minutes anonymous, 5 minutes
  signed in to a free account. The budget counts GPU seconds during generation,
  not time spent reading or editing. Tell classmates to sign in to Hugging Face
  before the demo.
- Supported: Python 3.10 or 3.12, PyTorch 2.8.0 to 2.13.0, Gradio 4 or newer.
  This package uses Python 3.12, torch 2.13.0, Gradio 6.30.0.
- The GPU is half of an RTX Pro 6000 (48 GB), so Qwen runs in full bf16. No
  4-bit loading is needed and bitsandbytes is not installed.

## Privacy check first

A public Space shows every file in its repository. This package includes
`data/splits/train.csv`, which contains your six LinkedIn post pairs. If you do
not want those public, see "Keeping the examples private" at the end before
uploading.

## Step 1: create the Space

1. Go to https://huggingface.co/new-space
2. Owner: your account (ewbaer). Space name: `personal-writing-style-ai`.
3. License: whatever you use on GitHub (MIT is fine).
4. Select the Space SDK: **Gradio**. Template: Blank.
5. Space hardware: **ZeroGPU**. If ZeroGPU is not in the list, your account is
   not eligible yet (needs verified email and 30 days of age).
6. Visibility: **Public**.
7. Click Create Space.

## Step 2: upload the files

Easiest route is the website:

1. Open the new Space and click the **Files** tab.
2. Click **Add file**, then **Upload files**.
3. Drag in these items from the unzipped package, keeping the folder structure:

   ```
   README.md
   app.py
   requirements.txt
   src/            (__init__.py, paths.py, rewrite_engine.py, ai_detector.py, hosted_engine.py)
   models/ai_detector.joblib
   data/splits/train.csv
   docs/           (optional, documentation only)
   tests/          (optional, local checks only)
   ```

   The browser uploader keeps folders when you drag a whole folder in. If it
   flattens them, upload one folder at a time and type the folder name in the
   path box.
4. Write a commit message such as "Initial Gradio ZeroGPU app" and click
   **Commit changes to main**.

Important: the README.md in this package replaces the one Hugging Face
generated. Its top block (`sdk: gradio`, `sdk_version: 6.30.0`,
`python_version: 3.12`) is what tells the Space how to build.

Command line alternative (needs `pip install huggingface_hub` and
`hf auth login` with a write token):

```
cd hf-space
hf upload ewbaer/personal-writing-style-ai . . --repo-type space
```

## Step 3: watch the first build

1. Click the **Logs** tab (or the "Building" badge at the top).
2. The build installs torch and transformers (a few minutes), then starts
   `app.py`, which downloads Qwen2.5-3B-Instruct (about 6 GB, usually one to
   three minutes on Hugging Face's network).
3. The badge turns to **Running** when the page is ready. Startup is allowed up
   to 30 minutes by default.

If the badge says **Runtime error**, open Logs and read the last lines. Common
causes and fixes:

- `No module named ...`: a package is missing from requirements.txt.
- `CUDA`-related errors at startup: confirm the hardware is ZeroGPU in
  Settings. The `@spaces.GPU` code expects it.
- `Detector not trained yet` in the Writing signals panel: `models/ai_detector.joblib`
  did not upload or landed in the wrong folder.
- `Missing data/splits/train.csv`: the examples file did not upload. "General
  rewrite" still works; "My examples" needs the file.
- A version error about scikit-learn when loading the detector: the
  requirements pin scikit-learn 1.7.2 to match the file you trained. Keep that pin.

## Step 4: test it yourself before sharing

Do these in order and write down what you see. These are the tests that have
NOT been run yet, because the hosted GPU cannot be reached from where the
package was built.

1. Open the Space URL in a browser while signed in to Hugging Face.
2. Paste a 200 to 250 word draft. Choose "My examples". Click Rewrite.
   - The status line should say "Waiting for a GPU", then "Writing…" with text
     streaming in, then "Revision ready".
   - Open "Workspace settings" and record the "to first token" and "generation"
     seconds. Do this once cold (first request after a restart) and once warm.
3. Check that "Text flagged as AI-like" appears under Writing signals.
4. Edit a word in the revision. The signals line should change to "Output
   edited". Click "Check edited output" and confirm it updates.
5. Tick "Highlight sentences" and confirm flagged sentences show in a
   highlighted list below the editor.
6. Click "Download experiment data (.json)" and open the file.
7. Try an empty input and a 1,001 word input. Both should show a short error
   without using GPU time.
8. Queue test: open the Space in two browser windows (or ask a classmate),
   click Rewrite in both within a few seconds. The second one should show
   "queue: 2/2" and then run. You can also run
   `python tests/queue_check.py https://ewbaer-personal-writing-style-ai.hf.space 3`
   from your laptop after `pip install gradio_client`.
9. Keep going until a visitor runs out of GPU budget and confirm the message
   says so clearly instead of a raw error.

Only after step 8 works with real people should the README or report claim
support for 5 to 15 users.

## Running the hosted app on your own computer

Useful for checking UI changes without touching the Space. It needs the same
GPU as the local Flask app, because full bf16 weights are used (about 7 GB of
VRAM). On the 6 GB laptop GPU it will fail to load; use the desktop.

```
cd hf-space
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt gradio==6.30.0
python app.py
```

Then open http://127.0.0.1:7860. The `@spaces.GPU` decorator does nothing
outside a Space, so this runs as a normal local Gradio app.

## Keeping the examples private (optional)

1. Create a **private dataset** repository, for example `ewbaer/writing-examples`,
   and upload `train.csv` to its root.
2. In the Space, delete `data/splits/train.csv`.
3. In the Space **Settings** tab, under "Variables and secrets", add:
   - Secret `HF_TOKEN`: a read token from https://huggingface.co/settings/tokens
   - Variable `WRITING_EXAMPLES_REPO`: `ewbaer/writing-examples`
4. Restart the Space. The app downloads the CSV at request time using the
   token. The posts are still sent to the model as part of the prompt, but they
   are no longer browsable in the Space files.

## Settings you can change without editing code

Space variables (Settings tab):

- `WRITING_MAX_NEW_TOKENS` (default 1200): longest allowed rewrite. Lower it
  to save GPU budget per request.
- `WRITING_MODEL_ID` (default `Qwen/Qwen2.5-3B-Instruct`): swap in a DPO-tuned
  model later once it is on the Hub.
