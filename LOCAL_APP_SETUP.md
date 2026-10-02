# Run your writing app on Windows

Copy these app files into the main Personal-Writing-Style-AI folder in VS Code, next to README.md. Keep your notebook and existing requirements.txt. The app uses data/splits/train.csv; the included copy is the six-row file you supplied.

## First-time setup

Use Python 3.11 (64-bit). In VS Code, open Terminal > New Terminal in the repo folder. Run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
.\.venv\Scripts\python.exe check_gpu.py
.\.venv\Scripts\python.exe app.py
```

If the first command says Python 3.11 is missing, install 64-bit Python 3.11 from python.org, then restart the VS Code terminal. If `py` is missing but `python --version` shows Python 3.11, use `python -m venv .venv` for the first command.

Use the CUDA 12.8 PyTorch build above for the RTX 5070 Ti, rather than copying LAUNCH's CUDA 12.6 setup. You need a compatible NVIDIA driver; pip provides the CUDA runtime used by this PyTorch build. You do not need a separate CUDA Toolkit for this app.

The GPU check runs an actual small calculation and should show your NVIDIA GPU and `GPU calculation passed`. If it fails, stop and share the error. Do not continue installing random packages.

The app opens http://127.0.0.1:7860 in your browser. If the browser does not open automatically, open that address yourself. The first rewrite downloads about 6 GB of model weights and then loads the model onto your GPU. Keep the terminal open. Later rewrites reuse the loaded model. Closing the app releases it.

## Open it next time

Double-click **start_app.bat**, or run:

```powershell
.\.venv\Scripts\python.exe app.py
```

Press Ctrl+C in the terminal to stop. This is a local browser application, not a packaged desktop executable. It does not need Jupyter or LAUNCH. The first model download needs internet. Your input is processed by local Qwen, not a paid API.

## What changed from the notebook

- app.py serves the local UI; rewrite_engine.py loads Qwen once and generates rewrites.
- web/ contains the interface, styling, and browser actions.
- Basic prompt and With my examples use the notebook UI's instructions, with an added instruction to treat input instructions as text.
- With my examples uses up to three valid pairs from train.csv and excludes a normalized matching input. It is fixed few-shot prompting, not RAG or training.
- The output limit is 1,600 tokens instead of 700 to allow longer drafts. A warning appears if it reaches the limit. Record this setting when comparing with old notebook results.
- Requests over 1,000 words, 20,000 characters, or 7,000 prompt tokens are rejected rather than silently truncated.
- Save result downloads JSON with the generated rewrite, edited rewrite, input, example IDs, model, generation settings, and timing. Timing excludes model loading and tokenization.
- Editing or saving an output does not train the model. DPO is still future work.
- Only the training CSV is loaded. Keep validation and test examples out of it.

## Checks before your demo

1. Run check_gpu.py and confirm the real GPU calculation passes.
2. Launch the app and rewrite a new short post with Basic prompt.
3. Try With my examples on the same post. Check facts and wording.
4. Edit the output, copy it, and save the JSON. Confirm both original output and edited output are present.
5. Close and reopen using start_app.bat.

The server and input checks were tested with a stub engine. Real Qwen inference and Windows launch must be verified on your computer. This app listens only on 127.0.0.1; its built-in server is intended for your local prototype.

Before committing, ensure your repo's .gitignore includes `.venv/` and `__pycache__/`. Do not commit your Python environment or model cache.

Official references:
- https://pytorch.org/blog/pytorch-2-7/ (Blackwell and CUDA 12.8 support)
- https://pytorch.org/get-started/previous-versions/ (installation commands)
- https://huggingface.co/Qwen/Qwen2.5-3B-Instruct (model)
