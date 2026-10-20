# Local setup

Start with [docs/RESTORE.md](RESTORE.md). It covers restoring the repo, Python setup, the GPU check, launching the app and restoring or retraining your detector.

## Manual setup instead of the setup launcher

From the repository folder in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/check_gpu.py
.\.venv\Scripts\python.exe app.py
```

The app listens at http://127.0.0.1:7860. Stop it with Ctrl+C.

## Memory settings

The default loads Qwen on the GPU, using 4-bit weights when total VRAM is under 10 GB. It does not silently offload to the CPU. To force 4-bit mode for one session:

```powershell
$env:WRITING_PRECISION = "4bit"
.\.venv\Scripts\python.exe app.py
```

Set it to `auto` to return to automatic selection, or `full` for full weights. Record the precision setting when comparing results. Generation can still run out of memory with long prompts or other GPU programs open.

## Detector training commands

```powershell
.\.venv\Scripts\python.exe scripts/prepare_detector_data.py --input data\AI_Human.csv --output data\detector_labeled.csv
.\.venv\Scripts\python.exe scripts/train_detector.py --csv data\detector_labeled.csv
```

The preparation step refuses to replace an existing CSV. The Python training command replaces `models/ai_detector.joblib` and current training reports, so back them up first. The Windows training launcher checks for an existing model before running.

## Optional development tests

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements/dev.txt
.\.venv\Scripts\python.exe -m pytest -q
npm --prefix tests install
npm --prefix tests test
```

Node is only needed for the optional browser-logic tests, not the app itself. Test fixtures use synthetic data and mocked Qwen calls. They do not establish real detection accuracy or GPU speed.
