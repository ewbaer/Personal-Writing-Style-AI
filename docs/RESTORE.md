# Restore the project on your PC

This folder contains the latest code available from your GitHub repo plus the detector and interface changes from this conversation. You do not need to install each earlier ZIP.

## 1. Put the files in your repo

Close any running copy of the app. Make a backup of your existing project folder first.

Open this extracted folder. Copy **everything inside it** into your existing `Personal-Writing-Style-AI` repo folder, next to `app.py`. Allow Windows to merge folders and replace matching files. Do not place the whole extracted folder inside the repo as another nested project. Keep your existing environment, Git history and any trained model files. This ZIP does not contain those.

The notebook, ten writing pairs, six/two/two splits and early result files are included from the current GitHub repo. If you made newer changes on your laptop that were never pushed or shared here, copy those separately before replacing an older version.

## 2. Set up and open the app

Double-click `setup_windows.bat`. It uses Python 3.11, installs CUDA PyTorch and the app packages, then runs a real GPU calculation. On the RTX 5070 Ti, the app uses full weights. On GPUs with less than 10 GB VRAM, it automatically uses 4-bit weights to leave room for generation. This is a memory setting, not DPO training.

If Python 3.11 is missing, run this in PowerShell, reopen the terminal, then run setup again:

```powershell
winget install -e --id Python.Python.3.11
```

After setup finishes, double-click `start_app.bat`. Open http://127.0.0.1:7860 if the browser does not open. Keep the terminal open. Qwen downloads on first use if it is not already cached, then loads into GPU memory. Restarting the app reloads those cached weights.

## 3. Restore your trained detector

**Your original trained detector was never uploaded here or committed to GitHub, so it is not in this ZIP.** Its October 15 metrics report is included as a record, but that report cannot recreate its weights.

On your laptop, find this file inside the project:

```text
models/ai_detector.joblib
```

Copy it into the same `models` folder on your PC, then restart the app. You do not need to retrain if you can copy that file. Only load your own trusted detector files.

If you cannot copy it, put the public dataset file `AI_Human.csv` in `data`, then double-click `scripts/train_detector.bat`. It prepares up to 10,000 examples per class and trains the detector. If you already have `data/detector_labeled.csv`, it uses that directly. The original large dataset is not included in this ZIP. The training launcher stops if a model file already exists so it does not replace it by accident.

Rewriting works while the detector is missing. The signals panel will say `Detector not trained yet.`

## 4. Quick check on your own PC

1. Rewrite a short draft using General rewrite, then My examples.
2. Check that the text appears on the right and both word counts update.
3. With a restored/trained detector, check the edited output. It needs at least 50 words.
4. Click Highlight sentences. The highlights appear in the revision editor. Click again to turn them off. Editing clears the highlights and asks you to check again.
5. Try light/dark mode, Copy and the JSON download under Settings.

The main result is the **percentage of words in sentences flagged as AI-like**. It is not the amount proven to have been written by AI. Whole-text classifier confidence appears separately under Score details. Sentence scoring uses the passage model and remains experimental.

## Verification included

See `TEST_REPORT.md` for the checks actually run. This environment could test the server, real detector training pipeline and browser logic. It could not run Windows batch files or real CUDA/Qwen generation. Run setup and the quick check above to verify those on your PC.
