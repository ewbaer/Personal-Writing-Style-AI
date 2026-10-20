# Personal Writing Style AI

A local Python/Flask app that rewrites text using Qwen2.5-3B-Instruct and optional examples of my writing. The goal is to match the user's style while keeping the original meaning.

**To run or restore the project, read [docs/RESTORE.md](docs/RESTORE.md).**

## Current state

The rewriting model uses prompting. It has not been DPO-trained. General rewrite uses instructions; My examples adds up to three writing pairs from the training split and excludes a matching input. This is fixed example prompting, not retrieval or model training.

The app has a two-column editor, editable output, word counts, a copy button, light/dark mode, JSON export and a local AI detector. Highlight sentences is a toggle inside the revision editor. Editing clears stale highlights until the text is checked again.

Qwen runs on an NVIDIA GPU. Large GPUs use full weights; GPUs under 10 GB use 4-bit weights. The model downloads once to the normal Hugging Face cache and reloads into GPU memory when the app restarts.

## Writing data

There are ten LinkedIn pairs with `ID`, `Type`, `AI Version` and `Ethan Rewrite` columns. The saved split has six training pairs, two validation pairs and two test pairs. The app reads only the training split. This is enough to test the workflow, not enough to draw strong conclusions about personal style training.

The notebook and early experiments remain in `notebooks` and `results`. Student rewrites and public data should be reviewed and kept separate from my personal style examples.

## Local detector

The detector uses TF-IDF word and two-word phrase features with logistic regression. The preparation script reads the public `AI_Human.csv` dataset, filters length, removes exact normalized duplicates and samples up to 10,000 human and 10,000 AI passages. Related examples need shared group IDs where those are available. Hash IDs cannot prevent related authors or prompts from crossing splits when the source lacks those IDs.

Training uses five group-separated folds: three for training, one for validation and one for testing. The fitted vocabulary comes only from training. Reports include accuracy, false positives, AI recall, ROC AUC, Brier score and a split manifest.

The original October 15 report is archived in `results/detector_previous_run`. It reported about 98.9% test accuracy on that dataset. It does not establish accuracy on LinkedIn posts, rewritten text or individual sentences.

**The trained `models/ai_detector.joblib` file and large detector dataset are not included.** Restore the model from the laptop or use the included scripts to retrain. The app can rewrite without a detector.

## Scores and highlights

The main status shows `Text flagged as AI-like`: words in flagged sentences divided by all words. The whole-passage classifier score is separate under Score details. Neither number proves authorship.

Sentences of eight or more words are checked individually using the same passage classifier, with a 0.5 cutoff. Shorter fragments are unassessed. Sentence accuracy and the cutoff have not been validated on sentence-labeled data. No rule forces a set number of highlights, and a full-text score can differ from the share of text flagged.

The detector uses its learned vocabulary and classifier. It does not reproduce ZeroGPT, measure perplexity/burstiness, or call an external detection API.

## Files

| File or folder | Purpose |
|---|---|
| `app.py` | Local Flask app and rewrite endpoint |
| `src/rewrite_engine.py` | Qwen loading, prompts and generation |
| `src/ai_detector.py` | Detector loading, sentence results and API routes |
| `web/` | Editor, theme, controls and inline highlights |
| `setup_windows.bat` | Environment setup and GPU check |
| `start_app.bat` | App launcher |
| `scripts/train_detector.bat` | Prepare data and train a missing detector |
| `scripts/prepare_detector_data.py` | Public CSV filtering and sampling |
| `scripts/train_detector.py` | Fit, test and save the detector |
| `data/` | Personal writing pairs and splits |
| `notebooks/` | Original data checks and experiments |
| `results/` | Saved experiments and archived training report |
| `tests/` | Server, training pipeline and browser-logic checks |

## Saved results

The JSON export keeps the input, generated output, edited output, prompting method, example IDs, model, precision, token counts and generation time. It also keeps the original detector results and the latest edited-output check when it matches the current editor text. Editing does not retrain either model.

Generation time excludes model loading and tokenization. Compare the same precision, prompts and generation settings across experiments. Inputs over 1,000 words or 20,000 characters are rejected; prompts above 7,000 tokens are rejected instead of silently truncated.

## Next steps

Evaluate the detector on separate LinkedIn writing, review more rewrite pairs, prepare DPO preferences and compare trained rewrites with the prompting baseline on held-out inputs. A second model or training method is still to be finalized.

## Folder layout

The root keeps the Flask entry point, README, requirements and Windows launchers.
`src` contains the rewriting and detection code. `scripts` contains data preparation,
training and the GPU check. `web` contains the interface. `docs` contains setup and
verification notes. `requirements` contains detector-only and development dependencies.
The optional Node test configuration lives in `tests`.

Start the app from the root with `start_app.bat`. To train a missing detector, run
`scripts\train_detector.bat` in PowerShell. This cleanup does not retrain either model.
