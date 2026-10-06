# Personal Writing Style AI

A local web app that rewrites AI-generated text using examples of a person’s writing. The goal is to match their wording, tone, and sentence structure while keeping the original meaning.

## Project Question

Does training a model on someone’s writing produce better results than giving an existing model instructions and writing examples?

## Current State

The project has a working Python/Flask prototype. It started in Jupyter on the LAUNCH cluster and now runs as a local web app.

The app uses **Qwen2.5-3B-Instruct** to rewrite text with two options:

- **General rewrite:** Uses basic rewriting instructions.
- **With my examples:** Includes personal writing samples in the prompt.

The current model has **not been fine-tuned**. These options use prompting, and DPO training is the next main step.

## Working Features

- Two-column editor for the original text and rewritten output.
- Light and dark mode.
- Word counts for both versions.
- Editable output and a copy button.
- JSON export for saving results.
- Local model inference using an NVIDIA GPU.

The app has run successfully on an RTX 5070 Ti. Laptop setup is also underway for an RTX 4050 with 6 GB of GPU memory, which needs a lower-memory model configuration.

## Data

The personal dataset currently contains 10 LinkedIn posts, each with an AI version and my own rewrite.

| Column | Contents |
|---|---|
| ID | Unique example number |
| Type | Type of writing |
| AI Version | Original AI-generated text |
| Ethan Rewrite | My rewritten version |

The notebook checks for missing and repeated entries, counts words, and creates training, validation, and test splits. The current split contains six training examples, two validation examples, and two test examples.

This small dataset is useful for testing the workflow, but it is not enough to draw strong conclusions about model performance.

A Google Form has been shared with my professor to collect more student rewrites. Each student chooses one AI-generated LinkedIn post and rewrites it in their own words. These responses will represent different writers, so they will be kept separate from my personal writing examples.

## Experiments So Far

I compared a basic rewrite prompt with a prompt containing three writing examples and saved the results.

The example-based output showed some differences in wording and structure, but the early results do not yet show a clear improvement in matching my style.

I am also reviewing public datasets, including De-GPT-DPO, for an initial training experiment. Any public data will need to be checked for quality and adapted to the rewriting task before use.

## Training Plan

The planned training method is **Direct Preference Optimization (DPO)**.

Each training example will contain:

- A rewriting request and its original text.
- A preferred rewrite.
- A less-preferred rewrite.

Preferences will consider writing style, readability, and whether the rewrite preserves the original facts. A human-written response will not automatically be treated as better without reviewing it.

The trained model will be compared with the existing prompting methods using the same held-out inputs. A second model or training method has not been finalized.

## Evaluation

The main questions are:

- Does the rewrite sound like the intended writer?
- Does it preserve the original meaning and facts?
- Is it clear and readable?
- How long does generation take?
- How much GPU memory and context does it use?

AI detector scores are being explored as an additional measurement. Automatic detector integration is not yet complete, and a detector score will not be treated as proof that text is human-written or as a measure of personal style.

## Running the App

After setting up the project’s Python environment and dependencies, run this from the repository’s main folder in PowerShell:

```powershell
.\start_app.bat
```

Then open:

```text
http://127.0.0.1:7860
```

Keep the terminal open while using the app.

The model downloads on its first use and is cached for later sessions. Restarting the app still requires loading the model into memory.

## Main Files

| File or folder | Purpose |
|---|---|
| `app.py` | Flask server and API routes |
| `rewrite_engine.py` | Model loading, prompts, and text generation |
| `web/` | Interface, styling, and browser code |
| `check_gpu.py` | GPU availability and calculation check |
| `start_app.bat` | Windows launcher |
| `requirements-local.txt` | Local app dependencies |
| `notebooks/` | Data checks and early experiments |
| `data/` | Writing examples and dataset splits |
| `results/` | Saved experiment outputs |

## Next Steps

- Review student responses when they arrive.
- Prepare and check preference pairs for DPO.
- Run a small training experiment.
- Compare trained and untrained outputs on held-out examples.
- Add an optional detector score to the interface.
- Test empty inputs, long text, failed requests, and saved-result handling.
