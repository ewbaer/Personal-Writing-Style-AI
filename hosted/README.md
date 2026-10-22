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

# Writing desk (hosted version)

Hosted copy of the Personal Writing Style AI project for class use. It rewrites a
pasted draft in Ethan's writing style with Qwen2.5-3B-Instruct, then scores the
original and the revision with a locally trained TF-IDF + logistic regression
detector. The revision is editable, can be re-checked without running Qwen again,
and flagged sentences can be highlighted.

Source project: https://github.com/ewbaer/Personal-Writing-Style-AI

## How it runs here

- Gradio app on Hugging Face ZeroGPU. The model is loaded in bf16 at startup and
  generation runs inside a `@spaces.GPU` function.
- Requests wait in a queue and run one at a time. Gradio shows your place in line.
- Each visitor has a daily ZeroGPU time budget set by Hugging Face. Signing in to
  Hugging Face gives more time than anonymous use.
- The detector runs on CPU and does not use GPU time.

## Limits

- Up to 1,000 words or 20,000 characters per request.
- Up to 1,200 new tokens per rewrite. Longer output is marked as cut off.
- The detector needs at least 50 words. Its score is an experimental classifier
  output, not proof of authorship.
- The rewrite model has not been DPO-trained. "My examples" adds up to three
  writing pairs to the prompt. This is fixed example prompting, not retrieval.

## Optional: keep the writing examples out of the public files

By default the app reads `data/splits/train.csv` from this repository. To keep the
examples private, delete that file from the Space, upload it as `train.csv` to a
private dataset repository, and add two Space secrets: `WRITING_EXAMPLES_REPO`
(for example `ewbaer/writing-examples`) and `HF_TOKEN` (a read token).
