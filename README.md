# Personal Writing Style AI

## About

For this project, I’m building an AI that can rewrite generated text to sound more like the person using it. The user will give it examples of their writing so it can pick up on things like the words they use, sentence length, and how they explain ideas.

The goal is to match their writing style without changing what the original text means.

## Main Question

Does training a model on someone's writing help it match their style better than just giving a regular model instructions and examples?

## What I’m Testing

I’m going to compare a few ways of rewriting the same text:

* **Basic prompting:** Tell the model how I want the text to sound.
* **Prompting with examples:** Give it a few writing samples to follow.
* **RAG:** Have it find useful examples from saved writing and use those when rewriting.
* **LoRA or QLoRA:** Train a model adapter using AI text paired with preferred rewrites.

I also want to see whether giving the model more examples or more context improves the results.

## Dataset

Right now, I have nine LinkedIn writing examples. I plan to add more as I work on the project, with most texts being around 200–250 words.

For each pair, I’ll save the original AI text and the edited version. I’ll also record the topic, writing type, and whether the rewrite was done by a person or AI.

If other people contribute examples, I’ll keep track of which samples belong to each writer without using their names. Their writing could help with general rewriting, but it would not count as my personal style.

I’ll set aside some examples for testing so the model does not get to use them during training or as examples in its prompts.

## How It Will Work

1. Read the user's writing samples.
2. Look for patterns in how they write.
3. Rewrite the input using the chosen method.
4. Check that the meaning stayed the same.
5. Save the results so the methods can be compared.

I also plan to save what the system learns about the user's style so it does not have to start over each time. Some steps will pass information through JSON files, which I’ll check for errors.

## Testing

The main things I’ll look at are:

* Does the rewrite sound like the user?
* Does it keep the original meaning and facts?
* Is it clear and easy to read?
* How long does it take?
* How much context and computing power does it need?

Each method will get the same test inputs to make the comparison fair. I may also record AI detector scores, but those scores alone will not show whether the writing is good or sounds like the user.

Other tests will cover empty inputs, unusual text, broken JSON, saved style preferences, and inputs that try to make the model ignore its instructions.

## Project Folders

| Folder       | What it contains                                             |
| ------------ | ------------------------------------------------------------ |
| `notebooks/` | Jupyter notebooks for preparing data and running experiments |
| `src/`       | Python code used throughout the project                      |
| `data/`      | Dataset notes and samples that can be shared                 |
| `results/`   | Test scores and rewritten outputs                            |
| `docs/`      | Project proposal, test plan, and other notes                 |

## Setup

I’m using Python and Jupyter Notebook on the LAUNCH cluster with an NVIDIA A30 GPU.

I’ll add the required packages and steps for running the project as I build it. Private writing samples, API tokens, downloaded models, and training checkpoints will stay out of the repo.

## Current Progress

* Created the GitHub repo and project folders.
* Collected nine LinkedIn writing examples.
* Working on organizing the data before testing the first model.

Training, testing, and the UI still need to be built.

## Final Demo

The user will provide writing samples and some text they want rewritten. The demo will show the original text alongside the different rewrites so we can compare how well each method follows their style.
