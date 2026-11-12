"""Rewrite inference for the Hugging Face ZeroGPU Space.

Differences from the local rewrite_engine.py:
- The model is loaded once at startup and placed on CUDA at module level,
  which is what the ZeroGPU documentation asks for. There is no lazy loading
  and no VRAM check, because ZeroGPU hands the function a 48 GB GPU slice.
- Full bf16 weights are used. No 4-bit loading, so bitsandbytes is not needed.
- Generation streams tokens so the page can show text early and the app can
  measure time to first token.
- Multiple visitors are handled by the Gradio queue, not by a lock that
  rejects the second request.

Prompt building, validation limits and the example rules are imported from
rewrite_engine.py so the hosted app behaves like the local one.
"""
import os
import threading
import time
from pathlib import Path

import torch

from .paths import ROOT
from .config import SETTINGS, model_profiles
from .prompts import build_messages, load_examples
from .model_backend import load_base, attach_adapter, training_messages, REWRITE_PREFIX
from .validation import validate_text
MAX_WORDS = SETTINGS.max_words
MAX_CHARS = SETTINGS.max_chars
MAX_PROMPT_TOKENS = SETTINGS.max_prompt_tokens
MAX_NEW_TOKENS = SETTINGS.max_new_tokens


def examples_path():
    """Local training split first. Otherwise a private Hugging Face dataset.

    Set the Space secret WRITING_EXAMPLES_REPO (for example ewbaer/writing-examples)
    and HF_TOKEN if you do not want train.csv inside the public Space files.
    """
    local = ROOT / "data" / "splits" / "train.csv"
    if local.exists():
        return local
    repo = os.environ.get("WRITING_EXAMPLES_REPO")
    if not repo:
        return local
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, "train.csv", repo_type="dataset",
                                token=os.environ.get("HF_TOKEN")))


def gpu_seconds(text, method):
    """Estimated GPU time for the ZeroGPU scheduler. Inputs match generate()."""
    words = len(str(text or "").split())
    # Rough budget: fixed overhead plus time per input word. Shorter requests
    # get better queue priority, so this is kept as low as is reasonable.
    return int(min(120, max(30, 15 + 0.06 * words)))


class HostedEngine:
    def __init__(self, profile=None):
        self.profile = profile or next(model_profiles())
        self.model = self.tokenizer = None
        self.precision = None
        self.load_error = None
        self.model_id = self.profile.model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        """Call once at import time, before the Gradio app starts."""
        try:
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            tokenizer, model = load_base(self.model_id, self.profile.base_revision, dtype=dtype,
                                         attn_implementation="sdpa")
            if self.profile.adapter:
                # ZeroGPU has no real CUDA device during startup. Load the
                # adapter on CPU before staging the complete model on CUDA.
                model = attach_adapter(model, self.model_id, self.profile.adapter,
                                       self.profile.adapter_revision, self.profile.base_revision,
                                       token=os.environ.get("HF_TOKEN"))
            model.to(self.device).eval()
            self.precision = str(dtype).replace("torch.", "")
            self.tokenizer, self.model = tokenizer, model
        except Exception as error:  # Keep the detector usable if the model fails.
            self.load_error = f"{type(error).__name__}: {error}"
            import logging
            logging.getLogger(__name__).exception("Rewrite model failed to load: %s", self.profile.key)

    def prepare(self, text, method):
        """Validation and prompt building. Runs on CPU before any GPU time."""
        text = validate_text(text)
        if self.profile.experiment_prompt:
            if method != "basic":
                raise ValueError("This experiment uses General rewrite only.")
            messages, example_ids = training_messages(REWRITE_PREFIX + text), []
        else:
            examples = load_examples(examples_path()) if method == "examples" else []
            messages, example_ids = build_messages(text, method, examples)
        if self.model is None:
            raise RuntimeError("The rewrite model is not loaded. " + (self.load_error or ""))
        tokens = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")
        prompt_tokens = int(tokens["input_ids"].shape[1])
        if prompt_tokens > MAX_PROMPT_TOKENS:
            raise ValueError("This draft and its examples are too long. Shorten the draft or use General rewrite.")
        return text, tokens, prompt_tokens, example_ids

    def stream(self, tokens):
        """Yield growing text, then a final dict with timing and token counts.

        Must run inside the @spaces.GPU function so real CUDA is available.
        """
        from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
        tokens = tokens.to(self.model.device)
        prompt_tokens = tokens["input_ids"].shape[1]
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True,
                                        skip_special_tokens=True)
        holder = {}
        stop = threading.Event()

        class StopRequested(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return stop.is_set()


        def run():
            try:
                with torch.inference_mode():
                    holder["output"] = self.model.generate(
                        **tokens, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                        streamer=streamer, stopping_criteria=StoppingCriteriaList([StopRequested()]), pad_token_id=(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id))
            except Exception as error:
                holder["error"] = error
                streamer.end()

        if self.device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        first_token = None
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        pieces = []
        try:
            for piece in streamer:
                if first_token is None and piece:
                    first_token = time.perf_counter() - started
                pieces.append(piece)
                yield "".join(pieces)
        finally:
            # Closing/cancelling the generator asks transformers to stop at the next token.
            # Join before releasing the GPU slot, including on errors and disconnects.
            stop.set()
            thread.join()
        if self.device == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        if "error" in holder:
            raise holder["error"]
        response = holder["output"][0, prompt_tokens:]
        rewrite = self.tokenizer.decode(response, skip_special_tokens=True).strip()
        if not rewrite:
            raise RuntimeError("The model returned an empty rewrite. Try again.")
        eos = self.model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        truncated = len(response) >= MAX_NEW_TOKENS and response[-1].item() not in eos_ids
        gpu = torch.cuda.get_device_name(0) if self.device == "cuda" else "CPU"
        yield dict(rewrite=rewrite, seconds=round(seconds, 2),
                   first_token_seconds=round(first_token, 2) if first_token else None,
                   output_tokens=int(len(response)), truncated=truncated, gpu=gpu,
                   max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                   precision=self.precision, model=self.model_id, model_key=self.profile.key,
                   base_revision=self.profile.base_revision, adapter_loaded=bool(self.profile.adapter),
                   adapter_revision=self.profile.adapter_revision)
