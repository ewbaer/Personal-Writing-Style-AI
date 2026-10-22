"""Qwen inference for the Hugging Face ZeroGPU Space.

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
from .rewrite_engine import MODEL_ID, build_messages, load_examples

MAX_WORDS = 1000
MAX_CHARS = 20000
MAX_PROMPT_TOKENS = 7000
# Lower than the local 1,600 so one request cannot hold the GPU too long.
MAX_NEW_TOKENS = int(os.environ.get("WRITING_MAX_NEW_TOKENS", "1200"))


def validate_text(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Paste some text first.")
    text = text.strip()
    if len(text) > MAX_CHARS or len(text.split()) > MAX_WORDS:
        raise ValueError("Use 1,000 words or fewer (and under 20,000 characters).")
    return text


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
    def __init__(self):
        self.model = self.tokenizer = None
        self.precision = None
        self.load_error = None
        self.model_id = os.environ.get("WRITING_MODEL_ID", MODEL_ID)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        """Call once at import time, before the Gradio app starts."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=dtype)
            model.to(self.device).eval()
            self.precision = str(dtype).replace("torch.", "")
            self.tokenizer, self.model = tokenizer, model
        except Exception as error:  # Keep the detector usable even if Qwen fails.
            self.load_error = f"{type(error).__name__}: {error}"

    def prepare(self, text, method):
        """Validation and prompt building. Runs on CPU before any GPU time."""
        text = validate_text(text)
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
        from transformers import TextIteratorStreamer
        tokens = tokens.to(self.model.device)
        prompt_tokens = tokens["input_ids"].shape[1]
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True,
                                        skip_special_tokens=True)
        holder = {}

        def run():
            try:
                with torch.inference_mode():
                    holder["output"] = self.model.generate(
                        **tokens, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                        streamer=streamer, pad_token_id=self.tokenizer.eos_token_id)
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
        for piece in streamer:
            if first_token is None and piece:
                first_token = time.perf_counter() - started
            pieces.append(piece)
            yield "".join(pieces)
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
                   precision=self.precision, model=self.model_id)
