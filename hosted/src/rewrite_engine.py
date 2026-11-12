"""Qwen inference extracted from Ethan's notebook. No training happens here."""
import csv
import os
import threading
import time
from pathlib import Path

from .paths import ROOT
from .config import SETTINGS
from .validation import validate_text
from .prompts import INSTRUCTION, build_messages, load_examples, normalize
MODEL_ID = SETTINGS.model_id


class RewriteEngine:
    def __init__(self):
        self.model = self.tokenizer = None
        # One generation at a time keeps multiple clicks from exhausting VRAM.
        self.lock = threading.Lock()

    def load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU detected. Run scripts/check_gpu.py using the app's .venv.")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        # Keep inference on the GPU. Smaller GPUs use 4-bit weights rather
        # than silently spilling layers into slow CPU/RAM offload.
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        precision = os.environ.get("WRITING_PRECISION", "auto").lower()
        if precision not in {"auto", "full", "4bit"}:
            raise RuntimeError("WRITING_PRECISION must be auto, full, or 4bit.")
        use_4bit = precision == "4bit" or (precision == "auto" and total_bytes < 10 * 1024**3)
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs = dict(dtype=compute_dtype, device_map={"": 0})
        if use_4bit:
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as error:
                raise RuntimeError("This GPU needs 4-bit loading. Run setup_windows.bat to install bitsandbytes.") from error
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype)
        elif free_bytes < 7 * 1024**3:
            raise RuntimeError("Not enough free VRAM for full weights. Close other GPU programs, or set WRITING_PRECISION=4bit.")
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs).eval()
        self.precision = "4bit" if use_4bit else str(compute_dtype).replace("torch.", "")
        self.tokenizer, self.model = tokenizer, model

    def rewrite(self, text, method):
        text = validate_text(text)
        examples = load_examples() if method == "examples" else []
        messages, example_ids = build_messages(text, method, examples)
        if not self.lock.acquire(blocking=False):
            raise ValueError("A rewrite is already running. Please wait for it to finish.")
        try:
            import torch
            self.load()
            tokens = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt"
            ).to(self.model.device)
            prompt_tokens = tokens["input_ids"].shape[1]
            # Fail clearly rather than silently cutting off examples or the draft.
            if prompt_tokens > SETTINGS.max_prompt_tokens:
                raise ValueError("This draft and its examples are too long. Shorten the draft or use Basic prompt.")
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                output = self.model.generate(
                    **tokens, max_new_tokens=SETTINGS.max_new_tokens, do_sample=False,
                    temperature=1.0, top_p=1.0, top_k=50,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            response = output[0, prompt_tokens:]
            rewrite = self.tokenizer.decode(response, skip_special_tokens=True).strip()
            if not rewrite:
                raise RuntimeError("The model returned an empty rewrite. Try again.")
            eos = self.model.generation_config.eos_token_id
            eos_ids = eos if isinstance(eos, list) else [eos]
            truncated = len(response) >= SETTINGS.max_new_tokens and response[-1].item() not in eos_ids
            return dict(original=text, rewrite=rewrite, method=method, model=MODEL_ID,
                        example_ids=example_ids, seconds=round(seconds, 2),
                        prompt_tokens=prompt_tokens, output_tokens=len(response),
                        truncated=truncated, gpu=torch.cuda.get_device_name(0),
                        max_new_tokens=SETTINGS.max_new_tokens, do_sample=False, precision=self.precision)
        finally:
            self.lock.release()
