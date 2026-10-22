"""Build a tiny random Qwen2-architecture model for local code-path testing.

This is NOT Qwen2.5-3B. It has random weights and a small tokenizer trained on
the project's training split, so its output is nonsense. It only proves that
tokenization, the chat template, streaming generation, truncation detection
and the Gradio wiring run without errors on the installed library versions.

Usage: python tests/make_tiny_model.py <output_dir>
"""
import csv
import sys
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

out = Path(sys.argv[1])
root = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader((root / "data/splits/train.csv").open(encoding="utf-8-sig")))
corpus = [r["AI Version"] for r in rows] + [r["Ethan Rewrite"] for r in rows]

tok = Tokenizer(models.BPE(unk_token="<unk>"))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=1200, special_tokens=[
    "<unk>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"])
tok.train_from_iterator(corpus, trainer)

CHAT_TEMPLATE = (
    "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + "
    "message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")
fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>",
                               eos_token="<|im_end|>", pad_token="<|endoftext|>",
                               additional_special_tokens=["<|im_start|>", "<|im_end|>"])
fast.chat_template = CHAT_TEMPLATE
fast.save_pretrained(out)

config = Qwen2Config(vocab_size=fast.vocab_size + 10, hidden_size=64, intermediate_size=128,
                     num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                     max_position_embeddings=8192, eos_token_id=fast.eos_token_id,
                     pad_token_id=fast.pad_token_id, tie_word_embeddings=True)
model = Qwen2ForCausalLM(config)
model.generation_config.eos_token_id = fast.eos_token_id
# Random weights often emit the end token first; force a few tokens so tests see text.
model.generation_config.min_new_tokens = 8
# Keep whitespace-only tokens out so the random output contains visible text.
blank = [i for i in range(fast.vocab_size)
         if i != fast.eos_token_id and not fast.decode([i], skip_special_tokens=True).strip()]
model.generation_config.suppress_tokens = blank
model.save_pretrained(out)
print("tiny model saved to", out, "vocab", fast.vocab_size)
