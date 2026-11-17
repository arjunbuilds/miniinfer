"""Naive baseline: HuggingFace generate() tokens/sec.

Runs on a Colab T4 (fp16, ~3 GB of weights). Usage:
    python benchmarks/baseline.py [--max-new-tokens 128] [--runs 3]
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "Explain how a transformer language model generates text, step by step."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # T4 has no bf16 support; use fp16 on GPU.
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
    model.eval()

    messages = [{"role": "user", "content": PROMPT}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.max_new_tokens,  # fixed length for stable timing
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    def sync() -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    with torch.inference_mode():
        model.generate(**inputs, **{**gen_kwargs, "max_new_tokens": 8, "min_new_tokens": 8})  # warmup
        rates = []
        for i in range(args.runs):
            sync()
            start = time.perf_counter()
            out = model.generate(**inputs, **gen_kwargs)
            sync()
            elapsed = time.perf_counter() - start
            new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
            rates.append(new_tokens / elapsed)
            print(f"run {i + 1}: {new_tokens} tokens in {elapsed:.2f}s = {rates[-1]:.1f} tokens/sec")

    print(f"\ndevice={device} dtype={dtype} model={MODEL_ID}")
    print(f"baseline: {sum(rates) / len(rates):.1f} tokens/sec (mean of {args.runs} runs)")


if __name__ == "__main__":
    main()
