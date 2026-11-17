# miniinfer

## Goal

Build a small LLM inference engine from scratch in PyTorch and Triton, measuring
each optimization against a naive HuggingFace `generate()` baseline and checking
correctness against the HuggingFace reference. Model: `Qwen/Qwen2.5-1.5B-Instruct`.

## Roadmap

- [ ] Baseline (naive `generate()`)
- [ ] KV cache
- [ ] Continuous batching
- [ ] Paged KV cache
- [ ] Triton attention kernel
- [ ] Speculative decoding
- [ ] Quantization

## Results

Hardware: _TBD_ (Colab T4). Model: Qwen/Qwen2.5-1.5B-Instruct.

| Optimization | Tokens/sec | Speedup vs baseline | Correctness test |
|--------------|-----------:|--------------------:|------------------|
| Baseline     | _TBD_      | 1.0x                | n/a              |
| KV cache     | _TBD_      | _TBD_               | _TBD_            |
| Continuous batching | _TBD_ | _TBD_              | _TBD_            |
| Paged KV cache | _TBD_    | _TBD_               | _TBD_            |
| Triton attention | _TBD_  | _TBD_               | _TBD_            |
| Speculative decoding | _TBD_ | _TBD_             | _TBD_            |
| Quantization | _TBD_      | _TBD_               | _TBD_            |

## Running the baseline

```bash
pip install -e .
python benchmarks/baseline.py
```
