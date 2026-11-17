# miniinfer

A mini LLM inference engine in PyTorch and Triton. Resume/learning project.
Reference model: `Qwen/Qwen2.5-1.5B-Instruct`. Target hardware: Colab T4 (fp16, no bf16).

## Rules

1. **Every optimization needs both:**
   - (a) a correctness test in `tests/` against the HuggingFace reference
     (greedy outputs/logits must match within a stated tolerance), and
   - (b) a benchmark in `benchmarks/` against the naive baseline
     (`benchmarks/baseline.py`), with results added to the README table.
2. **Keep commits small.** One logical change per commit.
3. **No model weights in git.** No `*.safetensors`, `*.pt`, or other checkpoints.
   Weights are downloaded from the Hugging Face Hub at runtime.
4. **Ask before adding dependencies** (including to `pyproject.toml`).

## Layout

- `src/miniinfer/` engine code
- `benchmarks/` runnable benchmarks (must work on a Colab T4)
- `tests/` pytest correctness tests
- `notebooks/` Colab/exploration notebooks
- `docs/` design notes
