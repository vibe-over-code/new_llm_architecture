# PyTorch/CUDA baseline

This is a small independent implementation of the project's core idea. It is
intended as a readable base for experiments, not as a drop-in replacement for
the distributed JAX trainer.

It contains:

- a decoder-only Transformer with causal sliding-window attention and RoPE;
- normal language-model pretraining (`--mode pretrain`);
- differentiable test-time adaptation and an outer meta-loss (`--mode meta`);
- `adapt()`, which makes an adapted copy and leaves the base model unchanged;
- an optional 1-D `.npy` token stream (`--data`).

From this directory:

```bash
pip install -r requirements-torch.txt
python -m torch_ttt.train --device cuda --mode pretrain --steps 100
python -m torch_ttt.train --device cuda --mode meta --steps 100
```

For a quick CPU smoke test, use a smaller model by editing `ModelConfig` or
run with `--device cpu`; CUDA is recommended for actual experiments.

The main extension points are `ModelConfig`, `SlidingWindowAttention`, the
`parameter_prefixes` argument in `adapt()`/`meta_loss()`, and the data loop in
`train.py`.
