# ai-learning

Building a small language-model stack from scratch in Rust, to learn how LLMs actually work — from tokenizing text to serving a model with a KV cache.

The path: a scalar autograd engine → a BPE tokenizer → a tensor library with its own autograd → a small transformer trained on TinyStories → sampling and a tiny instruction-tuning pass → a retrieval-augmented chat client → a minimal vLLM-style inference server.

## Ground rules

- **Everything ML-related is hand-written:** tokenizer, tensor library, autograd, layers, optimizers, training loop, inference.
- **No ML or transformer crates.** General plumbing crates are fine: `rand`, `clap`, `serde`, `rayon`, `regex` / `fancy-regex`.
- **One optional exception:** after writing matmul by hand, a BLAS backend (Apple Accelerate) may be added behind a feature flag for comparison.
- **Learn by measuring.** Each phase ends with tests, experiments and a short write-up in [`docs/`](docs/).

## Progress

- [x] **Phase 0** — Scalar autograd engine + MLP trained on XOR → [learnings](docs/phase-0-learnings.md)
- [ ] **Phase 1** — Byte-level BPE tokenizer
- [ ] **Phase 2** — Tensor library
- [ ] **Phase 3** — Tensor-level autograd
- [ ] **Side tracks** — Training visualization tool · grokking reproduction
- [ ] **Phase 4** — Neural-network layers and optimizers
- [ ] **Phase 5** — Training a small transformer
- [ ] **Phase 6** — Training diagnostics and experiments
- [ ] **Phase 7** — Sampling and tiny instruction tuning
- [ ] **Phase 8** — Retrieval-augmented chat client
- [ ] **Phase 9** — Mini vLLM: KV cache, batching, paged attention

The full plan, with goals and completion criteria for each phase, is in [`docs/plan.md`](docs/plan.md).

## Repository layout

```
ai-learning/
├── docs/                 Plan and per-phase learnings
├── libs/
│   └── minigrad/         Phase 0: scalar autograd, Neuron / Layer / MLP
│       └── examples/     Training examples (single neuron, XOR MLP)
└── examples/             Small standalone warmups
```

More crates (`tokenizer`, `tensor`, `autograd`, `nn`, `optim`, `train`, `infer`, `client`) are added as their phases begin.

## Running

```bash
# Run all tests
cargo test

# Train a 2 → 4 → 4 → 1 MLP on XOR
cargo run --release -p minigrad --example mlp_train -- -s 1000 -l 0.01 -r 10
```

`mlp_train` flags: `-s` steps, `-l` learning rate, `-r` random seed, `-p` print interval.