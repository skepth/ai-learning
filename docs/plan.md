# Plan

Ten phases, from a scalar autograd warmup to a minimal inference server, plus two side tracks. Each phase has a goal and a concrete "done when" so progress is measurable rather than vibes-based.

## Datasets

- **Tiny Shakespeare** — character-level first training run (small, fast, easy to sanity-check).
- **TinyStories** — main dataset for the BPE-tokenized transformer.
- **Synthetic data** — XOR (Phase 0), modular arithmetic (grokking side quest), and context/question/answer examples derived from TinyStories (Phase 7).

## Crates

`minigrad` (Phase 0) · `tokenizer` · `tensor` · `autograd` · `nn` · `optim` · `train` (binary) · `infer` · `client` · `trainlog` + `viz` (side track)

## Order

```
0 → 1 → 2 → 3 → [viz tool → grokking] → 4 → 5 → 6 → 7 → 8 → 9
```

Phases 2–3 are the long pole: expected to take roughly as long as everything after them combined.

---

## Phase 0 — Scalar autograd warmup ✓

A micrograd-style engine where every number is a graph node.

- Value nodes with parents and the op that produced them
- Reverse topological sort and reverse-mode backward
- Neuron, Layer and MLP built on top; train a small MLP on XOR

**Done when:** an MLP learns XOR, and the reason gradients accumulate (`+=`) is clear.
Write-up: [phase-0-learnings.md](phase-0-learnings.md)

## Phase 1 — Byte-level BPE tokenizer

- 256 base byte tokens, so any Unicode text is representable
- Regex pre-splitting (GPT-4-style pattern; needs lookahead → `fancy-regex` or a hand-rolled splitter)
- Count adjacent pairs within chunks; merge loop until the target vocabulary size (~4k–8k)
- Special tokens, including reserved chat-format tokens for Phase 7

**Done when:** `decode(encode(x)) == x` for any Unicode input, and compression is compared against tiktoken.

## Phase 2 — Tensor library

- `Vec<f32>` storage with shape and strides
- Views (transpose, slice, reshape without copying), broadcasting
- Matmul: naive → cache-blocked → optional parallel (`rayon`) / BLAS behind a feature flag
- Elementwise ops, reductions, softmax, gather

**Done when:** results match NumPy to ~1e-5.

## Phase 3 — Tensor autograd

- One graph node per tensor operation, with a backward rule per op
- Gradients of broadcast operations summed back to the input's shape
- Finite-difference gradient checking for every op
- Fused softmax + cross-entropy

**Done when:** every op passes gradcheck, and an MLP converges on MNIST or a makemore-style character model.

## Side track — Training visualization tool

A small, reusable way to *see* what training is doing, not just read loss numbers.

- `trainlog` crate: trainers append metrics as JSON lines to `runs/<run-name>/metrics.jsonl`
- `viz` binary: a localhost web server with interactive charts, live updates, and multiple runs overlaid (e.g. seed sweeps)
- Logging is decoupled from viewing, so a viewer problem can never break a training run
- Grows over time: loss and predictions first; train/val and learning rate in Phase 5; per-layer gradient norms, activation statistics, dead-unit counts and attention maps in Phase 6

## Side quest — Reproducing grokking

Reproduce "Grokking modular arithmetic" (Gromov, 2023): a two-layer MLP trained with MSE loss and plain gradient descent on modular addition, which first memorizes and then — much later — suddenly generalizes.

- Dataset: all pairs `(a, b)` for a prime `p`, labeled `(a + b) mod p`, randomly split into train and test; the training fraction is a knob to sweep
- Track: train vs test accuracy, the Fourier spectrum of the learned weights, and the weight norm over time

**Done when:** training accuracy reaches 100% early and test accuracy jumps much later, and the weights' Fourier spectrum collapses to a few sharp frequencies around the jump.

## Phase 4 — Layers and optimizers

- Embedding, Linear, RMSNorm / LayerNorm, causal self-attention, GELU MLP
- Residual blocks and a weight-tied output head
- AdamW (decoupled weight decay, bias correction), warmup + cosine learning-rate schedule, gradient clipping
- Principled weight initialization

## Phase 5 — Training a small transformer

- Character-level Tiny Shakespeare first, then BPE TinyStories
- Around 4 layers, 4 heads, model width 128–256, context 256, 1–5M parameters
- Overfit a single batch first as a sanity check
- Train/validation split; checkpoint weights *and* optimizer state

## Phase 6 — Diagnostics and experiments

- Train/val loss, perplexity, per-layer gradient norms, update-to-weight ratio (~1e-3), activation statistics, attention entropy and attention maps
- Experiments: learning-rate sweep, no warmup, removing residuals or norms, vocabulary size, depth vs width

## Phase 7 — Sampling and tiny instruction tuning

- Greedy, temperature, top-k and top-p sampling
- A chat template using the reserved special tokens
- Synthetic context/question/answer data from TinyStories, with the loss masked to assistant tokens — needed so retrieval in Phase 8 actually helps

## Phase 8 — Retrieval-augmented chat client

- Embeddings from a hand-written word2vec/skip-gram model and/or mean-pooled hidden states (compared)
- Brute-force cosine similarity (normalize, then dot product), then HNSW
- Prompt assembly within a ~256–512 token budget

**Done when:** a question is answered correctly with retrieval and incorrectly without it.

## Phase 9 — Mini vLLM

- Naive generation as a baseline
- KV cache, with measured speedup and memory use
- Static batching → continuous batching
- Paged KV cache (block tables)
- Prefix caching for a shared system prompt