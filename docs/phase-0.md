# Phase 0 — Scalar autograd and a first neural network

Phase 0 is a warmup: a scalar automatic-differentiation engine (in the spirit of micrograd), written from scratch in Rust, and used to train a small multi-layer perceptron on XOR. Everything here gets rebuilt at the tensor level in Phases 2–3; the point of this phase is to understand backpropagation one number at a time.

## What was built

| Piece | Where | What it does |
|---|---|---|
| `Value` | `libs/minigrad/src/grad.rs` | A node in the computation graph: data, grad, parents, and the op that produced it |
| Ops | `grad.rs` | `add`, `sub`, `mul`, `pow`, `tanh`, `relu`, each with a local derivative used in backward |
| Backward | `grad.rs` | Reverse topological sort, then the chain rule from the output back to every leaf |
| `Neuron` | `libs/minigrad/src/nn.rs` | Weights + bias as `Value`s; computes `activation(bias + Σ wᵢxᵢ)` |
| `Layer` | `nn.rs` | A group of neurons that all receive the same inputs; outputs one value per neuron |
| `MLP` | `libs/minigrad/src/mlp.rs` | Layers chained together via a builder: `MLP::new(2).layer(4, Tanh).layer(4, Tanh).output(1, Linear).build(&mut rng)` |
| Training | `libs/minigrad/examples/mlp_train.rs` | Trains a 2 → 4 → 4 → 1 MLP on XOR, with flags for steps, learning rate, and seed |

```bash
cargo run --release -p minigrad --example mlp_train -- -s 1000 -l 0.01 -r 10
```

## Key concepts

### Backpropagation is two local rules plus bookkeeping

- **Add** passes its gradient to both parents unchanged (`∂(x + y)/∂x = 1`).
- **Mul** gives each parent `grad × the other parent's value`.
- **tanh** scales the incoming gradient by `1 − out²`, where `out` is the tanh *output*, not its input.
- **ReLU** passes the gradient through where its input was positive and blocks it (0) elsewhere.

### `×` versus `+=`

- `×` is how a gradient flows along **one edge**: child grad × local derivative.
- `+=` is what happens when a node feeds **several** children: the contributions from each path are summed.

This showed up concretely in testing a layer: inputs shared by two neurons receive `w₀·k₀ + w₄·k₁`, one term per neuron. Assigning with `=` instead of `+=` would silently drop all but the last path.

### Parameters versus inputs

A neuron *owns* its weights and bias (they persist and get updated). Inputs are *passed in* on each forward call. Mixing the two — for example, storing input × weight products in the neuron — breaks learning, because the update then nudges intermediate results instead of the actual parameters.

### Layers output lists, not sums

A layer returns one value per neuron, and that list becomes the next layer's input. The connection between layers is "output list → input list", which is why each layer's input count equals the previous layer's size.

Parameter count per layer: `n_out × (n_in + 1)`. For 2 → 4 → 4 → 1: 12 + 20 + 5 = **37**.

### Why hidden layers and nonlinearities

- XOR is not linearly separable: no single line splits `{(0,1), (1,0)}` from `{(0,0), (1,1)}`.
- Hidden layers learn intermediate features (roughly "OR" and "NAND") that make the problem separable for the output neuron.
- Without a nonlinearity, stacked linear layers collapse into one: `W₂(W₁x + b₁) + b₂ = W′x + b′`.
- The output layer is linear so predictions can reach any target value.

### One training step

```
for each step:
    total_loss = 0
    for each example:
        prediction = mlp.forward(input)
        total_loss += (target − prediction)²
    zero_grad
    total_loss.backward()     ← one backward over all examples
    for each parameter: data −= lr × grad
```

The same four examples are reused every step, but the network is different each time (the weights moved), so each step produces new gradients. Many small steps are needed because the gradient is only accurate locally; a large learning rate overshoots.

## Experiment: activation functions across 20 seeds

The seed fixes the random number generator's starting point, which fixes the initial weights. Same seed and same layer sizes means identical starting weights across configurations, so only the activation differs.

Setup: 2 → 4 → 4 → 1, 1000 steps, learning rate 0.01, success = all four predictions within 0.1 of target.

| Hidden activations | Learned XOR | Failed seeds |
|---|---|---|
| tanh → tanh | **20 / 20** | — |
| ReLU → ReLU | **13 / 20** | 1, 2, 3, 5, 7, 10, 19 |
| tanh → ReLU | **16 / 20** | 3, 7, 12, 19 |

### What the failures were: dead ReLUs

ReLU outputs 0 for negative inputs, **with zero gradient**. A neuron stuck in that region gets no learning signal and cannot recover.

With seed 10, ReLU → ReLU converged to a total loss of exactly **0.6667**. Three of the four inputs produced the identical prediction 0.3339 — the network was blind to them and output the best constant for targets {0, 1, 0}, which is their mean, 1/3:

```
(1/3)² + (2/3)² + (1/3)² = 2/3
```

Contributing factors in this setup:
- Biases start at 0 and inputs are 0/1, so for input `(0,0)` every first-layer pre-activation is exactly 0, and ReLU's slope there is 0 — that input sends no gradient into the first layer at the start.
- 4-neuron layers leave little redundancy when a few units die.

A tanh first layer helps because tanh's slope at 0 is 1, so every input contributes gradient from the start. It rescued seeds 1, 2, 5 and 10, but not 3, 7 and 19 — likely second-layer initializations that leave its ReLUs dead regardless of the first layer.

### Takeaways

- **One run is an anecdote.** The same configuration produced both the best curve (ReLU, seed 50) and the worst (ReLU, seed 10). Judge configurations by their success rate across seeds.
- ReLU trains fast when its units stay alive (its gradient doesn't shrink like tanh's), but small networks feel its dead-unit risk the most.
- Standard mitigations — not tried here yet: wider layers, small positive bias initialization, Leaky ReLU, and principled weight initialization (He/Kaiming).
- 16 vs 13 out of 20 is suggestive, not conclusive; more seeds would be needed to separate the two ReLU configurations reliably.

## Reading loss curves

| Shape | Usual meaning | Seen here |
|---|---|---|
| Plateau, then a sudden drop | Started near a poor region (e.g. predicting ~0.5 everywhere) and escaped slowly | tanh → tanh, seed 10 |
| Flatline above zero | Stuck — e.g. dead units | ReLU → ReLU, seed 10 |
| Zigzag | Learning rate too high | early single-neuron experiments |
| Slow crawl | Learning rate too low | early single-neuron experiments |
| Train ↓, validation ↑ | Overfitting | not applicable to XOR (the 4 examples *are* the whole function) |

The loss shows *that* something went wrong; the network's internals (activation values, dead-unit counts, per-layer gradients) show *why*.

## Mistakes worth remembering

- **Wrong tanh derivative on paper:** using `1 − z²` (input) instead of `1 − out²` (output).
- **Adding instead of multiplying** in the mul backward rule on paper (the code was right — the linear-neuron test would have caught it).
- **Zeroing gradients per example** inside the training loop: only the last example's gradients survived to the update. Fix: sum all examples into one loss, then zero, backward and update once.
- **Every MLP layer built with the network's input size** instead of the previous layer's size.
- **Exact float equality** in tests: works by accident for some operation orders, then fails on a 1e-16 difference. Use a tolerance.

## Rust notes

- `Value` is a newtype over `Rc<RefCell<Node>>`. Cloning a `Value` clones the `Rc` — cheap, and it points at the *same* node, which is what makes `parameters()` work: updating a returned clone updates the network.
- Pass the RNG as `&mut StdRng` and let the caller own it. Cloning a seeded RNG for each neuron would give every neuron identical weights, and identical neurons get identical gradients forever.
- Functions that only read inputs take `&[Value]`, not `Vec<Value>`, so a layer can hand the same inputs to every neuron without cloning.
- Keep argument order consistent across constructors (`inputs, outputs` everywhere): two adjacent `usize` parameters compile fine when swapped.
- Return one flat `Vec<Value>` from `parameters()` at every level. The optimizer treats every parameter the same, and flat lists compose (neuron → layer → MLP) without the type changing.

## Why this doesn't scale

Each scalar is its own heap-allocated graph node. A single 512×512 matrix multiply would create hundreds of millions of nodes. Tensors fix this by making one node represent a whole array and one backward rule cover the whole operation — the subject of Phases 2–3.

## Next

Phase 1: a byte-level BPE tokenizer.