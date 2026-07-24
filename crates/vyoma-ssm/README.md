# vyoma-ssm — can the seed generate a *sequence model*?

> The Pillar 1 → Pillar 2 bridge. E1 showed a fractal seed can generate an MLP's
> weights. But Vyomarudra's backbone is an **SSM** (Pillar 2), not an MLP. This asks
> the harder question: can the same seed generate the weights of a real
> **state-space sequence model**, and does compression still pay?

## The target: a minimal diagonal SSM

A per-channel linear recurrence — the core of S4D / Mamba, minus the selective
gating — chosen because it is **stable by construction** (relevant to Pillar 5's
Lyapunov concern):

```
u_t = gelu(W_enc · x_t + b_enc)              # encode each 28-dim row
h_t = A ⊙ h_{t-1} + B ⊙ u_t,   A = σ(a) ∈ (0,1)   # stable diagonal recurrence
y   = C ⊙ h_T + D ⊙ u_T                      # read out final state
logits = W_cls · gelu(W_mix · y + b_mix) + b_cls
```

`A = σ(a) ∈ (0,1)` guarantees the recurrence cannot blow up — no eigenvalue
outside the unit disk. Task: **row-wise MNIST** — each image is a sequence of 28
rows of 28 pixels, so the model must actually carry information across timesteps.

## The comparison (same honest test as E1)

At each compression point: **fractal-generated SSM** vs. a **same-size plain SSM**
trained directly, against the **dense SSM** upper bound. All three share an
architecture-derived **per-group init-scale prior** (Kaiming-style; O(1) storage)
so the encoder / recurrence / readout each start at a healthy magnitude — without
it the generated SSM starts dead (input-independent) and never learns.

## Run it

```bash
cargo build --release -p vyoma-ssm
EPOCHS=12 SEEDS=2 DMODEL=256 TRAIN_SUBSET=20000 DATASET=mnist ./target/release/vyoma-ssm
```
Datasets are shared with `../vyoma-e1/data_cache/`. Env: `EPOCHS`, `SEEDS`,
`DMODEL` (SSM width), `TRAIN_SUBSET`, `DATASET` (mnist|fashion).

## What this is testing (and the hypothesis)

Feedforward weights (E1) compressed extremely well (up to ~200× on MNIST). SSM
weights may be **harder** to generate: errors in the recurrence parameters
(`a`, `B`, `C`) compound over 28 timesteps, so an approximate seed could hurt a
sequence model far more than a feedforward one. If the fractal-SSM trails a
same-size plain SSM, that is a real, useful finding — it says **generative
compression is modality-dependent, and dynamical systems resist it** — and it
would push the design toward generating only the feedforward/projection weights
while keeping the small, sensitive recurrence parameters stored directly (a
hybrid, exactly the kind Gate G0 allows).

## Results (12 epochs, 2 seeds, 20k subset, row-wise MNIST)

Dense SSM upper bound = **90.2%**.

| Compression | Seed | Fractal-SSM | Plain SSM (same size) | Edge |
|---|---|---|---|---|
| 8.3× | 9,257 | 85.73% | 87.18% | −1.45 |
| 28.6× | 2,685 | 73.66% | 83.99% | −10.34 |
| 48.0× | 1,599 | 55.78% | 81.24% | −25.46 |

**Verdict — the mirror image of E1.** The plain SSM is remarkably robust (81% at
48× on 1,551 params); the fractal-generated SSM collapses (56% at 48×), and the
gap widens with compression. So **generation loses to a small stored SSM** — the
opposite of the MLP result.

**Design consequence.** Don't generate the recurrence. SSMs are already tiny and
parameter-efficient (Pillar 2's point), and dynamical params don't tolerate
approximation.

### Hybrid (`MODE=hybrid`) — generate FFN, store recurrence

Generate only the feedforward matrices (75,264 params); store the recurrence +
biases (1,546) directly. Converged, vs. a same-footprint plain SSM:

| Gen-mass comp | Whole comp | Footprint | Hybrid | Plain SSM (same) | Edge |
|---|---|---|---|---|---|
| 8.1× | 7.1× | 10,803 | 83.45% | 87.83% | −4.38 |
| 28× | 18.2× | 4,231 | 76.07% | 84.08% | −8.01 |
| 47× | 24.4× | 3,145 | 68.10% | 81.61% | −13.51 |

The hybrid degrades **more gracefully** than pure generation (68% vs 56% at ~47×),
so storing the recurrence helped — but it **still loses** to a small stored SSM.

**Honest conclusion.** On this small, efficient sequence model, generative weights
(pure or hybrid) do not beat a same-footprint stored SSM. This is not a refutation
of Pillar 1 — E1 validated generation on redundant feedforward mass (up to 212×).
It bounds the claim: **generation pays on large, redundant weight mass, not on lean
efficient cores.** A small row-MNIST SSM can't stage the frontier regime (huge
redundant FFN/MoE experts a small model cannot match), so the next decisive test
is the E1 scaling law on feedforward targets, not more SSM generation.
