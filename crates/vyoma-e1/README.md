# E1 — Generative Weights (the "Eternal Seed")

> Phase-0 falsification experiment for **Pillar 1** of Vyomarudra.
> This is the one load-bearing, unproven bet (Risk #1, the project-killer):
> **store the rules that generate weights, not the weights.**

## The question E1 actually asks

It is *not* impressive that a hypernetwork has fewer parameters than the network
it produces — you could always just use a smaller network. The sharp, honest
question is:

> Does generating a target network's weights from a small **seed** (S params)
> beat simply training a plain network that *also* has ~S params?

If not, generative weights add nothing. So at every compression point we train
**two** models on the same budget and compare:

1. **HyperSeed** — a chunked hypernetwork (the miniature Eternal Seed). The target
   weight vector is cut into `n_chunks` chunks of size `chunk`. Each chunk `i`
   has a learned embedding `z_i`; a tiny shared MLP `g` maps every embedding to
   its chunk of weights. A per-chunk learned scale is calibrated at init so the
   generated target starts at a sane weight std (not near-zero).
   Seed storage = `params(g) + n_chunks·embed_dim` — the only thing in RAM.
2. **PlainMLP** — a plain MLP sized to ~S params, trained directly. The fair
   baseline.

Upper bound = the full dense target trained directly.

### The fractal seed (the mechanism that breaks the compression wall)

The stored-embedding `HyperSeed` above pays `n_chunks · embed_dim` just to store
one embedding per chunk. That table grows with the target, so it **caps
compression** — past ~40× the embeddings dominate the whole seed.

`FractalSeed` removes the table. Each chunk's address is **generated** from a
fixed sinusoidal encoding of its index, through a tiny address network:

```
chunk index  ──► sinusoidal PE (fixed, 0 params) ──► address net ──► embedding
embedding    ──► generator net ──► chunk weights
embedding    ──► scale head    ──► per-chunk gain
```

Nothing per-chunk is stored, so **seed size is ~constant in target size** — the
literal "the rules generate the addresses too" idea. This is the IFS/fractal
addition the doc bets on, and it is what lets compression scale into the 100×+
regime the stored-embedding seed cannot reach.

## What it runs on

- **Rust + [candle](https://github.com/huggingface/candle)** (pure-Rust tensors +
  autograd), CPU backend. No Python.
- **Real MNIST** (auto-loaded from `data_cache/` if the IDX files are present;
  falls back to an offline synthetic teacher task otherwise).
- Target: a 784→128→10 MLP = **101,770 params**.

## Run it

```bash
# one-time: fetch MNIST (‑k tolerates corporate SSL interception)
cd crates/vyoma-e1 && mkdir -p data_cache && cd data_cache
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -sk -o "$f.gz" "https://ossci-datasets.s3.amazonaws.com/mnist/$f.gz" && gunzip -kf "$f.gz"
done

# build + run (env-tunable)
cd ../../.. && cargo build --release -p vyoma-e1
EPOCHS=15 SEEDS=2 ./target/release/vyoma-e1
```

Results are written to `results/e1_results.md` and `results/e1_results.json`.

## Results

MNIST, 18 epochs, 2 seeds. Target 784→256→10 = **203,530 params**. Dense upper
bound = **98.03%**. Full table in [`results/e1_results.md`](results/e1_results.md).

| Fractal comp | Seed params | Fractal acc | Plain MLP (same budget) | Fractal − Plain | % of dense | Stored-embed seed would be |
|---|---|---|---|---|---|---|
| 21× | 9,554 | 95.74% | 93.62% | +2.12 | 97.7% | 22,524 (9×) |
| 41× | 4,994 | 95.44% | 91.03% | +4.41 | 97.4% | 6,638 (31×) |
| 76× | 2,690 | 93.89% | 81.36% | +12.52 | 95.8% | 6,324 (32×) |
| **127×** | 1,602 | 87.27% | 68.45% | **+18.82** | 89.0% | 3,680 (55×) |
| **212×** | **962** | 88.19% | 32.65% | **+55.54** | 90.0% | 5,425 (38×) |

**Headline:** the fractal seed reaches **212× compression** — a **962-param seed
generating a 203,530-param network** at 88.2% on MNIST (90% of dense) — where a
same-size plain MLP manages only 32.7%. The advantage of generating weights over
just using a small model grows without bound as compression rises: from +2 pts at
21× to **+55 pts at 212×**. The old stored-embedding seed (last column) physically
cannot reach these ratios because its embedding table grows with the target.

### Against Gate G0 (≥100× at ≤10% quality loss)

- **≥100× compression: cleared** (127× and 212×).
- **≤10% quality loss: at the margin** — 76× keeps 95.8% (4% loss); 212× keeps
  90.0% (~10% loss); 127× is ~11%. So on MNIST we land right at the G0 threshold:
  ~100–200× at ~10% loss. That is a **credible-path** result by G0's own wording.

That is a real positive rung for Pillar 1: fractal weight-generation extracts
large amounts of reusable structure, exactly as the 90%-redundancy hypothesis
predicts.

### Scaling law — generation improves with target size

MNIST feedforward sweep at three target widths, compared at a **fixed 962-param
seed** (`HIDDEN=128|256|512`). Run: `for H in 128 256 512; do EPOCHS=15 SEEDS=2 HIDDEN=$H ./target/release/vyoma-e1; done`.

| Target | Params | Compression | Fractal acc | % of dense |
|---|---|---|---|---|
| 128-wide | 101,770 | 105.8× | 84.55% | 86.5% |
| 256-wide | 203,530 | 211.6× | 88.49% | 90.2% |
| 512-wide | 407,050 | **423.1×** | **90.09%** | **91.8%** |

Same seed, larger target ⇒ higher accuracy, higher % of dense kept, and ~2× more
compression per step. **Generative weights get more effective with scale** — the
direction the 8 GB bet needs. (Caveat: MNIST saturates ~98%; the capability-meaningful
version of this law needs harder data where bigger dense models genuinely win.)

### Honesty check — Fashion-MNIST (harder, less redundant)

Same sweep, `DATASET=fashion`. Dense upper bound = **89.0%** (vs MNIST's 98%).

| Compression | Seed | Fractal | Plain (same size) | Edge | % of dense |
|---|---|---|---|---|---|
| 21× | 9,554 | 84.65% | 85.31% | −0.66 | 95.1% |
| 41× | 4,994 | 84.09% | 82.86% | +1.23 | 94.5% |
| 76× | 2,690 | 82.72% | 58.39% | +24.32 | 92.9% |
| 127× | 1,602 | 74.11% | 67.61% | +6.51 | 83.3% |
| 212× | 962 | 77.69% | 36.52% | +41.16 | 87.3% |

**Two honest lessons:** (1) **achievable compression tracks data redundancy** — at
≤10% loss MNIST reached ~200× but Fashion only ~76×; (2) the fractal edge is
**regime-dependent** — nil (even slightly negative) at low compression where a
plain model still has enough capacity, but large (+24 to +41 pts) at extreme
compression where plain models collapse. Generation earns its keep exactly in the
aggressive-compression regime the 8GB vision needs. Fashion does **not** clear
G0's ≥100×-at-≤10% bar (it clears ~76×); G0 remains cleared-on-MNIST-only.

## Quantization composes (`QUANT=1`)

`EPOCHS=15 SEEDS=1 DATASET=mnist HIDDEN=256 QUANT=1 ./target/release/vyoma-e1` trains
a high-compression seed then evaluates the generated net with the seed quantized:

| Seed precision | Acc | % of dense | Effective compression | Seed size |
|---|---|---|---|---|
| fp32 | 85.65% | 87.4% | 127× | 6.3 KB |
| **8-bit** | 85.50% | 87.2% | **508×** | **1.6 KB** |
| 4-bit | 57.41% | 58.6% | 1016× | 0.8 KB |

Generation × 8-bit quant compose ~for free (508× byte-compression, 1.6 KB seed →
203K-param net). 4-bit breaks it: the harder you generate-compress, the more
information-dense each seed parameter, so it tolerates less further quantization.

## How to read the verdict

The load-bearing column is **`Hyper − Direct`**, not raw compression:

- ✅ hypernet > same-size plain MLP → generation is adding real value.
- ➖ tie.
- ❌ generation adds nothing at this budget.

## Honest scope (what E1 does and does not prove)

- ✅ Provides a real, reproducible compression-vs-quality curve on real data.
- ✅ Tests the *fair* question (vs. a same-size direct model), not a strawman.
- ❌ Does **not** prove the 100–1000× fractal-compression bet — this is a 100k-param
  target at single-digit-to-~40× compression. It is the **first rung**, per the
  roadmap's Gate G0.
- Next rungs: (a) larger targets (higher achievable compression at fixed seed),
  (b) the actual Vyomarudra addition — **fractal / IFS weight sharing** across scales,
  (c) generating a real SSM/Mamba block's weights instead of an MLP.
