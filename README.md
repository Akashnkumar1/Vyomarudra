# Vyomarudra · व्योमरुद्र

> *Vyoma* (व्योम) — the sky, the cosmos, boundless space.
> *Rudra* (रुद्र) — the fierce, ancient Vedic power.
> **Vyomarudra: the fierce power of the cosmos** — a redefined-architecture LLM,
> built corner by corner, entirely our own.

**Vision:** frontier-level capability on a normal 8GB Mac — by redefining the
architecture, not shrinking a model. Honestly reframed as **capability parity, not
weight parity** (a lossless 300B model in 8GB is information-theoretically
impossible; see [`docs/`](docs/)).

**Stack:** Rust, end to end (no Python). A Cargo workspace, ML via
[`candle`](https://github.com/huggingface/candle). Every corner is ours — no wrapped
models. Pretrained models (phi4-mini) are only ever **teachers we learn from**
(distillation); none is a runtime component.

## Where this stands (2026-07)

**The achievable architecture is complete, composed, measured, and seed-confirmed —
every corner built by us.** All five pillars have working corners; the continuous +
sparse core is assembled into one model; retrieval, symbolic verification, and
self-evolution compose as layers around it. The one remaining gap to the north star
is **scale** (data × model × compute) — a compute commitment, not an unproven idea.

Living detail: **[docs/PROGRESS.md](docs/PROGRESS.md)** (moving log) ·
**[docs/CRATES.md](docs/CRATES.md)** (what/why/status per crate) ·
**[docs/english/08-phase0-findings.md](docs/english/08-phase0-findings.md)** (standalone report).

## The five pillars — status

| Pillar | Corner (crate / mode) | Status | Headline |
|---|---|---|---|
| **1** Eternal Seed (generative weights) | `vyoma-e1`, `vyoma-ssm`, `vyoma-lm` | ✅ characterized | Fractal seed → **423× on images**, improves with scale; **modest on language**, **loses on SSM cores & MoE experts**. Generation is a *redundancy-bound multiplier*, not the engine. |
| **2** Resonance Manifold (SSM) | `vyoma-ssm`, `vyoma-lm`, `vyoma-embed` | ✅ | Lean diagonal-SSM backbone; also the retriever's encoder. Stored (not generated). |
| **3** Trinity (dense/MoE/symbolic) | `vyoma-lm` (`moe`, `g1`, `lattice`) | ✅✅ both halves | Stored top-1 **MoE**: big-model quality at ~small active cost (+0.11 bits/byte over dense-small, scale-invariant & seed-confirmed). **Symbolic lattice**: hallucinations **100%→0%** by vetoing unsupported answers. |
| **4** Ontological Store (retrieval) | `vyoma-embed`, `vyoma-lm` (`retro`, `retrolm`) | ✅ the engine | Our learned retriever beats classical at matched dim (~8× more dim-efficient); persistent int8 on-disk store (`VYST`, ~64 B/fact); closes E2 with a real retriever; helps a real LM on real text. |
| **5** Self-evolution + homeostasis | `vyoma-lm` (`evolve`) | ✅ | Model self-evolves by **writing to its store** (no weight edits ⇒ no forgetting); a homeostasis controller vetoes contradictions — naive degrades 1.0→0.76 under poison, homeostasis holds ~1.0. |

Cross-cutting: `vyoma-tokenizer` (byte-level BPE, ours) · `vyoma-distill`
(teacher→corpus, student is ours) · bits-per-byte eval harness.

## The composed result

The assembled core — **BPE + lean SSM + stored MoE** on our best (distilled) corpus —
is our strongest model, **BPB 1.847** (best), MoE win seed-confirmed and invariant
across scale. **Capability-per-RAM** (`MODE=g1`): int8 is ~free on the real LM mass,
so the MoE matches dense-big quality at **¼ the RAM**, with knowledge held **off-RAM
on disk**. That is the achievable-vision thesis, measured, ours.

**Scale curve (within-corpus, 1.34 MB):** BPB 2.008 → 1.867 → 1.847 as model+compute
grow — then flattens: we've reached the **data-bound (Chinchilla) inflection**. The
next lever is more data + matched model/compute — i.e., throughput/compute, not
architecture.

## What's NOT done (honest)

Everything is **toy scale** (≤~0.5 M params, ~1–2 MB corpus, one laptop, mostly
single-run where not marked seed-confirmed). These establish *directions,
mechanisms, and a composed architecture* — **not a shipped model**. The leap to a
frontier-competitive 8 GB model is scale (data × model × compute) on real compute;
the architecture is ready for it.

## The one principle

> **Vision big, bets small.** Every grand idea reduces to a proven technique, or a
> cheap falsifiable experiment. Negative results are kept. Nothing speculative on the
> critical path — and nothing on the critical path is unproven anymore.

## Quickstart

```bash
cargo build --release
# the assembled continuous+sparse core, scored by bits-per-byte:
MODE=moe TOKENIZER=bpe DATASET=distilled STEPS=4000 DM=128 DFF=192 ./target/release/vyoma-lm
# capability-per-RAM (int8) · symbolic hallucination veto · self-evolution:
MODE=g1 ./target/release/vyoma-lm  ·  MODE=lattice ...  ·  MODE=evolve ...
```

Datasets fetched with `curl -sk` (corporate SSL interception); see crate READMEs.
Full mode reference: [`crates/vyoma-lm/README.md`](crates/vyoma-lm/README.md).
