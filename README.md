# Vyomarudra · व्योमरुद्र

> *Vyoma* (व्योम) — the sky, the cosmos, boundless space.
> *Rudra* (रुद्र) — the fierce, ancient Vedic power.
> **Vyomarudra: the fierce power of the cosmos** — a redefined-architecture LLM,
> built corner by corner, entirely our own.

**Vision:** frontier-level capability on a normal 8GB Mac — by redefining the
architecture, not shrinking a model. Honestly reframed as **capability parity,
not weight parity** (a lossless 300B model in 8GB is information-theoretically
impossible; see [`docs/`](docs/)).

**Stack:** Rust, end to end (no Python). A Cargo workspace that grows from
Phase-0 experiments toward the full system — and, eventually, a Rust AI OS.
ML via [`candle`](https://github.com/huggingface/candle).

## Layout

```
docs/                 the vision (five pillars, roadmap, honest risk register)
  english/            source of truth
  hinglish/           mirror
crates/               every corner built by us — no wrapped models
  vyoma-e1/           generative weights ("Eternal Seed") + quantization
  vyoma-ssm/          diagonal SSM backbone (Pillar 2)
  vyoma-lm/           integrated char/LM: SSM + generated FFN, kNN-LM, diag/kv modes
  vyoma-retrieval/    retrieval-quality-at-scale (Ontological Store, Pillar 4)
  vyoma-rag/          E3: externalized knowledge validated on a real model (teacher = lesson only)
  vyoma-distill/      distillation pipeline — OUR model learns from a teacher (teacher never shipped)
  vyoma-tokenizer/    byte-level BPE, from scratch
  (next) scaled distillation corpus · full assembled model · eval harness
```

## Bottom line so far

Phase-0 is complete. Full standalone report: **[docs/08 — Phase-0 Findings](docs/english/08-phase0-findings.md)**.
In one line: weight-generation is real but redundancy-bound (great on image MLPs,
modest on language, loses on SSM cores); **externalized knowledge (retrieval) is
the decisive, scalable win**; the achievable architecture is a retrieval-centric
hybrid, not a magic seed.

## The one principle

> Vision big, bets small. Every grand idea must reduce to a proven technique, or
> become a cheap, falsifiable Phase-0 experiment. Nothing speculative on the
> critical path.

## Status

| Experiment | Pillar | Status | Headline |
|---|---|---|---|
| [E1 — generative weights](crates/vyoma-e1/) | 1 (Eternal Seed) | ✅ + scaling law | Fractal seed: **212× on MNIST** (90% of dense from a 962-param seed). **Scaling law:** fixed seed, bigger target ⇒ better — **423× at 90%** for a 512-wide target. Generation improves with scale. Fashion-MNIST: ~76× @ ≤10% loss (data-dependent). |
| [vyoma-ssm — seed → sequence model](crates/vyoma-ssm/) | 1→2 bridge | ✅ decisive negative | **SSM weights resist generation** (pure & hybrid) — a small stored SSM beats a fractal-generated one; hybrid helps but still loses. Principle: generation pays on **redundant FFN mass**, not lean efficient cores. |
| [vyoma-lm — integrated model](crates/vyoma-lm/) | 1+2 (Phase 1) | ⚠️ modest, scale-growing | Stored SSM + **generated FFN**, char-LM on real text. At dm=128 the generated FFN ties a small FFN; **at dm=256 it beats it by +2–4 pts** (edge grows with scale & compression) but stays ~11 pts below dense. Generation = a modest supporting player on language, not the engine. |
| [vyoma-lm — E2 externalized knowledge](crates/vyoma-lm/) | 4 (Ontological Store) | ✅ decisive positive | `MODE=kv`: memorization **collapses** as facts grow (98%→12%), retrieval stays **flat at 97%**. Knowledge belongs on disk, not in weights — capability decoupled from model size. The core of the achievable hybrid. |
| [vyoma-retrieval — E2b scaling](crates/vyoma-retrieval/) | 4 (Ontological Store) | ✅ positive, **13s** | Retrieval accuracy vs library size: with adequate embedding dim (1024) stays **flat ~97%** across 8.7k passages; too small (128) collapses. The big library holds — governed by embedding dim. No training. |
| [vyoma-e1 — quantization](crates/vyoma-e1/) | 1 + quant | ✅ composes at 8-bit | `QUANT=1`: 8-bit-quantizing the seed is ~free (127×→**508× byte-compression**, 85.6%→85.5%, 1.6 KB seed). 4-bit breaks it (→57%) — the more you generate-compress, the higher-information each seed param, so less quant headroom. |
| [vyoma-rag — E3 on a real model](crates/vyoma-rag/) | 4 (Ontological Store) | ✅✅ **real-scale win** | **phi4-mini (3.8B) on the laptop**: 0/20 on novel facts alone → **20/20 with retrieval** (+100 pts). A real model gains capability it lacks in weights, from an external store. The vision's engine, end-to-end, real scale. |
| E2 — externalized knowledge (RETRO) | 4 | ⬜ next | — |
| E3 — field dynamics (Mamba + coupling) | 2 | ⬜ | — |
| E4 — sparse awakening (router) | 3 | ⬜ | — |
| E5 — stability tooling (Lyapunov) | 5 | ⬜ | — |

**Gate G0** (roadmap): E1 must reach ≥100× at ≤10% quality loss, *or* a credible
hybrid path. Current: **data-dependent** — MNIST ~200× @ ≤10% loss (cleared), but
harder **Fashion-MNIST only ~76×** @ ≤10%. Achievable compression tracks data
redundancy. Cleared on MNIST; the real test is sequence/SSM targets + language.

## Quickstart (E1)

```bash
cargo build --release -p vyoma-e1
EPOCHS=15 SEEDS=2 ./target/release/vyoma-e1   # needs MNIST in crates/vyoma-e1/data_cache
```

See [`crates/vyoma-e1/README.md`](crates/vyoma-e1/README.md) for data setup and full results.
