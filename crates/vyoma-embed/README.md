# vyoma-embed — learned neural retrieval embeddings (Pillar 4 engine upgrade)

The RETRO-quality replacement for E2b's classical baseline. E2/E2b proved that
retrieval is the decisive engine of the achievable Vyomarudra ("small brain + big
library"): a tiny model that *retrieves* facts stays ~97% accurate while one that
must *memorize* them collapses. But E2b measured that with a deliberately **weak**
embedding — hand-hashed char-trigram bags — which only reach the ~0.97 plateau at a
large dimension (dk=1024) and collapse at dk=128.

This crate builds a **learned** encoder that is entirely ours:

```
bytes → embedding → our diagonal SSM backbone (Pillar 2, reused) → mean-pool → projection → L2-normalize
```

trained **contrastively** (in-batch InfoNCE): a query is a random 48-byte fragment
of a 128-byte passage; its positive is the full passage; every other passage in the
batch is a negative. No teacher, no library — 100% our architecture.

Train and test passages are **disjoint** (80/20), so we measure generalization.

## Two falsifiable claims (both measured head-to-head vs the trigram baseline)

1. **Dimension efficiency** — the learned encoder should reach the trigram plateau
   at a *smaller* output dim `dk`. Smaller `dk` = cheaper store per fact = a bigger
   library fits in the same budget.
2. **Robustness** — under query corruption (random byte substitution), the learned
   encoder should degrade gracefully where a surface-n-gram bag breaks.

## Run

```bash
cargo build --release -p vyoma-embed
# head-to-head vs trigram (sweeps dk), ~5 min at 1500 steps:
STEPS=1500 BATCH=128 DMODEL=128 ./target/release/vyoma-embed
# add QUANT=1 to also report int8/int4 store cost; DK=128 to pin one output dim.
QUANT=1 DK=128 STEPS=1500 BATCH=128 DMODEL=128 ./target/release/vyoma-embed
# LAYERS=N stacks N SSM blocks (depth tested neutral — width is the scale lever).
```

## The persistent Ontological Store (`MODE=store`)

Turns the retriever into a real, reusable Pillar-4 artifact: train the encoder,
encode a corpus, quantize keys to **int8**, and write them to disk in our own
`VYST` format; then reload from disk and query it. Knowledge lives on disk; the
encoder (skills) stays small in RAM. No teacher anywhere in the loop.

```bash
MODE=store DK=128 STEPS=1500 BATCH=128 DMODEL=128 ./target/release/vyoma-embed
# writes data_cache/ontological_store.vyst, reloads it, and runs queries from disk.
```

Env knobs: `STEPS`, `BATCH`, `DMODEL`, `LAYERS`, `DK`, `QUANT`, `MODE`. Needs
`tinyshakespeare.txt` in `../vyoma-lm/data_cache/` (shared with the other crates).

## Used as a library

This crate is **lib + bin**. The reusable retriever/store lives in `lib.rs`
(`Encoder`, `train_encoder`, `embed_all`, `nearest`, `write_store`/`load_store`,
`VYST` format); `main.rs` is just the evaluation harness. `vyoma-lm` depends on the
library for `MODE=retro` (our retriever fetches, our LM answers — no teacher).

## Status / what's left

✅ beats classical at matched dim (~8× more dim-efficient); scales with WIDTH not
depth; keys quantize to int4 free; persistent on-disk store; ~6× faster after
batching the scan. **Left:** a semantic query set (as teacher-*generated training
data*, never teacher-at-inference); attention pooling if we must beat the classical
high-dim ceiling on literal matching.

See [`docs/CRATES.md`](../../docs/CRATES.md) for the full crate map and
[`docs/PROGRESS.md`](../../docs/PROGRESS.md) for results.
