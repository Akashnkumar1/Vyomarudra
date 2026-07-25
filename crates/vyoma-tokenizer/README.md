# vyoma-tokenizer — byte-level BPE (ours, from scratch)

A byte-level Byte-Pair-Encoding tokenizer with **zero external dependencies** — no
library, no teacher. 256-byte base vocabulary (any input encodes; no UNK) +
iterative most-frequent-pair merges to a target size.

## Run

```bash
cargo build --release -p vyoma-tokenizer
DATASET=text VOCAB=512 ./target/release/vyoma-tokenizer     # tinyshakespeare
DATASET=distilled VOCAB=1024 ./target/release/vyoma-tokenizer
DATASET=fineweb VOCAB=4096 SAMPLE_MB=30 ./target/release/vyoma-tokenizer  # real web text
```
Env: `DATASET` (text|distilled|fineweb), `VOCAB` (≥257), `SAMPLE_MB` (default 20 —
merges are trained on this many MB of the corpus; training cost scales with
*unique words seen*, not total size, but the cap keeps it bounded on a large
corpus like FineWeb). Writes `bpe_merges.txt` into `vyoma-lm/data_cache/` for our
model to reuse (`vyoma-lm TOKENIZER=bpe DATASET=fineweb`) — the learned merges are
applied to the FULL corpus at that point, even though only a sample trained them.

## Result (tiny-shakespeare, lossless round-trip ✓)

| vocab | compression |
|---|---|
| 512 | 2.02 bytes/token |
| 1024 | 2.62 |
| 2048 | 3.22 |

Learned meaningful subwords — `" shall"`, `" thou"`, `" lord"`, `" your"`. Trains in
~2 s. Wired into `vyoma-lm`, it lowered bits-per-byte **25%** vs char-level at
matched compute — a real, measured win (see `docs/PROGRESS.md`).
