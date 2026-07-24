# vyoma-tokenizer — byte-level BPE (ours, from scratch)

A byte-level Byte-Pair-Encoding tokenizer with **zero external dependencies** — no
library, no teacher. 256-byte base vocabulary (any input encodes; no UNK) +
iterative most-frequent-pair merges to a target size.

## Run

```bash
cargo build --release -p vyoma-tokenizer
DATASET=text VOCAB=512 ./target/release/vyoma-tokenizer     # tinyshakespeare
DATASET=distilled VOCAB=1024 ./target/release/vyoma-tokenizer
```
Env: `DATASET` (text|distilled), `VOCAB` (≥257). Writes `bpe_merges.txt` into
`vyoma-lm/data_cache/` for our model to reuse (`vyoma-lm TOKENIZER=bpe`).

## Result (tiny-shakespeare, lossless round-trip ✓)

| vocab | compression |
|---|---|
| 512 | 2.02 bytes/token |
| 1024 | 2.62 |
| 2048 | 3.22 |

Learned meaningful subwords — `" shall"`, `" thou"`, `" lord"`, `" your"`. Trains in
~2 s. Wired into `vyoma-lm`, it lowered bits-per-byte **25%** vs char-level at
matched compute — a real, measured win (see `docs/PROGRESS.md`).
