# vyoma-retrieval — E2b: retrieval quality vs. library size

Answers the question E2's oracle skipped: **does retrieval stay accurate as the
knowledge store grows?** Pure Rust, no training — passages of tiny-shakespeare are
embedded as hashed char-trigram bags (dim `dk`) and retrieved by cosine; a query is
a fragment of a source passage, correct if the nearest passage is the source.

## Run

```bash
cargo build --release -p vyoma-retrieval
./target/release/vyoma-retrieval          # ~13 s
```

## Result (retrieval accuracy vs #passages)

| dk | N=100 | N=1000 | N=5000 | N=8714 |
|---|---|---|---|---|
| 128 | 0.80 | 0.45 | 0.34 | 0.31 |
| 256 | 0.93 | 0.77 | 0.66 | 0.60 |
| 1024 | 1.00 | 0.99 | 0.98 | **0.97** |

Retrieval reliability is governed by **embedding dimension**: too small collapses
(collisions); adequate (1024) stays flat ~97% across the whole corpus. So the "big
library" holds — and this is a *weak* baseline (trigram bags); learned embeddings
do better at lower dim. With E2, the retrieval engine is validated end to end.
