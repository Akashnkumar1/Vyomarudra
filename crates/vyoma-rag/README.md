# vyoma-rag — E3: externalized knowledge on a REAL model

Validates the vision's engine at real scale: a real 3.8B model (phi4-mini via local
Ollama) + pure-Rust lexical retrieval over a store of **novel/fictional facts** the
model cannot have memorized. Baseline (ask directly) vs RAG (retrieve the fact into
context, then ask).

> This is a **lesson we own** — proof that externalized knowledge works at scale —
> not a shippable system. phi4-mini is a reference here, never part of Vyomarudra.

## Run

```bash
# needs `ollama serve` with phi4-mini
cargo build --release -p vyoma-rag
./target/release/vyoma-rag
```

## Result

| | phi4-mini alone | phi4-mini + retrieval |
|---|---|---|
| Novel-fact accuracy | **0/20** | **20/20** |

Retrieval hit-rate 100%, lift **+100 pts**. A real model, on the laptop, gains
capability it does NOT have in its weights, from an external store — "small brain +
big library," end to end. (Facts are deliberately novel to isolate retrieval;
validates the knowledge pillar, not reasoning-parity.)
