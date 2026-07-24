# vyoma-distill — distillation data pipeline (learn from a teacher, don't ship it)

Generates a clean, varied training corpus using a **teacher** model (phi4-mini via
the local Ollama API) and writes it to disk. **Our** model (`vyoma-lm
DATASET=distilled`) then trains on it.

**Principle (firm):** a pretrained model is a *teacher we learn from* — never a
component of Vyomarudra. The teacher produces training data; the corpus and the model
that learns from it are entirely ours. This is the roadmap's Phase-1 mechanism
("distill from an open teacher" — the student is ours).

## Run

```bash
# needs `ollama serve` with phi4-mini
cargo build --release -p vyoma-distill
./target/release/vyoma-distill                 # writes vyoma-lm/data_cache/distilled.txt
DATASET=distilled MODE=diag ./target/release/vyoma-lm   # our model learns from it
```

## Status

Pipeline verified end to end: teacher → corpus → our model trains (loss falls,
wider FFN helps). Current corpus is a small proof (18 prompts); the road to
capability is scaling it (thousands of prompts) + more training.
