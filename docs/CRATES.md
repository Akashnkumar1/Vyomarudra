# Vyomarudra — Crate Map (what / why / status / what's left)

The one-stop index of every corner we've built. Each crate is **ours** — no wrapped
models. The teacher (phi4-mini) only ever generates *training data we learn from*;
it is never a runtime component. For the moving results log see
[`PROGRESS.md`](PROGRESS.md); for the vision see [`english/`](english/).

The workspace is a Cargo workspace (Rust, no Python; ML via `candle`). Build all:
`cargo build --release`. Datasets are fetched with `curl -sk` (corporate SSL MITM).

---

## The five pillars → which crate touches them

| Pillar | Idea | Crates |
|---|---|---|
| 1 — Eternal Seed | generate weights from a tiny seed | `vyoma-e1`, `vyoma-ssm`, `vyoma-lm` |
| 2 — Resonance Manifold | SSM backbone | `vyoma-ssm`, `vyoma-lm`, `vyoma-embed` |
| 3 — Trinity (dense/MoE/symbolic) | sparse + symbolic | `vyoma-lm` (moe + lattice) — BOTH halves started ✅ |
| 4 — Ontological Store | knowledge on disk, retrieved | `vyoma-retrieval`, `vyoma-embed`, `vyoma-rag`, `vyoma-lm` (kv/knn/retro) |
| 5 — Self-evolution | bounded continual learning | `vyoma-lm` (evolve) — self-evolve via store + homeostasis ✅ |

Cross-cutting: `vyoma-tokenizer` (BPE), `vyoma-distill` (learn-from-teacher pipeline).

---

## Crates

### `vyoma-e1` — generative weights (the load-bearing bet), Pillar 1
- **What.** A fractal "Eternal Seed" (tiny generator) produces a target network's
  full weight vector; compared fairly against a plain model of the *seed's* size.
- **Why.** Pillar 1 is the one unproven, project-killing bet. If generation doesn't
  beat "just use a small model," the 8GB vision loses its compression engine.
- **Status.** ✅ Strong on redundant image weights: **423×** compression, improves
  with scale; 8-bit quant composes (508× bytes, ~free); 4-bit breaks.
- **What's left.** The image win is redundancy-bound; it does NOT transfer to
  language FFNs (see `vyoma-lm`). Generation is a *multiplier*, not the engine.
- **Run.** `EPOCHS=15 SEEDS=2 DATASET=mnist HIDDEN=256 ./target/release/vyoma-e1`

### `vyoma-ssm` — can the seed generate a sequence model? (Pillar 1 → 2)
- **What.** Fractal-generate a diagonal SSM's weights vs a same-size stored SSM.
- **Why.** Our backbone is an SSM, not an MLP — the E1 win must transfer or it
  doesn't help the real architecture.
- **Status.** ❌ (decisive, useful) SSM cores **resist generation** — a small
  *stored* SSM beats a generated one at every compression. Locked-in principle:
  **generative value ∝ target redundancy**; store lean cores, generate redundant mass.
- **What's left.** Nothing here — the negative result redirected effort correctly.
- **Run.** `EPOCHS=12 SEEDS=2 ./target/release/vyoma-ssm` (`MODE=hybrid` too)

### `vyoma-tokenizer` — byte-level BPE (ours, from scratch)
- **What.** 256-byte base + iterative most-frequent-pair merges to a target vocab;
  lossless round-trip; writes `data_cache/bpe_merges.txt`.
- **Why.** Char-level was a real ceiling; subwords ~halve sequence length and give
  the model linguistic units. Foundational; zero external deps.
- **Status.** ✅ vocab 2048 ≈ 3.2 bytes/token; BPE beats char 2.14 vs 2.85 bits/byte.
- **What's left.** Larger vocab / trained on the bigger distilled corpus.
- **Run.** `./target/release/vyoma-tokenizer`

### `vyoma-lm` — the integrated model + knowledge experiments
- **What.** Our LM: BPE + stored SSM backbone + generated FFN, plus modes:
  `diag` (does FFN width matter), `sweep` (gen-FFN vs plain vs dense), `kv` (E2
  memorize-vs-retrieve), `knn` (kNN-LM), `bpb` (bits-per-byte eval), `vyoma` (the
  assembled model), **`retro`** (E2 closed with OUR retriever — uses `vyoma-embed`),
  **`retrolm`** (the retro loop on REAL language, scored by masked bits/char),
  **`moe`** (Pillar 3 sparse MoE), **`g1`** (capability-per-RAM: stored int8 MoE ≈
  dense-big at ¼ the RAM — the Gate-G1-spirit win).
- **Why.** The place the corners come together and get scored on one honest scale.
- **Status.** ✅ assembled and running. Generation on language FFNs is modest
  (+0–4pt, width not depth). BPE + retrieval are the real levers. `retro` puts our
  retriever + our store + our LM in one loop, no teacher; `retrolm` runs it on real
  language (retro < random by −0.040 bits/char over 2 seeds — small, consistent, noisy).
- **What's left.** Scale (bigger distilled corpus + matched compute — the data lever,
  confirmed: 205→652 KB dropped BPB 3.87→2.09). Then MoE (Pillar 3). `retrolm` on real
  text: our retriever's neighbor beats a random one by −0.040 bits/char over 2 seeds
  (small, consistent-in-sign, noisy — needs scale to firm up).
- **Run.** `STEPS=3000 DM=160 DFF=512 SEQ=96 TOKENIZER=bpe MODE=bpb ./target/release/vyoma-lm`

### `vyoma-retrieval` — E2b: does retrieval hold as the library grows?
- **What.** Classical baseline: hashed char-trigram bags, cosine kNN, no training.
- **Why.** Establishes the *scaling law* of retrieval and a fair baseline to beat.
- **Status.** ✅ flat ~0.97 at dk=1024, collapses at dk=128 — reliability is governed
  by embedding dimension. Deliberately weak (superseded by `vyoma-embed`).
- **What's left.** Superseded; kept as the reference baseline.
- **Run.** `./target/release/vyoma-retrieval`

### `vyoma-embed` — learned neural retriever + persistent store (Pillar 4 engine)
- **What.** Our SSM-encoder retriever, trained contrastively (InfoNCE); plus the
  persistent on-disk **Ontological Store** (`MODE=store`, our `VYST` format). Exposed
  as a **library** (`lib.rs`) so `vyoma-lm` can retrieve with our encoder.
- **Why.** Retrieval is the decisive engine; the classical baseline was weak. This is
  the RETRO-quality, all-ours upgrade.
- **Status.** ✅ beats classical at matched dim (dk=128: ~0.70 vs 0.43), ~8× more
  dimension-efficient; scales with WIDTH not depth; keys quantize to **int4 free**;
  ~6× faster after batching the scan; store round-trips losslessly (260 B/fact).
- **What's left.** A semantic query set (as teacher-generated *training data*, never
  teacher-at-inference); attention pooling if we must beat the classical ceiling.
- **Run.** `STEPS=1500 DMODEL=128 ./target/release/vyoma-embed` · `MODE=store ...`

### `vyoma-rag` — E3: externalized knowledge at real scale (a lesson, not a component)
- **What.** phi4-mini (3.8B, via local Ollama) + lexical retrieval over novel facts.
- **Why.** Proved the retrieval thesis at real scale early (base 0/20 → +retrieval
  20/20). **A lesson we own, not shippable** — the teacher does NOT go in the system.
- **Status.** ✅ lesson banked; retrieval decouples capability from model size.
- **What's left.** Nothing to ship here; the ours-version is `vyoma-embed` + `retro`.
- **Run.** needs Ollama + phi4-mini. `./target/release/vyoma-rag`

### `vyoma-distill` — learn-from-teacher data pipeline
- **What.** The teacher generates a clean, varied corpus to disk (incremental,
  timeout-robust); OUR model trains on it (`vyoma-lm DATASET=distilled`).
- **Why.** Phase-1 mechanism: capability without frontier training compute. The
  student is ours; the teacher only teaches.
- **Status.** ✅ pipeline works; corpus grown to 652 KB; the data lever is confirmed.
- **What's left.** Scale the corpus toward MBs+ (the honest road to capability).
- **Run.** needs Ollama + phi4-mini. `COUNT=250 ./target/release/vyoma-distill`

---

## What's NOT built yet (honest gaps)
- **All five pillars now have ours-built corners** (P1 generation, P2 SSM, P3 MoE +
  symbolic lattice, P4 retriever + store, P5 self-evolution + homeostasis) — but all
  at **toy scale / synthetic tasks**. They establish *directions and mechanisms*, not
  a shipped model.
- **Scale** is the dominant remaining gap — everything is ≤~0.5 M params on one
  laptop. Capability = data × compute in the right ratio (Chinchilla), a real
  commitment, not an overnight run.
- **Full-system integration at scale** — the pillars are validated as separate modes;
  running them as one model on a real corpus with matched compute is the next epoch.

## The through-line
Generation is a redundancy-bound *multiplier*; the *engine* is externalized
knowledge (our retriever + our on-disk store) composed with a lean stored SSM
backbone, an honest BPE tokenizer, and int8 quantization — every corner ours,
scored on bits-per-byte and retrieval accuracy. See [`PROGRESS.md`](PROGRESS.md).
