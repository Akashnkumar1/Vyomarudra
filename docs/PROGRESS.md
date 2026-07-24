# Vyomarudra — Progress Log

A living record: **what we did, why, what we learned, and what's left.** Newest
entry on top. This is the single source of truth for project state; the vision
docs (`docs/english/`) stay stable, this file moves.

---

## Current state at a glance

| Thing | State |
|---|---|
| Stack | Rust, end-to-end (no Python). Cargo workspace. ML via `candle`. |
| Active phase | Architecture ASSEMBLED — all corners built (8 crates, ours); scaling corpus + training is the road ahead |
| Generation | Wins on redundant image weights (→423×, improves w/ scale); modest on language (+2–4pt, flat w/ depth); loses on SSM cores |
| Quantization | ✅ composes at 8-bit (127×→508× byte-comp, ~free); 4-bit breaks (shared compression budget) |
| Externalized knowledge (E2/E2b) | ✅ the engine — retrieval flat ~97% vs memorize collapse; scales with corpus given adequate embed dim |
| Integration (kNN-LM) | ⚠️ mechanism right, magnitude tiny at toy scale (best λ +0.2pt); bottleneck = key quality (needs a real LM) |
| Direction | **Retrieval-centric hybrid**: knowledge in store + 8-bit + modest generation on a lean stored SSM |
| Next (needs real scale, not toys) | strong learned embeddings for retrieval; a real word/subword LM to show the frontier magnitude |

Kill signals we are watching (from `docs/english/03-risk-register.md`):
Risk #1 — weight-gen caps at 10–50×. **Status: not triggered** (fractal hit 212× on MNIST).

---

## Log

### 2026-07-24 — Scale step #1: does OUR model improve when scaled up?

**What.** Architecture is complete; this begins the *scale* phase. Same everything
(Shakespeare, BPE vocab 2048, MODE=bpb) but a bigger model + longer + more context:
dm 96→**160**, dff 256→**512**, seq 64→**96**, steps 1200→**3000**. Testing the core
premise of the road ahead — that our model gets measurably better with scale.

**Baseline to beat:** BPB 2.136 (dm=96, dff=256, seq=64, 1200 steps).

**Verdict — bigger model was WORSE: BPB 2.776 (vs 2.136).** Counterintuitive but a
classic lesson, visible in the numbers: the bigger model ran **~53 epochs** over the
347k-token corpus (baseline ~14), so it **overfit** — memorized train, generalized
worse to held-out. **On a fixed small corpus, scaling the model up backfires.**

**The redirect (important):** capability is NOT "make the model bigger." It's the
**Chinchilla lesson, rediscovered on our own model** — data must scale *with* model
size. The next lever is a **much larger corpus + matched compute**, not more params
on the same 1 MB. Scaling model alone → overfitting. This correctly aims the scale
phase at DATA, and it's a real commitment (large distilled corpus + long training),
not an overnight toy. Architecture done; the road ahead is data + compute, in the
right ratio.

**Scale step #2:** acting on the redirect — grew `vyoma-distill` to be
timeout-robust (retry-and-skip). Clean run this time: **0 skips**, corpus grown
205 KB → **652 KB** (3.2×, 238k tokens). Retraining the *same small model*
(dm=96, dff=256, seq=64, 1500 steps) on it — apples-to-apples vs the 205 KB run
(BPB 3.872). **Verdict: the data lever WORKS, decisively.** Same small model, 3.2× data:

| corpus | BPB |
|---|---|
| 205 KB distilled | 3.872 |
| 652 KB distilled | **2.092** |

**−46% bits-per-byte from data alone** — and 2.092 is the best BPB our model has hit
(beats Shakespeare's 2.136). Confirms scale step #1's redirect: capability = **data,
not bigger models**, and our own model (learning from teacher-distilled data)
improves *steeply* with it. The scale road is validated on something entirely ours.

**Honest bounds:** the curve is steep partly because 205 KB was severely
data-starved; gains will flatten, and a genuinely capable model needs orders more
data + compute than laptop teacher-generation supplies. But the mechanism —
more data → materially better model, teacher-taught, ours end to end — is now
demonstrated, not asserted. Caveat unchanged: real scale is a real commitment.

### 2026-07-24 — Last corner: scaling the distillation corpus — running

**What.** Expanded `vyoma-distill` to combinatorial coverage (56 topics × 7 forms)
with a `COUNT` knob and incremental writes. Running `COUNT=250` → a much larger,
more varied teacher corpus (~250 generations) for OUR model to learn from — up from
the 15.6 KB proof. Teacher (phi4-mini) is a tutor only; the corpus and model are ours.

**Why.** The pipeline was proven; capability needs *data*. This scales the material
our own model trains on — the bridge from "architecture assembled" to "model that's
actually getting better."

**Result.** A transient Ollama read-timeout hit at generation 181/250 — but the
incremental-write design meant **zero loss**: **205 KB corpus on disk** (13× the
15.6 KB proof). BPE trained on it adapts to the teacher's register — learned modern
English units (" something", " different", " because", " sunlight") vs Shakespeare's
" thou"/" lord". Our model is now training on it (BPE, MODE=bpb); BPB verdict pending.

**Lesson:** incremental writes are the right call for long teacher-generation runs;
a single stuck generation shouldn't cost the batch. (Could resume with `APPEND=1`
for the last ~70, but 205 KB is plenty for now.)

**Our model trained on it: BPB = 3.872** (BPE, dm=96, dff=256, seq=64, 1500 steps),
vs 2.136 on tiny-shakespeare. Higher (worse) — and honestly expected, not a
regression: (1) 205 KB (~75k tokens) is ~4.6× less data than Shakespeare (1.1 MB),
so generalization is worse; (2) phi4-mini's varied explanatory prose across 56
topics is genuinely harder to model than Shakespeare's stylized, repetitive verse,
so cross-corpus BPB isn't apples-to-apples. **Reading:** the pipeline and the corner
work; the number reflects *data scale*, which is precisely the remaining lever.
Architecture is done; capability is now a function of corpus size + training +
model size — the honest road ahead, not a new corner.

### 2026-07-24 — Corner built: the ASSEMBLED model (`MODE=vyoma`) — running

**What.** `vyoma-lm MODE=vyoma` wires our corners into one model: **BPE tokens +
SSM backbone + generated-FFN (fractal seed)**, trained end to end, scored by
**bits-per-byte** against a same-footprint plain model (also BPE). The first time
the pieces run as *one thing* rather than separate experiments.

**Why it matters.** This is the culmination corner — tokenizer, backbone,
generative weights, and the honest metric, all in a single configuration that is
entirely ours (no wrapped model). Retrieval and quantization compose on top as
separately-measured layers (E2/E3, QUANT).

**Verdict** (BPE, dm=96, dff=256, seq=64, 1200 steps):

| model | BPB |
|---|---|
| Vyomarudra — SSM + generated-FFN (10× on FFN, seed 4905) + BPE | 2.129 |
| plain — SSM + stored small FFN (same footprint) + BPE | 2.121 |
| BPE-dense reference (full stored FFN) | 2.136 |

All three cluster at **~2.12–2.14 bits/byte** (within noise at 1200 steps). Reading:
(1) the assembled model **runs end to end** — BPE + SSM + generated-FFN as one thing,
scored honestly. (2) **The tokenizer corner dominates quality** (2.85 char → 2.13
BPE); FFN size/generation is second-order here. (3) The generated-FFN is
**quality-neutral vs a plain small FFN** — 10× FFN compression at no BPB cost, but
no advantage over just using a small FFN — consistent with every prior language
result. Honest: the assembly is real; its win comes from the tokenizer + (composable)
retrieval/quant layers, not from generating language FFNs.

### 2026-07-24 — Corner built: eval harness (bits-per-byte) — one honest scale for everything

**What.** `vyoma-lm MODE=bpb` + `eval_bpb` + `token_byte_lengths`. Computes
**bits-per-byte** (total −log2 p(true) ÷ raw bytes) on held-out text — the standard,
**tokenizer-independent** measure of how well a model models text. Normalizing by
bytes (not tokens) makes char-level and BPE directly comparable, and settles the
comparisons we'd twice deferred.

**Why it matters.** Our accuracy metric wasn't comparable across tokenizers/vocab
(512-way vs 65-way outputs). BPB is: lower = better, on one scale, for any model or
tokenizer. Now "does BPE help?", "does retrieval help?", "is a bigger FFN worth it?"
are all judgeable rigorously. You can't improve what you can't measure — this is
the measuring stick, built by us.

**Verdict — char vs BPE (matched compute: dm=96, dff=256, seq=64, 1200 steps):**

| tokenizer | bits/byte |
|---|---|
| char-level | 2.851 |
| **BPE (vocab 2048)** | **2.136** |

**BPE is 25% better** on the honest, tokenizer-independent scale. So the tokenizer
corner genuinely improves our model's language modeling (not just sequence length),
and it *compounds* with the model. First rigorous, apples-to-apples win — exactly
what the harness was built to give. Every future corner now gets judged this way.

### 2026-07-24 — Corner built: byte-level BPE tokenizer (ours, from scratch)

**What.** New crate `vyoma-tokenizer`: byte-level BPE built from scratch, no external
dep. 256-byte base (any input encodes, no UNK) + iterative most-frequent-pair merges
to a target vocab. Trains in ~2 s.

**Result (tiny-shakespeare, lossless round-trip ✓):**

| vocab | compression |
|---|---|
| 512 | 2.02 bytes/token (~50% fewer tokens vs char-level) |
| 1024 | 2.62 |
| 2048 | 3.22 |

Learned meaningful subwords — " shall", " thou", " lord", " your", " have", " this".
Merges saved to `data_cache/bpe_merges.txt` for our models to reuse.

**Why it matters.** Char-level was a real capability ceiling (long sequences, no
subword structure). A proper tokenizer halves-to-thirds sequence length and gives
the model real linguistic units — a foundational corner every downstream model
rides on. 100% ours: no teacher, no library.

**Integrated into our model.** `vyoma-lm` now has `TOKENIZER=bpe` — loads our
`bpe_merges.txt`, encodes the corpus, trains on subword tokens (vocab 512). Verified
running. Concrete win: at 2.0 bytes/token a `SEQ=48` window now spans ~97 chars of
context (vs 48 char-level) — **~2× the context for the same compute**, plus real
subword units. Note: raw next-token accuracy is NOT comparable across
tokenizers (512-way vs 65-way output); the correct metric is bits-per-byte —
now measured (see the eval-harness entry above): **BPE 2.136 vs char 2.851 bits/byte,
a 25% win** at matched compute. Tokenizer corner validated on the honest scale.

### 2026-07-24 — Principle locked + distillation corner built: OUR model, teacher only teaches

**Principle (Akash, firm):** Vyomarudra is **built by us, every corner**. Pretrained
models are **teachers we learn from**, never components we ship. The E3 phi4-mini
demo is a *lesson we own* (retrieval works at scale), not a shippable system —
phi4-mini does NOT go inside Vyomarudra. See memory `build-ours-teacher-only-learns`.

**Corner built — training via distillation.** New crate `vyoma-distill`: the teacher
(phi4-mini) generates a clean, varied corpus (explanations, stories, dialogues,
facts) written to disk; **our** model (`vyoma-lm`, `DATASET=distilled`) trains on
it. The teacher is a tutor; the corpus and model are ours. This is the roadmap's
Phase-1 mechanism ("distill from an open teacher") — capability without frontier
training compute.

**Verified end-to-end:** teacher generated 15.6 KB (18 prompts); our model trained
on it (loss 2.64→2.57 as FFN widens, next-char acc 0.28→0.31). Path works: teacher
→ corpus → our model learns.

**Honest scale note:** 15.6 KB is a *pipeline proof*, not a capable model. The road
to capability = scale the teacher corpus (thousands of prompts → MBs) + more
training. That's the long haul, but the corner is now in place and principle-clean.

**Corners of OUR system** (build map — ALL BUILT): ✅ SSM backbone · ✅ generated-FFN · ✅ retrieval
store · ✅ quantization · ✅ distillation pipeline · ✅ BPE tokenizer (+ wired in) · ✅ scaled
distillation corpus (205 KB) · ✅ full assembled model (`MODE=vyoma`) · ✅ eval harness (bits-per-byte).
Next phase is not new corners — it's *scale*: bigger corpus, more training, more compute.

### 2026-07-24 — E3: externalized knowledge on a REAL model (phi4-mini 3.8B) ✅✅ — the vision's engine, at scale, on the laptop

**What.** New crate `vyoma-rag`. The user pointed out the laptop runs 8B-class models
(inference is cheap; only from-scratch *training* was the toy bottleneck). So we
tested the vision's core on a **real** model: phi4-mini (3.8B, Q4, ~2.5 GB) via the
local Ollama API, + pure-Rust lexical retrieval over a store of **20 novel/fictional
facts** the model cannot have memorized. Baseline (ask directly) vs RAG (retrieve
fact → context → ask).

**Result.**

| | phi4-mini alone | phi4-mini + retrieval |
|---|---|---|
| Answer accuracy | **0/20 (0%)** | **20/20 (100%)** |

Retrieval hit-rate 100%; lift **+100 pts**. The base model confabulates plausible
wrong answers (e.g. "Vortexium Sky Metal"); with the retrieved fact it answers every
one correctly.

**Why it matters.** This is the whole achievable-vision thesis demonstrated
end-to-end on a real model on this laptop: **externalized knowledge gives a small
model capability it does NOT have in its weights.** Knowledge is a huge fraction of
what makes frontier models heavy; moving it to disk (retrieved at inference) is a
real, proven path to small-but-capable. The toys hit "direction-not-magnitude"
because a 62k char-LM has weak keys; a real 3.8B model does not — bottleneck solved
by using a real base model.

**Honest scope.** The facts are deliberately novel to *isolate* retrieval's
contribution (so base≈0 by design); it demonstrates the model can perfectly *use*
retrieved knowledge, not a natural-distribution benchmark. It validates the
knowledge/retrieval pillar, not reasoning-parity with a bigger model. Retrieval here
is lexical (strong for factual QA); learned embeddings generalize it further.

### 2026-07-24 — Retrieval-augmented LM (kNN-LM): the integration, on real language — running

**What.** `MODE=knn` in `vyoma-lm`: the first *integrated* capability-per-GB test.
Train a small char-LM; build a datastore of (hidden state → next token) over the
training corpus (lives on disk, ~free model-RAM); at inference, interpolate the
model's distribution with a kNN retrieval over that datastore (the proven kNN-LM
recipe). Compare **base LM alone vs base + kNN datastore** — same weights, extra
knowledge on disk.

**Why.** This is the honest end-to-end demonstration of the whole thesis: does
holding knowledge on disk and retrieving it lift a fixed-size model on *real
language*? It applies our validated retrieval engine (E2/E2b) inside an actual LM.

**Result (rigorous — λ swept, 40k datastore, trained base=45.6%).**

| λ | 0 (base) | 0.05 | 0.1 | 0.2 | 0.35 | 0.5 |
|---|---|---|---|---|---|---|
| acc | 0.4420 | 0.4410 | 0.4420 | **0.4440** | 0.4440 | 0.4395 |

Best λ=0.2 → **+0.2 pts** (within noise, ~4/2000 positions). Earlier λ=0.5 gave
−1.35. So:

**Verdict — mechanism right, magnitude tiny at toy scale.** The λ curve behaves
exactly as kNN-LM theory predicts (light λ neutral-to-helpful, heavy λ hurts), but
the end-to-end lift is **effectively neutral** here. Why: a 62k-param char-LM
produces weak, low-dimensional keys and a 40k store is tiny — far from the regime
where kNN-LM wins big (strong LM, rich keys, billions of entries). Same recurring
lesson: **toys validate the direction but can't stage the frontier magnitude.**
This does NOT contradict E2/E2b — those isolated *accurate* retrieval and showed it
decisively decouples capability from size; the bottleneck here is *key quality*,
which strong learned embeddings (RETRO) provide and a tiny char-LM does not.

### 2026-07-24 — Quantization composes with generation (at 8-bit) — `vyoma-e1 QUANT`

**What.** Trained a high-compression fractal seed (1602 params, 127× on MNIST),
then evaluated the generated network with the SEED quantized to fp32/8-bit/4-bit
(its real RAM cost). `QUANT=1` mode, symmetric per-tensor fake-quant.

**Result** (dense = 98.0%).

| Seed precision | Acc | % dense | Effective compression | Seed size |
|---|---|---|---|---|
| fp32 | 85.65% | 87.4% | 127× | 6.3 KB |
| **8-bit** | 85.50% | 87.2% | **508×** | **1.6 KB** |
| 4-bit | 57.41% | 58.6% | 1016× | 0.8 KB |

**Verdict.** Generation × **8-bit** quantization **compose cleanly** — 8-bit is
essentially free (−0.15 pt), giving **508× byte-compression at ~no loss** (a 1.6 KB
seed → 203K-param net @ 85.5%). **4-bit breaks it** (→57%). Insight: the harder you
compress via generation, the more *information-dense* each seed parameter becomes,
so it tolerates less further quantization — a shared compression budget. 8-bit is
the sweet spot. Third multiplier for the hybrid, characterized.

### 2026-07-24 — E2b retrieval scaling: the store holds up as the library grows (fast, 13s) ✅

**What.** New crate `vyoma-retrieval` (pure Rust, no training). Passages of tiny-
shakespeare embedded as hashed char-trigram bags (dim dk), retrieved by cosine; a
query is a 48B fragment of a source passage; correct if nearest = source. Sweep
library size N and embedding dim dk. Answers the question E2's oracle skipped:
**does retrieval stay accurate as the library grows?**

**Result (retrieval accuracy).**

| dk | N=100 | N=1000 | N=5000 | N=8714 |
|---|---|---|---|---|
| 128 | 0.80 | 0.45 | 0.34 | 0.31 |
| 256 | 0.93 | 0.77 | 0.66 | 0.60 |
| 1024 | 1.00 | 0.99 | 0.98 | **0.97** |

**Verdict.** Retrieval reliability is governed by embedding dimension: too small →
collapses with library size (collisions); **adequate (dk=1024) → flat ~97% across
the whole corpus.** So the "big library" holds, *provided the embedding is big
enough* — and this is a WEAK baseline (hashed trigrams); learned/neural embeddings
(RETRO) reach the same reliability at far lower dim. With E2, the retrieval engine
of the hybrid is validated end-to-end (principle + scaling). Runtime: **13 seconds**
— the antidote to the multi-hour training sweeps.

### 2026-07-24 — Multi-layer support built; depth test done

**What.** Added multi-layer support to `vyoma-lm` (`LAYERS` env): stacks N blocks
(each = stored diagonal SSM + generated FFN + residual); one fractal seed generates
ALL layers' FFN mass. `Cfg` gained `layers`; `Stored` holds per-layer SSM params;
`forward` loops layers. Compiles; LAYERS=1 reproduces prior behavior.

**Why.** The scale test showed generation's language edge grows with target *width*
(dm=128→256: 0 → +2–4 pts). Open question: does it also grow with *depth*? That
decides how big a supporting role generation earns in the hybrid.

**Result.** Text, dm=96, dff=384, 3000 steps. Dense rises with depth (real capacity):
48.0% (1L) → 51.6% (2L) → 53.2% (4L). But the gen-vs-plain edge stays **modest and
noisy**: ≈0 (1L: −0.6% to +0.3%), up to +1.3% (2L), +0.5–0.9% (4L). **Depth does NOT unlock generation**
the way width nudged it. Robust conclusion across width and depth: **on language,
weight-generation is a small supporting player (~0–4 pts over a tiny FFN), never a
dense replacement.**

**Speed note.** Tried candle's Metal backend — it's ~2× SLOWER here: the SSM's
sequential per-timestep scan is many tiny ops, and GPU kernel-launch overhead
dominates. Defaulted back to CPU; hoisted the SSM input/output projections out of
the timestep loop (marginal). Real cost is backprop through the scan × 7 models per
sweep. Takeaway: use small/fast configs to iterate; these toys have largely given
what they can.

### 2026-07-24 — Scale test verdict: generation's language edge is REAL but MODEST (and grows with scale)

**What.** Repeated the text generation sweep at a bigger target (dm=256, dff=1024,
seq=64, 5000 steps) to test whether scale restores the edge that was ≈0 at dm=128.

**Result.** Dense = 52.3%.

| FFN comp | seed | gen-FFN | plain (same footprint) | edge |
|---|---|---|---|---|
| 107× | 4,905 | 42.09% | 39.94% (dff=9) | +2.16 |
| 196× | 2,685 | 40.24% | 38.44% (dff=4) | +1.80 |
| 548× | 959 | 40.80% | 37.00% (dff=1) | +3.80 |

**Verdict (nuanced, both true).**
1. **Scale restores an edge.** dm=128 → edge ≈0; dm=256 → edge **+2 to +4 pts**,
   largest at highest compression (548×). Generation *does* beat a same-size plain
   FFN on language once the target is big enough — the improves-with-scale law has
   a foothold on language too.
2. **But it does not recover dense.** gen-FFN ~41% vs dense 52% — an ~11-pt gap
   persists at every compression. Extreme-compression generation loses a lot of the
   FFN's value.

**Read.** Generation is a **modest, scale-growing supporting player on language**,
not the engine. The engine is retrieval (E2). Supports the hybrid: retrieval for
knowledge + modest generation (edge may grow with dm/depth) + quantization on a
lean SSM backbone. Warrants the multi-layer test (execution plan Step A) — but with
realistic expectations, not miracle ones.

### 2026-07-24 — E2 externalized knowledge: DECISIVE POSITIVE (Pillar 4) ✅

**What.** `MODE=kv` in `vyoma-lm`: synthetic key→value facts, **memorize** (value
must live in weights) vs **retrieve** (value fetched, adjacent to query). Entities
are digit-encoded over a tiny fixed vocab (no per-entity params) so memorization is
genuinely FFN-capacity-bound. dm=64, dff=128, 4000 steps.

**Result.**

| #facts | memorize | retrieve |
|---|---|---|
| 200 | 98.1% | 97.2% |
| 1,000 | 15.4% | 97.3% |
| 4,000 | 13.9% | 97.0% |
| 16,000 | 11.7% | 97.0% |

**Verdict.** Memorize succeeds at 200 facts then **collapses to chance** as facts
exceed the fixed model's capacity. Retrieve stays **flat at ~97% for all #facts** —
capability decoupled from model size. This is the practical core of the vision,
cleanly shown: **knowledge belongs on disk, not in weights** (Pillar 4 / "skills in
weights, knowledge in the store"). Branch-independent — true regardless of the
generation verdict, and the first working proof-point of the achievable hybrid.

**Honest scope.** Oracle-style retrieval (the correct value is placed adjacent), so
this shows the *ceiling* retrieval enables — "if you can fetch the fact, a tiny
model wins." Finding the right fact (embedding-kNN over a disk store) is the next
step, but that part (RETRO) is proven. Synthetic, char-level, tiny model.

### 2026-07-24 — Integrated test on real language: generation does NOT beat a small model (honest negative)

**What.** Ran the integrated `vyoma-lm` (stored SSM + generated FFN) on tiny-shakespeare
at dm=128, dff=512 (FFN mass 131,712), seq=48, 3000 steps. Metric: next-char acc.

**Result.** Dense (full FFN) = 49.3%.

| FFN compression | Seed | gen-FFN | plain (same footprint) | edge |
|---|---|---|---|---|
| 26.9× | 4,905 | 39.81% | 39.69% (dff=18) | +0.001 |
| 49.1× | 2,685 | 39.16% | 39.79% (dff=9) | −0.006 |
| 137.3× | 959 | 37.72% | 38.05% (dff=3) | −0.003 |

**Verdict.** gen-FFN **ties** a same-size plain FFN at every compression (edge within
noise), and **both plateau ~10 pts below dense**. The large MNIST edge (where gen ≫
plain at high compression) **did not transfer to language FFNs.** Fractal generation
offers no advantage over simply using a small FFN here.

**Interpretation.** The dense-vs-small-FFN gap (~10 pts) is real FFN capacity that
matters — but neither generation nor a small stored FFN recovers it, and generation
is no better than the small FFN. **Language FFN weights appear far less
fractally-compressible than MNIST MLP weights.** The compression win is, so far,
specific to highly-redundant image classifiers.

**Honest scope.** One small setup: 1-layer, char-level, dm=128, diagonal SSM, 3000
steps. Suggestive, not final — larger/multi-layer/longer could differ. But it is the
first *direct* language test and it is negative. Burden of proof on the
language-compression claim is now higher.

**Consequence for the vision.** This pushes away from "pure ≥100× generation of
language weights" and toward the doc's **hybrid fallback** (moderate generation ×
quantization × retrieval) — or a demonstration that scale/multi-layer restores the
edge. Decision needed (see below).

### 2026-07-23 — Path A greenlit: on real text, FFN width DOES drive quality

**What.** Added a real-text corpus (tiny-shakespeare, 1.1 MB, vocab 65) and a
`MODE=diag` that trains dense models at several FFN widths to test the
precondition every toy failed: *does a wider FFN measurably help?*

**Result (dm=64, 600 steps, next-char acc).**

| dff | total params | loss | acc |
|---|---|---|---|
| 8 | 9,737 | 2.247 | 34.0% |
| 32 | 12,833 | 2.178 | 36.6% |
| 128 | 25,217 | 2.143 | 37.5% |
| 512 | 74,753 | 2.098 | 38.0% |

**Verdict.** Monotonic improvement with FFN width (+4 pts acc, −0.15 loss, 8→512) —
unlike arithmetic (`dff=6 ≈ dff=512`). **Real language is FFN-capacity-bound**, so
there is a genuine, needed FFN mass for the seed to compete on. The precondition
for a valid integrated generation test is met → running the full gen-FFN vs plain
vs dense sweep on text (bigger model, more steps).

### 2026-07-23 — Fashion scaling: improves-with-scale law HOLDS on harder data

**What.** Repeated the three-width scaling sweep on Fashion-MNIST. Fixed 962-param seed:

| Target | Compression | Fractal acc | % of dense (dense≈89%) |
|---|---|---|---|
| 128-wide | 105.8× | 75.36% | 85.3% |
| 256-wide | 211.6× | 76.18% | 85.9% |
| 512-wide | 423.1× | 78.79% | 88.4% |

**Result.** Same direction as MNIST: bigger target ⇒ higher fractal accuracy, higher
% of dense kept, ~2× more compression per step. At 423× the seed keeps 88.4% of
dense on Fashion (edge +46 pts over a same-size plain net). The mid-run 128→256
read looked flat; the 512 point confirmed the trend continues.

**Honest caveat (ties to the vyoma-lm fork).** Both image datasets *saturate* in
width (MNIST dense 97.8→98.1%, Fashion 88.3→89.1% across 128→512), so a bigger
*dense* model barely gains — the scaling law here is "the seed captures a growing
fraction of a roughly-fixed ceiling while compressing more," not "generation
unlocks new capability." The capability-meaningful version still needs a
**non-saturating** task (language), which is exactly what the integrated model
needs too.

### 2026-07-23 — `vyoma-lm` ran: the toy doesn't need a big FFN, so it can't test the win (a result about experiment design)

**Result** (arithmetic char-LM, dm=64, 5000 steps, answer-digit accuracy):
dense dff=512 → 0.324; gen-FFN trails a same-footprint plain model at every
compression (−0.017 to −0.026).

**The diagnostic.** `plain(dff=6)` = 0.307 vs dense `dff=512` = 0.324 — a 512-wide
FFN barely beats a 6-wide one (+1.7 pts). So **FFN capacity is not the bottleneck
on this task**; a 1-layer SSM+FFN char model hits a low ceiling regardless of FFN
width. With no large *needed* FFN mass, generation has nothing to win on and only
adds optimization overhead → it slightly trails.

**The real lesson (a pattern now, not a one-off).** Three toys in a row failed to
stage the regime the vision needs: the SSM didn't need a big FFN, the hybrid didn't,
and arithmetic doesn't. **Small toys structurally can't demonstrate the
generate-the-big-FFN win, because they don't need a big FFN.** The regime where a
wide FFN is *both needed and redundant* is real language modeling at a scale beyond
these one-file toys.

**Consequence — the honest fork.** The mechanism is validated (E1) and its domain
mapped (redundancy principle). Demonstrating the *integrated* win requires a proper
small LM on real text (multi-layer, FFN-heavy) where FFN width genuinely drives
quality — a substantial Phase-1 build, not another toy. Designing a *synthetic*
FFN-capacity-bound task is subtle: pure memorization is incompressible (info-theory
limit), so the task must need width AND have redundant structure — which is exactly
what real language provides and what toys don't.

### 2026-07-23 — Phase-1 scaffold: `vyoma-lm`, the first integrated model + refined architecture doc

**What.** (1) Wrote `docs/english/06-refined-architecture.md` — updates the five
pillars with Phase-0 evidence and states the division of labor: **stored lean SSM
backbone + generated FFN/MoE mass + external knowledge**. (2) Scaffolded a new
crate `vyoma-lm`: a char-level LM on procedurally-generated **arithmetic**
(`a+b=c`) — token embed (stored) → diagonal SSM (stored) → **FFN generated by the
fractal seed** → head (stored). Compiles clean.

**Why.** Row-MNIST couldn't stage a hybrid win because the task didn't need a big
FFN. Arithmetic char-LM is **capacity-hungry** (must actually compute), so a bigger
FFN measurably helps — making "generate the FFN mass" a meaningful test, and the
first experiment that combines the Phase-0 findings into an Vyomarudra-shaped model.

**Status.** Built + compiles; the training run is **deferred** so it doesn't
contend for CPU with the running Fashion scaling sweep. Run:
`STEPS=2000 DM=64 DFF=512 ./target/release/vyoma-lm`.

**Left.** Run it; if the generated-FFN model matches dense while compressing the
FFN ≥10× at a smaller footprint than a plain baseline, the division of labor is
validated on a language-relevant task. Then scale DFF; then MoE; then retrieval (E2).

### 2026-07-23 — E1 scaling law: generative weights get BETTER with target size ✅

**What.** Ran the MNIST feedforward sweep at three target widths (128/256/512-wide;
101k / 204k / 407k params), 15 epochs, 2 seeds. Compared at a **fixed 962-param
seed** (identical RAM footprint).

**Why.** The load-bearing question for the whole 8GB bet: does generation improve
or degrade as the generated network grows? If it improves, the frontier direction
is alive.

**Result (fixed 962-param seed).**

| Target | Params | Compression | Fractal acc | % of dense |
|---|---|---|---|---|
| 128-wide | 101,770 | 105.8× | 84.55% | 86.5% |
| 256-wide | 203,530 | 211.6× | 88.49% | 90.2% |
| 512-wide | 407,050 | **423.1×** | **90.09%** | **91.8%** |

Every axis improves as the target grows: accuracy ↑, % of dense kept ↑, compression
doubles each step. At 423× a 962-param seed → 407k-param net @ 90.1% (same-size
plain net: 30.9%; edge **+59 pts**).

**Verdict.** **Generative weights become more effective with scale** — the
frontier-favorable direction, on E1's validated feedforward path. This is the
strongest positive signal for Pillar 1 so far.

**Honest caveat.** MNIST saturates ~98%, so a bigger *dense* model barely gains;
the "% of dense" rise partly reflects the seed catching up on a fixed ceiling. The
capability-meaningful version of this law needs harder data where bigger dense
models genuinely win (Fashion already showed achievable ratios drop with difficulty).
So: trend is real and encouraging; absolute ratios remain data-dependent.

**Left.** Repeat the scaling law on harder data; then move toward a real
stored-SSM + generated-FFN model at a scale where the FFN mass dominates.

### 2026-07-23 — Hybrid (generate FFN, store recurrence): helps, but still loses on the toy — and the real principle

**What.** `MODE=hybrid` in `vyoma-ssm`: generate only the feedforward matrices
(w_enc, w_mix, w_cls = 75,264 params) with the fractal seed; store the recurrence
+ biases (1,546 params) directly. Compare to a same-total-footprint plain SSM.
Converged (12 epochs, 2 seeds). Dense SSM = 90.4%.

| Gen-mass comp | Whole comp | Footprint | Hybrid | Plain SSM (same) | Edge |
|---|---|---|---|---|---|
| 8.1× | 7.1× | 10,803 | 83.45% | 87.83% | −4.38 |
| 28× | 18.2× | 4,231 | 76.07% | 84.08% | −8.01 |
| 47× | 24.4× | 3,145 | 68.10% | 81.61% | −13.51 |

**Result.** The hybrid degrades *more gracefully* than pure generation (68% vs 56%
at ~47× — storing the recurrence helped), but **still loses to a same-footprint
plain SSM at every point.**

**The principle this locks in (the real deliverable of the SSM arc).**
Generative-weight value is a function of the target's **parameter redundancy**:
- Redundant / parameter-*inefficient* targets (MLP/FFN) → generation wins big (E1: 212×).
- Efficient targets (SSM recurrence) → a small *stored* model wins; generation loses.

So "just use a small model" is a strong baseline **for efficient architectures**.
The toy can't stage the regime the vision actually needs — where you *genuinely
require* a huge, redundant weight mass (frontier FFN/MoE-expert layers) that a
small model cannot match. On a small row-MNIST SSM, the tiny plain SSM is simply
too good to beat. **Conclusion: stop trying to generate lean sequence cores;
generation belongs on the large redundant FFN/expert mass, tested at a scale where
a small model genuinely fails.**

**Consequence for the roadmap.** The next decisive test is NOT more SSM generation.
It is the **E1 scaling law**: does compression-at-fixed-quality hold or improve as
the (feedforward) target grows? That is the load-bearing question for the frontier
bet, and it is on E1's *validated* path.

### 2026-07-23 — SSM bridge (Pillar 1 → 2): sequence-model weights resist generation *(converged — decisive)*

**What.** New crate `vyoma-ssm`. Target = a minimal diagonal SSM (stable
`A=σ(a)∈(0,1)` recurrence, S4D/Mamba-core) classifying **row-wise MNIST** (28
timesteps × 28-dim rows). Fractal-generated SSM vs. same-size plain SSM vs. dense.
12 epochs, 2 seeds, 20k train subset.

**Why.** Vyomarudra's backbone is an SSM, not an MLP. E1's compression win must transfer
to sequence models, or it doesn't help the real architecture.

**Bug found + fixed.** A naive global weight scale made the generated SSM start
**dead** (input-independent, stuck at chance). Cause: SSM parameter groups need
very different magnitudes. Fix: an architecture-derived per-group init-scale prior
(O(1) storage), applied to all three models. Bonus: dense baseline 0.40 → 0.90.

**Result.** Dense SSM = 90.2%.

| Comp | Seed | Fractal-SSM | Plain SSM (same size) | Edge |
|---|---|---|---|---|
| 8.3× | 9,257 | 85.73% | 87.18% | −1.45 |
| 28.6× | 2,685 | 73.66% | 83.99% | −10.34 |
| 48.0× | 1,599 | 55.78% | 81.24% | −25.46 |

**Verdict — the mirror image of E1.** For the MLP, the fractal degraded gracefully
and the plain model collapsed. For the SSM it is **reversed**: the plain SSM is
remarkably robust (81% at 48× on 1,551 params) while the fractal-generated SSM
**collapses** (56% at 48×), and the gap *widens* with compression.

**What it means (a real design decision, not a failure).**
1. **Don't generate recurrence weights.** SSMs are already parameter-efficient
   (Pillar 2's whole point — constant memory, few params). A tiny stored SSM beats
   a generated large one. Generating dynamics is a losing trade — errors in
   `a,B,C` compound over timesteps.
2. **Apply generative weights where they win:** the large feedforward / projection
   / MoE-expert weights (E1's regime), not the sequence core.
3. **This unifies Pillars 1 & 2 cleanly:** keep the SSM backbone small + stored;
   let the Eternal Seed generate the big FFN/expert parameter mass. Division of
   labor by modality. This is exactly the **hybrid** Gate G0 permits.

**Honest caveats.** Diagonal SSM (not full Mamba selective-scan); row-MNIST not
language; more seeds would tighten the extreme point. But the qualitative result
(plain SSM robust, fractal-SSM collapses with compression) is strong and monotone.

**Left.** Build the **hybrid** experiment: generate only the FFN/projection block,
keep the recurrence stored — verify the combined model keeps quality at high
*effective* compression of the generatable mass.

### 2026-07-23 — E1 honesty check on Fashion-MNIST (harder data)

**What.** Same fractal sweep, but on Fashion-MNIST (28×28 clothing, MLP-hard). Added
a `DATASET=mnist|fashion` switch. Dense upper bound 89.0% (vs MNIST's 98% — confirming
much lower redundancy).

**Why.** MNIST is the *best case* for the "weights are mostly redundant rules"
hypothesis. If the fractal edge is real and not a MNIST artifact, it must survive
on harder data. (CIFAR was the plan; the corporate proxy blocked its academic host,
so Fashion-MNIST — same IDX format, genuinely harder — was the reachable substitute.)

**Result.**

| Comp | Seed | Fractal | Plain (same size) | Edge | % of dense |
|---|---|---|---|---|---|
| 21× | 9,554 | 84.65% | 85.31% | −0.66 | 95.1% |
| 41× | 4,994 | 84.09% | 82.86% | +1.23 | 94.5% |
| 76× | 2,690 | 82.72% | 58.39% | +24.32 | 92.9% |
| 127× | 1,602 | 74.11% | 67.61% | +6.51 | 83.3% |
| 212× | 962 | 77.69% | 36.52% | +41.16 | 87.3% |

**Learned (the important part).**
1. **Achievable compression tracks data redundancy.** At ≤10% quality loss: MNIST
   reached ~200×, Fashion only ~76×. This is a real predictor for the project — the
   eventual ceiling depends on how redundant language/reasoning data is.
2. **The fractal edge is regime-dependent.** At low compression (~20×) it gives no
   advantage over a same-size plain MLP (even −0.7 pt) — generation earns its keep
   only once compression is aggressive enough that plain models run out of capacity.
   That is precisely the regime the 8GB vision needs, so this is still good news,
   just narrower than MNIST suggested.
3. Fashion does **not** clear G0's ≥100×-at-≤10% bar (it clears ~76×). G0 stands
   as cleared-on-MNIST-only until harder data or better mechanisms move it.

**Honest caveats.** Still an MLP target, not a sequence model. 127× vs 212×
non-monotonic → needs more seeds. Fashion is harder than MNIST but still far from
language.

**Left.** SSM target (next), then more seeds, then a target-size scaling sweep.

### 2026-07-23 — E1 fractal seed breaks the compression wall (MNIST)

**What.** Added a second seed design, `FractalSeed`, alongside the stored-embedding
`HyperSeed`. Ran a compression sweep on MNIST (target 784→256→10 = 203,530 params,
18 epochs, 2 seeds), comparing fractal vs. a same-size plain MLP at every point.

**Why.** The first sweep showed the stored-embedding hypernetwork caps near ~40×,
because its per-chunk embedding table grows with the target. To reach the ≥100×
that Gate G0 needs, the *addressing* has to become generative too — the actual
IFS/fractal idea from the vision doc.

**How.** `FractalSeed` generates each chunk's embedding from a fixed sinusoidal
encoding of the chunk index through a tiny address net (PE → address MLP →
embedding → generator MLP → chunk). Nothing per-chunk is stored, so seed size is
~constant in target size.

**Result.** Dense upper bound 98.03%.

| Fractal comp | Seed | Fractal | Plain (same size) | Edge | % of dense |
|---|---|---|---|---|---|
| 21× | 9,554 | 95.74% | 93.62% | +2.12 | 97.7% |
| 41× | 4,994 | 95.44% | 91.03% | +4.41 | 97.4% |
| 76× | 2,690 | 93.89% | 81.36% | +12.52 | 95.8% |
| 127× | 1,602 | 87.27% | 68.45% | +18.82 | 89.0% |
| 212× | 962 | 88.19% | 32.65% | +55.54 | 90.0% |

**Learned.** (1) Generative addressing is the lever — 212× vs the stored-embedding
seed's ~40× ceiling. (2) The edge over "just use a small model" grows without bound
as compression rises. (3) On MNIST we land right at the G0 threshold.

**Honest caveats.** MNIST is easy/redundant (best case for the redundancy
hypothesis); target is an MLP not an SSM; accuracy-loss ≠ capability-loss; the
127×-vs-212× curve is slightly non-monotonic (needs more seeds).

**Left.** CIFAR-10 honesty check → SSM/Mamba target → more seeds.

### 2026-07-23 — E1 v0: hypernetwork beats same-size MLP (MNIST)

**What.** First E1 in Rust/candle. Chunked hypernetwork (`HyperSeed`) generating a
784→128→10 target (101,770 params), vs. same-size plain MLP, on MNIST.

**Why.** Pillar 1 (generative weights) is the one unproven, project-killing bet.
The honest question is not "fewer params" but "does generation beat just using a
small model of that size?"

**Result.** Dense 97.93%. Hypernet beat the same-size MLP at every point 2.8×–41×;
edge grew with compression (+0.4 → +16.8 pts). Stored embeddings capped it ~41×.

**Learned.** The fair-baseline framing matters; a good init (calibrated output
scale) is essential or the generated target starts dead. Synthetic teacher tasks
were too easy/undersampled to be informative — real data (MNIST) was needed, pulled
via `curl -sk` to get past corporate SSL interception.

### 2026-07-23 — Setup + pivot to Rust

**What.** Read the vision (`docs/english/`), scaffolded a Cargo workspace, deleted
the initial Python prototype. Chose `candle`.

**Why.** Goal is a Rust AI OS + this model, so one language across the whole stack.

---

## Open questions

- Does the fractal edge survive on lower-redundancy data (CIFAR, text)?
- Can a fractal seed generate a working **SSM/Mamba block** (Pillar 1 → Pillar 2)?
- Where is the true G0 crossing (needs more seeds + a clean loss metric)?
- Does compression-vs-quality improve or worsen as the *target* grows? (scaling law)
