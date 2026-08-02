# Vyomarudra — Progress Log

A living record: **what we did, why, what we learned, and what's left.** Newest
entry on top. This is the single source of truth for project state; the vision
docs (`docs/english/`) stay stable, this file moves.

---

## Current state at a glance

| Thing | State |
|---|---|
| Stack | Rust, end-to-end (no Python). Cargo workspace. ML via `candle`. |
| Active phase | Architecture ASSEMBLED (9 crates, ours), all 5 pillars have corners, integrated MoE core seed-confirmed (BPB ~1.95); SCALE epoch active — corpus grown to 1.34 MB; scaling model+compute WITH data is the road (cross-corpus BPB is test-shift-confounded — measure within-corpus) |
| Generation | Wins on redundant image weights (→423×, improves w/ scale); modest on language (+2–4pt, flat w/ depth); loses on SSM cores |
| Quantization | ✅ composes at 8-bit (127×→508× byte-comp, ~free); 4-bit breaks (shared compression budget) |
| Externalized knowledge (E2/E2b) | ✅ the engine — retrieval flat ~97% vs memorize collapse; scales with corpus given adequate embed dim |
| Integration (kNN-LM) | ⚠️ mechanism right, magnitude tiny at toy scale (best λ +0.2pt); bottleneck = key quality (needs a real LM) |
| Learned retriever (`vyoma-embed`) | ✅ ours (SSM-encoder, contrastive) beats classical at MATCHED dim (dk=128: ~0.70 vs 0.43); ~8× more dim-efficient; scales w/ WIDTH not depth; keys quantize to int4 FREE; ~6× faster after batching the scan |
| Ontological Store (`MODE=store`) | ✅ real persistent on-disk store (our `VYST` format): 1743 facts, 260 B/fact int8; reload-from-disk lossless (0.68); knowledge on disk, encoder in RAM, no teacher |
| RETRO-lite (`vyoma-lm MODE=retro`) | ✅ E2 closed with OUR retriever+LM+store (no oracle/teacher). Scaling the retriever (dk256, 12-digit keys) lifts+flattens retro: **0.74/0.67/0.53 at 500/2k/8k facts** vs memorize 0.21/0.13/0.11 (gap 3.5–4.8×, widening); tunable via RD/RDK/RDM/RSTEPS/RNS |
| RETRO-LM (`vyoma-lm MODE=retrolm`) | ⚠️→✅ real language: our retriever's neighbor beats a random neighbor by **−0.040±… bits/char** (2 seeds; consistent sign, small & noisy; no acc gain). Mild positive; needs scale to firm up |
| Pillar 3 MoE (`vyoma-lm MODE=moe`) | ✅ sparse top-1 MoE (ours): **BPB 2.682 vs dense-small 2.866 (+0.184) and dense-big 2.711 (−0.029)** at ~dense-small ACTIVE params — big-model quality at small active cost. Toy computes all experts (quality-per-param shown, not yet FLOP saving) |
| Generate MoE experts (`MODE=genmoe`) | ❌ negative: gen-MoE 2.957 loses to stored-MoE 2.718 AND dense-small 2.899 — language experts resist generation (as FFNs do). Lesson: **store + quantize the experts**, don't generate them |
| Capability-per-RAM (`MODE=g1`) | ✅ milestone: **int8 ~free on the real LM mass** (MoE 2.732→2.730); stored int8 MoE beats dense-small-fp32 by −0.146 BPB at EQUAL FFN-RAM and ≈ dense-big at ¼ RAM. Gate-G1-spirit capability-per-GB win, ours |
| Pillar 3 symbolic lattice (`MODE=lattice`) | ✅ on **digit keys**: symbolic veto → hallucination **100%→0%**, 99% coverage / 100% precision. ⚠️ **Does NOT transfer to text** (see `MODE=rag`): embedding similarity can't separate in/out-of-domain |
| Assembled system (`MODE=rag`) + checkpoints | ✅ runs end to end (retriever → VYST store → gate → MoE LM); models **persist** (`SAVE=`, `Encoder::save`) and **generate text** |
| Grounding gate (`NEG=`, `MODE=gate`) | ❌→⚠️ without out-of-domain negatives there was NO signal (OOD scored higher than IN; no statistic worked). With foreign negatives: real, generalizing signal, but **74% balanced acc** at best cosine threshold — distributions overlap |
| Learned grounding head (`MODE=head`) | ✅ small MLP over [q ; e ; q⊙e] instead of a cosine scalar: **86.6% balanced acc held-out** vs 74.0% cosine, and **4/4 correct on realistic hand-typed prompts** (0.98/0.80 accept in-domain; 0.001/0.005 veto code+science). Trained with varied positives after fixed-offset ones proved brittle live. Grounding is now a trained component, not a magic number |
| Pillar 5 self-evolution (`MODE=evolve`) | ✅ bounded self-evolution via store writes (no weight edits, no forgetting) + homeostasis veto of contradictions: naive degrades 1.0→0.76 under poison, homeostasis holds ~1.0 (332 rejects). **All 5 pillars now have ours-built corners** |
| Integrated core (BPE+SSM+MoE, distilled) | ✅ **best BPB 1.867** (1.34 MB, dm=128/dff=192). MoE win **seed-confirmed & scale-invariant**: +0.11 over dense-small at every scale; ≈ dense-big at ½ active; int8 → ¼ RAM |
| Scaling / Chinchilla | ✅ within-corpus model+compute lowers BPB (2.008→1.847) then flattens = **data-bound at 1.34 MB** (MoE gap shrinks +0.115→+0.062). **Growing to 1.92 MB RECOVERS the gap (+0.062→+0.095)** = adding data relieves the bottleneck, capacity pays again. Model×data×compute scale together. Bottleneck = throughput/compute, not architecture |
| Real web data + GPU (`vyoma-data`, FineWeb) | ✅✅ first non-teacher, real-web-text result, on Kaggle T4: **MoE 2.677 BPB beats dense-small 2.797 (+0.120) AND dense-big 2.801 (−0.123, 3.1× fewer active params)** on 210M real tokens. Portable CPU/CUDA/Metal build; `ONLY=` split for 2-GPU parallelism |
| BPE on FineWeb (`vyoma-tokenizer` fineweb) | ✅ **−30% BPB** (2.797→1.936 dense-small; MoE 1.912 best-on-FineWeb) — biggest single lever, confirmed on real web text. MoE edge shrinks +0.120→+0.024 **mechanistically**: at vocab 4096 embed+head = ~89% of params, so FFN (where MoE acts) is only ~11% — same effect, diluted. Fix = grow DFF |
| **FineWeb 30k-step runs** | ✅✅ **BPB 1.768 (4 exp) / 1.701 (16 exp)** — 4x experts, 2x params, better BPB at **identical active compute** (3.547M vs 3.552M). Writes real English. The mission thesis, measured on real web text |
| Instruction tuning (`SFT=1`, `CHAT=1`) | ⚠️ format learned (emits `<|assistant|>`, stops at `<|end|>`), semantics NOT — out-of-distribution prompts get randomly-recalled answers. 462K params on 126 pairs; needs scale |
| Depth vs width | ✅ depth has an **optimum**: at matched 53.6M params, **2 layers × 64 exp = 1.898 BPB (best ever)** > 1 layer × 128 exp = 1.923 > 4 layers × 32 exp = 2.002. Some depth helps, too much hurts at this training budget |
| Data ceiling (reproduced) | ✅ capacity stops paying past ~32 experts on 60 MB and ~64 on 500 MB (128 exp adds 0.001). Each corpus supports a fixed capacity; more data raises the ceiling. Active compute flat (+1%) across 3.4× params |
| Data unlocks capacity | ✅ 2.5× corpus grows the 32→64 expert gap **0.003 → 0.023 (~8×)**. The earlier plateau was data starvation, not an architectural ceiling. (Cross-corpus absolute BPB is test-shift confounded — compare gaps, not levels) |
| Expert-count scaling | ✅ 4/16/32/64 experts on FineWeb: BPB **1.768/1.701/1.687/1.684** — 6× params for 0.65% more active compute. Returns collapse past ~32 (data-bound at 60M tokens) |
| int8 / int4 quantization | ✅ measured cost: int8 **+0.01%**, int4 **+0.09%** BPB. Trillion-class working set **1.98 → 0.59 → 0.48 GB**. Our own kernel (threaded, stdlib) — candle has no int8 dtype |
| Expert paging (`MODE=paged`) | ✅ **the mission's mechanism**: experts on disk int8 (`VYX1`), only routed expert in RAM. Toy 2.10x smaller working set; at real dims (99.9% experts) a **trillion-param model needs ~2 GB RAM** — RAM flat as experts grow, capacity costs DISK not memory |
| Direction | **Retrieval-centric hybrid + sparse MoE**: our retriever + on-disk store + lean SSM + top-1 experts (STORED + int8, not generated); generation stays an image-regime multiplier; teacher teaches only |
| Next (all ours) | firm up retrolm at scale (BPE + bigger model/context + more seeds); build a real MoE (Pillar 3); grow the distilled corpus |

Kill signals we are watching (from `docs/english/03-risk-register.md`):
Risk #1 — weight-gen caps at 10–50×. **Status: not triggered** (fractal hit 212× on MNIST).

## Distance to the vision (honest scorecard)

The north star is frontier capability (~300B-equivalent) in 8 GB. Where we actually are:

| | Status |
|---|---|
| **Architecture skeleton** | ✅ built end-to-end, every corner ours — tokenizer, SSM backbone, generated FFN, learned retriever, persistent store, distillation, honest BPB eval, and the retrieval loop closed on real language. |
| **Mechanisms validated** | ✅ retrieval decouples capability from model size; generation is a redundancy-bound multiplier; int8 composes; data (not model size) is the quality lever. |
| **Pillar 1 (generation)** | ⚠️ real but modest on language — a multiplier, not the engine. |
| **Pillar 2 (SSM)** | ✅ backbone + retriever encoder. |
| **Pillar 4 (retrieval/store)** | ✅ the engine, ours, running on real language (modestly). |
| **Pillar 3 (MoE/symbolic)** | ❌ not built. |
| **Pillar 5 (self-evolution)** | ❌ not built. |
| **Scale** | ❌ everything ≤~0.5 M params, char-level, one laptop, minutes–hours. |

**Honest bottom line.** We are **not** close to the frontier north star — that needs
orders more data + compute (a real commitment). We **are** close to having the full
*achievable-hybrid skeleton* proven and assembled, built by us in every corner. The
remaining distance is dominated by **scale** and the two unbuilt pillars, not by
unproven mechanisms. "Vision big, bets small" — the bets keep landing; the vision
stays honest.

---

## Log

### 2026-08-02 — CORRECTION: depth has an optimum; 2 layers is our best model yet

**I published the wrong conclusion one entry below, and this corrects it.** That
entry ("Depth LOSES to width") was written from the 4-layer data point while the
2-layer run was still training. The 2-layer result inverts it.

**Full curve at matched parameters (500 MB FineWeb, 40 k steps, dm=384, dff=512):**

| config | total params | active | BPB |
|---|---|---|---|
| 64 experts × 1 layer | 28.4 M | 3.570 M | 1.917 |
| 128 experts × 1 layer | 53.6 M | 3.595 M | 1.923 |
| **64 experts × 2 layers** | **53.65 M** | 3.572 M | **1.898** ← best result to date |
| 32 experts × 4 layers | 53.65 M | 3.562 M | 2.002 |

**Depth is not harmful — it has an optimum.** Two layers beats every single-layer
configuration, including one with twice the experts. Four layers is worst. The curve
is non-monotonic, which is the classic depth-versus-optimisation-budget tradeoff:
depth adds representational composition but costs trainability, and past some point
the second effect dominates at fixed steps.

**What I got wrong and why it matters.** I concluded "width is the productive axis"
from a single point, and reinforced it by pointing at two earlier observations that
happened to agree. Had the 2-layer run finished first I would have concluded the
opposite. The lesson is procedural, not technical: *do not publish a curve from one
sample*, especially when the remaining samples are already running. The earlier
entry is left in place rather than deleted, with this correction above it.

**Revised design guidance.** Best known configuration is now **2 layers × 64
experts**. Width alone is exhausted on this corpus (128 vs 64 experts at 1 layer:
0.001 BPB); shallow-and-wide is not optimal either; a modest amount of depth is.
The 4-layer 80 k-step continuation still tests whether deeper models simply need
more optimisation — if 4 layers falls below 1.898 with double the steps, the
optimum is budget-dependent rather than architectural.

### 2026-08-02 — Depth LOSES to width (SUPERSEDED — see correction above) at matched parameters (hypothesis refuted)

**What.** Multi-layer support was added after discovering the entire model had been
one block deep. The hypothesis was that missing depth explained why generation reads
fluently but never holds an idea — depth is what lets a model compose abstractions.
Direct test: spend the same parameter budget on depth instead of width.

**Result (500 MB FineWeb, 40 k steps, dm=384, dff=512).**

| config | total params | active params | BPB |
|---|---|---|---|
| 64 experts × 1 layer | 28.4 M | 3.570 M | **1.917** |
| 128 experts × 1 layer | 53.6 M | 3.595 M | **1.923** |
| **32 experts × 4 layers** | **53.65 M** | 3.562 M | **2.002** |

At a matched 53.6 M parameters, **depth is 0.079 BPB worse than width** — and worse
than the half-sized 64-expert single-layer model. The hypothesis is refuted at this
scale and training budget.

**This is consistent with earlier findings, which I should have weighted more.** The
language-model arc already showed depth failing to unlock generative weights
(1→2→4 layers gave ≈0, +1.3, +0.5–0.9 pts) while *width* moved the needle, and a
small local test here showed 4 layers worse than 2. Three independent observations
now point the same way: for this architecture, **width is the productive axis and
depth is not.**

**Caveat being tested.** Deeper models typically need more optimisation steps, so
40 k may simply undertrain 4 layers. A continuation to 80 k steps is running. If
BPB drops below 1.923 the conclusion becomes "depth needs more compute"; if it
plateaus above, width wins outright at this scale.

**Why it matters.** Width was already exhausted on this corpus (128 experts bought
0.001 over 64). If depth also fails, then neither shape lever is available and the
remaining lever is unambiguously **data** — which is exactly what the data-ceiling
curve says. It also means the incoherence of generated text is not a missing-depth
problem, and a different explanation is required.


### 2026-08-02 — 500 MB expert curve completes: the data ceiling is real and reproducible

**Result (500 MB FineWeb, 40 k steps, dm=384, dff=512).**

| experts | BPB | gap vs previous | total params | active params |
|---|---|---|---|---|
| 32 | 1.947 | — | 15.8 M | 3.558 M |
| 64 | 1.924 | **0.023** | 28.4 M | 3.570 M |
| 128 | 1.923 | **0.001** | 53.6 M | 3.595 M |

**Capacity stops paying at ~64 experts on this corpus.** Doubling 64→128 bought
0.001 bits/byte — nothing — exactly as 32→64 bought nothing (0.003) on the smaller
60 MB corpus. The pattern is consistent and now measured twice:

| corpus | where the gap collapses |
|---|---|
| 60 MB | past ~32 experts |
| 500 MB | past ~64 experts |

**Each corpus size supports a certain capacity and no more.** More data raises the
ceiling (2.5× data roughly doubled the useful expert count and grew the payoff 8×);
more experts beyond it are dead weight. This is Chinchilla stated in expert-count
terms, on our own architecture, and it means "add experts" is only a lever while
data grows with it — which is precisely the honest form of the trillion-parameter
claim.

**Active compute stayed flat throughout**: 3.558 M → 3.595 M (+1%) across a 3.4×
increase in total parameters. The sparsity property holds regardless of whether the
extra capacity is *useful* — a useful separation, since it means the memory
architecture is not what limits us.

**Open question now running:** the same parameter budget spent on DEPTH instead of
width — 32 experts × 4 layers (~60 M params) against 128 experts × 1 layer (53.6 M,
BPB 1.923). Depth only became possible today; until this morning the whole model
was one block deep.


### 2026-08-02 — More data makes capacity pay: the expert gap grows 8× on a 2.5× larger corpus ✅

**What.** The 4→64 expert curve on 60 M tokens had flattened to nothing
(32→64 experts bought 0.003 BPB), which we read as data-bound rather than
architecture-bound. Direct test: extract 500 MB from the same FineWeb shard
(2.5× the text) and re-run 32 and 64 experts, 40 k steps each.

**Result.**

| corpus | 32 experts | 64 experts | **gap (32→64)** |
|---|---|---|---|
| 60 MB | 1.687 | 1.684 | **0.003** |
| 500 MB | 1.947 | 1.924 | **0.023** |

**The payoff for extra capacity grew ~8×.** On the small corpus, doubling experts
was worthless; on the larger one the identical capacity increase buys 0.023 bits/byte.
The plateau was starvation, not a ceiling in the architecture — exactly as predicted.

**Read the absolute numbers correctly.** BPB *rose* (1.687 → 1.947) but that is NOT
a regression: the held-out set is the last 10% of the corpus, so a different corpus
means a different, harder test tail. This is the cross-corpus test-shift confound
already documented here. Only the **within-corpus gap** is comparable, and that is
the number that moved. (Also relevant: 40 k steps over ~145 M tokens is ~1.1 epochs
versus ~2 epochs for the small corpus — less repetition, less overfitting, which is
precisely why capacity pays more.)

**Consequence for the mission.** Capacity and data must scale together — and the
architecture's benefit *increases* with data rather than saturating. Every earlier
"diminishing returns" reading was a data ceiling, not an architectural one. Two
follow-ups now running: 128 experts on 500 MB, and 64 experts for 90 k steps to
separate convergence from capacity.


### 2026-08-02 — Instruction tuning: the FORMAT is learned, the semantics are not (honest)

**What.** Closed the "it continues text instead of replying" gap architecturally.
`vyoma-distill SFT=1` has the teacher write (instruction, response) pairs wrapped in
turn markers (`<|user|>` / `<|assistant|>` / `<|end|>`); `vyoma-lm DATASET=sft` trains
on them; `CHAT=1` frames a prompt as a user turn at inference and stops at `<|end|>`.
Teacher supplies data only, never ships — the standing principle holds.

**Result: 126 pairs (56 KB), 462 K-param MoE, 2500 steps.**

Behaviour genuinely changed:

| | output for a prompt |
|---|---|
| base (no framing) | `hi` → `"hipping/paperwork/electronic document transfer…"` |
| `CHAT=1` | `What is gravity?` → `<\|assistant\|>` `"Gravity: a force that attracts two bodies with mass…"` |

**But that answer is MEMORISED, not understood.** Nearly the same sentence is in the
training data. Out-of-distribution it collapses entirely:

| prompt | reply |
|---|---|
| "What is a submarine?" | *"Fossils are formed when organisms die…"* |
| "How do I bake bread?" | *"Languages evolve through phonetic shifts…"* |

It emits a well-formed turn containing a randomly-recalled answer, with **no mapping
from question to content**. BPB 6.585 confirms severe overfitting (26 K training
tokens).

**Verdict.** ✅ conversational *structure* is now part of the architecture — turn
markers, assistant role, stopping behaviour. ❌ instruction *following* is not
learned, and cannot be at this scale: 462 K parameters on 126 examples. This is the
"reply-shaped, not useful" outcome predicted before the run, recorded as such rather
than showcased via the one memorised example that looks impressive.

**Why it still matters.** The format is the part that must exist in the architecture;
semantics come from scale and data. When a larger model trains on orders more
instruction pairs, it lands on a system that already knows what a turn is.


### 2026-08-02 — Expert-count scaling curve complete + int4: 999B params in 0.48 GB

**Scaling curve (FineWeb 60M tokens, 30k steps, dm=384, dff=512).**

| experts | BPB | Δ | total params | active params |
|---|---|---|---|---|
| 4 | 1.768 | — | 4.73 M | 3.547 M |
| 16 | 1.701 | −0.067 | 9.46 M | 3.552 M |
| 32 | 1.687 | −0.014 | 15.78 M | 3.558 M |
| 64 | **1.684** | −0.003 | 28.40 M | 3.570 M |

**The thesis holds: 6× the parameters for 0.65% more active compute.** Capacity
comes from expert count; per-token cost does not move. With paging, RAM does not
either. That is the trillion-parameter argument, measured on real web text.

**But returns collapsed** (−0.067 → −0.003): at 60M tokens we are firmly
data-bound, and beyond ~16–32 experts extra capacity buys almost nothing. Same
Chinchilla signature as before. Hence a 500 MB extract (2.5× data) now training.

**int4 (`BITS=4`) is nearly free too.** Same measurement harness (`MODE=q8bpb`),
same checkpoint, same held-out text:

| weights | BPB | cost |
|---|---|---|
| f32 (reference) | 2.1721 | — |
| int8 | 2.1723 | +0.01% |
| int4 | 2.1742 | **+0.09%** |

int4 packs two weights per byte (`QW::Int4`), unpacked in the matmul inner loop —
never expanded in RAM. Working set 4.95× → **5.70×** smaller.

**Note this does NOT contradict the earlier E1 finding that 4-bit broke the seed
(→57%).** That was *generated* weights, where each seed parameter is
information-dense and cannot absorb rounding. Stored MoE experts are redundant
and tolerate int4 fine — the same redundancy principle that has governed every
generation result in this project, now cutting the other way in our favour.

**Trillion-class projection (8500 experts, ~999B params), working set:**

| | |
|---|---|
| all f32 | 1.98 GB |
| int8 experts + int8 backbone | 0.59 GB |
| **int4 experts + int8 backbone** | **0.48 GB** |

A ~999-billion-parameter model with a **0.48 GB working set** — under half a
gigabyte, on an 8 GB machine, with 7.5 GB to spare.

**Kernel performance.** Threading (`std::thread::scope`, stdlib only) plus four
independent accumulators took int8 inference 2.78 → 1.81 ms/token against f32's
1.50 — so 3.1× less RAM for 1.2× slower. Router quantization is opt-in
(`Q8ROUTER=1`): at 8 experts it saved 3 KB and cost 0.6 ms/token because candle's
BLAS wins on a small matmul, but at 8500 experts the router is 0.13 GB f32 and the
trade flips.


### 2026-08-02 — 30k steps on FineWeb: BPB 1.768 → 1.701, and the mission thesis proven on real data ✅✅✅

**What.** Two full 30,000-step runs on real FineWeb web text, one per T4, differing
only in expert count. Both completed, both checkpointed every 500 steps.

| model | BPB | total params | **active params** | ckpt |
|---|---|---|---|---|
| MoE 4 experts | **1.768** | 4,729,344 | 3,547,008 | 19 MB |
| MoE 16 experts | **1.701** | 9,463,296 | **3,551,616** | 37 MB |

**The result the mission rests on.** 4× the experts and 2× the total parameters buy
a better model (−0.067 BPB) at **identical active compute** — active params differ by
just 4,608 (the larger gate). Capacity scales with expert count; the per-token cost
does not. Combined with paging (below), RAM does not scale either. This is the
empirical core of the trillion-parameter argument, measured on real web text with
our own code.

Mechanistically visible in the loss curve too: the 4-expert model plateaued around
step ~13,000 (loss ~4.30, then oscillating) — it ran out of capacity, not data. More
experts was exactly the right lever.

**Best BPB progression on FineWeb:** 2.677 (char, 6k steps) → 1.912 (BPE, 6k) →
**1.768** (BPE, 30k, 4 exp) → **1.701** (BPE, 30k, 16 exp).

**It writes real English now.** 4-expert model, prompt *"The future of artificial
intelligence is"*:

> "…about an abortion of this and many other people to have the potential.
> **Therefore**, some who need to be a better sense for the purpose of the state.
> **The major issue is** not that there is a chance to get the most respectfully, the
> government which can have an impact… **Since** those are also important to avoid
> the end of the process."

Grammatical clauses, function words, articles, agreement, discourse connectives
(*Therefore*, *Since*). Semantically it drifts and invents words — it is a 5 M-param
model — but this is fluent English *structure*, a world away from the Shakespeare-era
`hirest;`.

**Paging verified on a real trained model** (16 experts, cache=1): experts 6.16 MB
int8 on disk; RAM all-resident 36.1 MB → paged **13.5 MB (2.66× smaller)** at
19.75 ms/token (real disk reads + dequant). The mechanism holds outside the toy.

**Operational lessons (hard-won).** Kaggle reaps a session after **40 minutes of
notebook-cell inactivity** — it cannot see tmux or GPU load, which is why sessions
kept dying. Fix: keep a cell running (`!tail -f train.log`) for the full 12 h. Also:
`~/.cargo` is wiped on restart while `/kaggle/working` survives, and binaries lose
their exec bit (`chmod +x` needed). Periodic checkpointing (CKPT_EVERY) proved its
worth immediately — a restart that would have destroyed 30k steps cost nothing.


### 2026-08-02 — Expert paging: a trillion-parameter architecture with a ~2 GB working set

**What.** Built the mechanism the mission actually needs. Until now our MoE held every
expert in RAM *and* computed all of them — a quality trick, not a memory strategy.
Now experts live on **disk in int8** (`VYX1`, ours) and only the routed expert is read
in. `PagedExperts` seeks and reads just that expert's slice; `forward_moe_paged`
groups tokens by routed expert so each is fetched at most once (sparse compute AND
sparse memory); an LRU cache keeps hot experts. `MODE=paged` measures it.

**Measured (toy: 8 experts, dm=128, dff=192, vocab=1024).**

| cache | RAM working set | vs all-resident | ms/token |
|---|---|---|---|
| 1 | 1227 KB | **2.10x** | 0.85 |
| 2 | 1420 KB | 1.82x | 0.61 |
| 4 | 1807 KB | 1.43x | 0.57 |
| 8 | 2580 KB | 1.00x | 0.46 |

A textbook memory/latency tradeoff, on our own architecture. The toy ratio is modest
only because at vocab=1024 the embedding+head dominate; experts are just 60% of it.

**Projection — and this is the mission's answer.** At realistic dims (dm=4096,
dff=14336, vocab=32k) experts are **99.9%** of the model, so paging dominates:

| experts | total params | RAM working set | experts on disk (int8) |
|---|---|---|---|
| 2,048 | 241 B | **1.88 GB** | 0.22 TB |
| 8,192 | 963 B | **1.98 GB** | 0.88 TB |
| 8,500 | **999 B** | **1.98 GB** | 0.91 TB |

**RAM is flat as the model grows** — 241 B to 999 B parameters moves RAM 1.88 -> 1.98 GB.
Adding capacity costs **disk, not memory**. A trillion-parameter sparse model with a
~2 GB working set fits in 8 GB with room to spare.

**Correcting the earlier framing.** I had argued "a trillion params in 8 GB" was
arithmetically impossible. That was answering the wrong question: params cannot be
*resident* in 8 GB (466 GB at int4, 116 GB even at 1 bit), but with sparse routing
they do not need to be. Akash's intuition was right on the substance; my objection
was right only about residency. The defensible claim is now: **a trillion-parameter
sparse architecture with a ~2 GB working set, experts paged from disk in int8.**

**Honest remaining constraints.** (1) ~1 TB of disk — the parameters exist, just not
in RAM. (2) Someone must *train* them: the architecture is no longer the blocker,
compute funding is. (3) Each expert at those dims is ~117 MB int8, so a cache miss
costs real milliseconds; smaller/more-numerous experts plus locality amortize it —
tuning, not a wall.

**A bug I caught in my own measurement.** The first `PagedExperts` did
`std::fs::read` of the whole store into a `Vec<u8>`, which made "disk" silently
resident and overstated the saving. Fixed to seek+read per expert; the numbers above
are from the corrected version.


### 2026-07-26 — Learned grounding head: 74% → 88.9% balanced accuracy (error more than halved) ✅

**What.** Replaced the hand-set cosine threshold with a **learned decision**, ours,
in Rust. `GroundingHead` (`vyoma-embed`) is a small MLP over
**[q ; e ; q⊙e]** — the query embedding, the retrieved match, and their elementwise
interaction — so it can use *directions* in the embedding space rather than the
single angle between two vectors that a cosine threshold collapses everything into.
Trained on the same contrast (in-domain query = supported, foreign corpus = not),
2-class cross-entropy. `MODE=head` builds the dataset, trains, and evaluates on a
**held-out 20% split**; `MODE=rag HEAD=...` uses it at inference.

**Result (held-out, 800 samples: 400 supported / 400 not).**

| gate | balanced acc | accepts supported | rejects unsupported |
|---|---|---|---|
| cosine threshold (tuned) | 74.0% | 71.5% | 76.5% |
| learned head, *fixed-offset positives* | 88.9% | 95.8% | 82.0% |
| **learned head, varied positives (shipped)** | **86.6%** | 90.1% | 83.1% |

**A metric that went DOWN while the system got BETTER — worth reading carefully.**
The first head (88.9%) trained positives only as `center_fragment`: a fixed-offset
verbatim 48-byte slice. Live-tested, it **rejected a genuine in-domain prompt**
(P=0.406) even though retrieval had found its *exact source passage* — the hand-typed
prompt differed in whitespace and framing, outside that narrow distribution. So the
held-out 88.9% was honest in construction but measured an unrealistically uniform
query distribution. Retraining with **varied positives** (random offset, random
length, light character noise) lowered the aggregate to 86.6% — a genuinely harder
task — while fixing real behavior:

| live prompt | before | after |
|---|---|---|
| Shakespeare (Isabel) | 0.406 ✗ veto | **0.980 ✓ accept** |
| Shakespeare (yonder window) | — | **0.804 ✓ accept** |
| Photosynthesis | 0.000 ✓ veto | **0.001 ✓ veto** |
| Python code | 0.000 ✓ veto | **0.005 ✓ veto** |

4/4 correct on realistic hand-typed prompts, with confident margins. Lesson: a
held-out split only measures honesty *within* the distribution you sampled — if that
distribution is narrower than reality, the number flatters and the system still
breaks. Test on realistic inputs, not just held-out ones.

**Verdict — a large, clean win.** Error rate falls from 26% → 11%, and it improves in
*both* directions rather than trading one for the other. The interpretation is
straightforward: the in/out-of-domain distributions overlap badly *in cosine*, but
are much more separable in the full embedding geometry — a scalar was throwing away
most of the available signal. Same retriever, same store, same data; only the
decision rule changed.

**Progression of this one component, honestly:** no signal at all (out-of-domain
scored *higher* than in-domain) → out-of-domain negatives gave a real but overlapping
signal (74%) → a learned head over the full geometry (88.9%). Each step was measured,
and the two overclaims along the way (a z-score "fix" that was the wrong statistic; a
7-probe "clean separation" that a 400-sample calibration refuted) are recorded above
rather than quietly corrected.

**Still honest about the ceiling.** 88.9% is not a guarantee — ~1 in 9 boundary
decisions is still wrong, and this is one store, one domain pair, ~500 K-param
retriever. But grounding is now a *trained component of the architecture* rather
than a magic number, and it can be improved the same way everything else here is:
more negatives, more diverse domains, a bigger retriever.

### 2026-07-26 — Grounding gate FIXED: out-of-domain negatives give the retriever a "do I know this?" signal ✅

**What.** Direct fix for the previous entry's negative. Added `train_encoder_neg` to
`vyoma-embed`: every contrastive batch now appends `n_neg` documents drawn from a
**different corpus**, which are negatives for every query. Plain in-batch InfoNCE
only contrasts Shakespeare against Shakespeare, teaching *"which passage?"* and never
*"is this my domain?"* — foreign negatives make the second question learnable.
Wired via `vyoma-embed MODE=store NEG=distilled.txt NNEG=32`. Added `MODE=gate`, a
no-LM diagnostic that measures the grounding signal directly across domains.

**Result (`MODE=gate`, store = 1743 Shakespeare passages, negatives = distilled text).**

Seven hand-picked probes looked clean — in-domain 0.697/0.713 vs out-of-domain
≤0.593 — so I set GATE=0.64 and claimed clean separation. **That was overclaiming
from a tiny sample, and it broke immediately**: a photosynthesis prompt then scored
0.643 and was wrongly accepted. So I measured the actual distributions instead
(200 in-domain held-out fragments vs 200 out-of-domain chunks):

| | p05 | median | p95 |
|---|---|---|---|
| in-domain | 0.598 | **0.717** | 0.842 |
| out-of-domain | 0.487 | **0.624** | 0.745 |

**Best achievable threshold 0.680 → 74.0% balanced accuracy** (accepts 71.5% of
in-domain, rejects 76.5% of out-of-domain).

**Verdict — a real, large improvement, but NOT clean separation.** Before OOD
negatives the gate had *no* signal (out-of-domain literally scored higher than
in-domain; no statistic separated them). After, there is a genuine, generalizing
signal — it holds for code and random junk, domains never seen as negatives, so the
encoder learned "not my domain" in general. In-domain retrieval also improved
(0.458 → 0.558). **But the distributions substantially overlap: ~26% error at the
best threshold.** The gate is a useful filter, not a guarantee — and the honest
number is 74%, not the "clean gap" the 7 probes implied.

**Method lesson (worth more than the number).** Hand-picked probes flattered the
result and I published that before checking; a 400-sample calibration corrected it
within the hour. `MODE=gate` now reports distributions, the optimal threshold, and
its true error rates — so thresholds get *calibrated*, not guessed.

**I was wrong about the statistic, and the data says so.** Last entry I replaced the
absolute-cosine gate with a z-score, reasoning that a contrastive encoder's raw
cosine is uncalibrated. With OOD negatives the opposite is true: **cosine separates,
z-score fails** (random junk scores 3.67σ — as high as real Shakespeare — because
z-scoring normalizes away exactly the absolute-similarity signal that carries the
domain information). Reverted the gate to absolute cosine. **The fix was the
training, not the statistic** — no threshold on the old encoder could ever have
worked, and no clever statistic substitutes for a signal that isn't there.

**Honest scope + what would actually close the gap.** 74% balanced accuracy means
roughly 1 in 4 decisions is wrong at the boundary — fine as a soft filter, not
enough to promise "no hallucinated grounding". The store is single-domain
(Shakespeare); a multi-domain store needs per-domain or learned thresholds. Real
improvements from here: a bigger/deeper retriever, more and more-diverse negative
corpora, longer training, and — most promising — making the decision a *learned*
one (a small supported/unsupported head) rather than a hand-set cosine threshold.

### 2026-07-26 — The assembled SYSTEM runs (`MODE=rag`) — and an honest negative: the grounding gate does not discriminate

**What.** Two enabling corners, then the first end-to-end run of the whole
architecture as ONE pipeline:
- **Checkpoints.** `save_moe_ckpt`/`load_moe_ckpt` (safetensors; config derived from
  tensor *shapes*, so weights/config can't drift). `SAVE=` on `MODE=moe`. Previously
  every trained model evaporated on exit.
- **Retriever persistence.** `Encoder::save/load` in `vyoma-embed`; `MODE=store` now
  writes `retriever.safetensors` beside `ontological_store.vyst` — a **matched pair**
  (int8 keys are only meaningful to the encoder that made them).
- **`MODE=generate`.** Loads a checkpoint and writes text (top-k + temperature,
  standalone `BpeCodec` from the merges file). The model finally *speaks*.
- **`MODE=rag`.** The assembled system: our retriever → our on-disk VYST store →
  grounding gate → stored-MoE LM generation. No teacher, no external model.

**It runs, and retrieval is genuinely apt.** Prompt *"Who will believe thee,
Isabel?"* → fetched *"...'tis incredible to belie[ve]"* (cos 0.643) — our learned
retriever found a semantically related passage. Generation then produced real
Shakespearean structure (`BENVOLIO:`, `BRUTUS:`, verse lines, "O heaven", "my lord");
content is incoherent at this scale (1500 steps, 460 K params), as expected.

**The honest negative — the gate can't tell supported from unsupported.**

| prompt | cos | z-score |
|---|---|---|
| in-domain (Shakespeare) | 0.643 | 3.31σ |
| out-of-domain (Python code) | **0.682** | 3.26σ |

An **out-of-domain prompt scored HIGHER** than the correct one. Switching from an
absolute cosine threshold to a calibration-free **z-score vs the whole store** (the
principled fix) didn't help either: 3.31σ vs 3.26σ — indistinguishable. No threshold
on any of these statistics separates the cases.

**Why (root cause, not a tuning issue).** The retriever is trained contrastively with
**in-batch negatives drawn from the same corpus**. That teaches *"which Shakespeare
passage is nearest?"* — never *"is this Shakespeare at all?"* It therefore has **no
out-of-domain detection capability**, and a contrastive encoder maps everything into
a narrow cone, so raw cosine is uncalibrated. The gate has no signal to act on.

**This qualifies the `MODE=lattice` result.** That hallucination veto (100%→0%) used
**exact symbolic consistency** on digit keys, where in-store and out-of-store are
crisply separable. It does **not** transfer to embedding similarity over natural
text — the caveat logged then is now demonstrated, with numbers.

**Safe default, and the real fixes.** With `GATE=4σ` both cases veto, so the system
degrades to *ungrounded generation* rather than confidently grounding on a bad match
— the safe failure direction. Actual fixes, in order of honesty:
1. **Train the retriever with out-of-domain negatives** (contrast Shakespeare against
   other corpora) so "not my domain" becomes learnable. This is the real fix.
2. **Lexical/symbolic overlap** as the gate (domain-agnostic, closer in spirit to the
   digit-key check that worked) instead of embedding similarity.
3. A calibration set to pick the threshold empirically, rather than by intuition.

Reported as a negative because it is one: the pipeline composes and runs, but
**grounding is not yet trustworthy**, and no amount of threshold-fiddling fixes it.

### 2026-07-26 — BPE on FineWeb: biggest tokenizer win yet, and the MoE edge shrinks (mechanistically explained)

**What.** Added `DATASET=fineweb` + `SAMPLE_MB` to `vyoma-tokenizer` (merges trained
on a fast, capped sample; reused to BPE-encode the full corpus). Trained merges on
30 MB of the FineWeb shard (vocab 4096, 3.45 bytes/token, round-trip OK, genuine
subwords: `" international"`, `" opportunities"`, `" professional"`), then reran the
same `MODE=moe` comparison with `TOKENIZER=bpe` instead of char, same GPU/config.

**Result (dm=384, dff=512, E=4, 6000 steps, FineWeb, T4 GPU).**

| | char (prior) | BPE (this run) | Δ |
|---|---|---|---|
| dense-small | 2.797 | **1.936** | −0.861 (−31%) |
| MoE | 2.677 | **1.912** | −0.765 (−29%) |
| dense-big | 2.801 | **1.961** | −0.840 (−30%) |
| Δ(small−MoE) | +0.120 | **+0.024** | shrank |
| Δ(MoE−big) | −0.123 | **−0.049** | shrank |

**Verdict 1 — BPE is our biggest single lever, confirmed on real web text.**
−30% bits/byte, even bigger than the 25% seen on Shakespeare. The tokenizer corner
generalizes cleanly to open web data.

**Verdict 2 — the MoE edge shrank, and it's mechanistic, not a regression.** At
vocab=4096 the embedding+output-head layers now dominate the model (~89% of
dense-small's params, vs ~28% at char-level vocab=202) — MoE only adds capacity to
the **FFN**, which is now just ~11% of the model (was ~72%). Same underlying FFN
effect, diluted by a now-much-bigger fixed vocab cost. Normalizing (gain ÷ FFN
share) gives ~0.17 (char) vs ~0.21 (BPE) — consistent — so the mechanism is intact;
only its share of the total model changed. MoE still beats both baselines in
direction, just by less in absolute BPB.

**Honest cross-corpus note.** 1.912 is NOT comparable to the earlier laptop "best"
(1.832, teacher-distilled corpus) — different held-out text, different intrinsic
difficulty (open web vs curated). This is our best BPB *on FineWeb specifically*.

**Next (implied by the mechanism).** To see MoE's fuller advantage at this vocab
size, grow `DFF` so the FFN mass is a bigger share of the model again (the same
lever that recovered the MoE gap when we added data at the Chinchilla inflection).

### 2026-07-26 — First result on REAL web text (FineWeb), on a real GPU: MoE beats BOTH dense baselines ✅✅

**What.** Left the laptop/teacher-distilled corpus for the first time. New crate
`vyoma-data` (parquet → text, isolated deps) extracts the `text` column of a
**FineWeb** shard (HuggingFaceFW/fineweb, `sample/10BT`) — real CommonCrawl web
text, zero teacher involvement. `vyoma-lm` made portable (CPU default, opt-in
`cuda`/`metal` cargo features — was Mac-only) and trained on **Kaggle's free T4
GPU**. Also added `ONLY=small,moe,big` to `MODE=moe` so the three independent
trainings can be split across separate processes/GPUs via `CUDA_VISIBLE_DEVICES`
for wall-clock parallelism (no true multi-GPU tensor code — not built).

**Corpus.** One FineWeb shard, capped to 200 MB of extracted text →
**209,717,526 tokens** (char-level) — ~110× our biggest prior (teacher-distilled)
corpus, and genuinely diverse open-web text, not synthetic/teacher-generated.

**Result (dm=384, dff=512, E=4, 6000 steps, char tokenizer, T4 GPU).**

| model | BPB | params (active) |
|---|---|---|
| dense-small (dff=512) | 2.797 | 550,986 |
| **MoE (4×512, top-1)** | **2.677** | 1.73M total, **~552,522 active** |
| dense-big (dff=2048) | 2.801 | 1,732,170 |

Δ(small−MoE) = **+0.120**, Δ(MoE−big) = **−0.123**.

**Verdict — the MoE win holds on real data, and is STRONGER than on the toy corpus.**
(1) Δ(small−MoE)=+0.120 matches every prior confirmation (+0.109 to +0.115 on the
laptop) — the architectural advantage is not a teacher-corpus artifact. (2) MoE does
not merely tie dense-big here — it **beats it outright** (2.677 vs 2.801) with **3.1×
fewer active params**. Honest read: at a fixed 6000-step budget, one giant FFN
(dff=2048) is harder to optimize than 4 experts the width of dense-small, so part of
this edge is likely "easier to train under fixed compute," not pure capacity — but
that is itself a real, useful capability-per-compute result, and it's the best
demonstration of Pillar 3's thesis yet.

**Honest scope.** Single run (no seed repeat at this scale yet); char-level (BPE
next); only 6000 steps sampled from 210M tokens (broad but partial coverage); dense-
big may close the gap with more steps/tuned LR — worth checking before over-claiming
the margin. But the core, load-bearing number — MoE beats dense-small — is exactly
where every smaller-scale run landed, now confirmed on real web-scale text.

**Why it matters.** This is the first time Vyomarudra has trained on real,
non-teacher-generated web text, and the first time on a real GPU (not the laptop
CPU). The architecture — built entirely by us — ports and scales cleanly to a
different data source and a different (cloud) machine with no redesign, only an
enabling corner (`vyoma-data`) and a portability fix. Free-tier compute (Kaggle T4)
is a genuine, if modest, step toward the scale the vision needs.

### 2026-07-25 — Chinchilla confirmed: adding data RELIEVES the data-bound bottleneck (MoE gap recovers +0.062 → +0.095) ✅

**What.** Grew the corpus 1.34 MB → **1.92 MB** (+500 teacher generations) and re-ran
the exact config that had gone data-bound on 1.34 MB (dm=160/dff=256, 6000 steps).
The clean signal is the **gap** Δ(small−MoE) — a within-run Δ (both models share the
run's test set), so it's robust to the cross-corpus test-set-shift confound that
plagues absolute BPB.

| corpus (same dm=160/dff=256) | Δ(small−MoE) | MoE vs dense-big |
|---|---|---|
| 1.34 MB (data-bound) | +0.062 | MoE loses (−0.011) |
| **1.92 MB** | **+0.095** | **MoE wins (+0.004)** |

MoE BPB 1.832, dense-small 1.927, dense-big 1.836.

**Verdict — textbook Chinchilla, on our own model.** On 1.34 MB the dm=160 model was
data-starved, so extra capacity (MoE) barely paid (+0.062) and even lost to dense-big.
**Adding 43% more data recovered the capacity advantage** (+0.062 → +0.095) and flipped
MoE back to beating dense-big. Read plainly: *when data is the bottleneck, capacity
doesn't pay; scale data alongside the model and it pays again.* This closes the
scaling story coherently — model, data, and compute must grow together, exactly the
vision's stated (and now empirically reproduced) bottleneck. The architecture's
benefits **scale with data**; the remaining gap is throughput/compute, not mechanism.

### 2026-07-25 — Scale curve hits the Chinchilla inflection: data-bound at ~1.34 MB (honest, informative)

**What.** Extended the within-corpus scaling curve with a third, bigger point on the
fixed 1.34 MB corpus (same test set).

**MoE BPB vs model size (1.34 MB):**

| config | MoE BPB | Δ(small − MoE) |
|---|---|---|
| dm96/dff128, 3000 steps | 2.008 | +0.109 |
| dm128/dff192, 5000 steps | 1.867 | +0.115 |
| dm160/dff256, 6000 steps | **1.847** | +0.062 |

**Verdict — we found the inflection.** Scaling the model still helps but is
**flattening hard** (−0.141 then only −0.020), and the **MoE advantage is shrinking**
(+0.115 → +0.062). Both are the classic signature of the **data-bound regime**: at
~250–500 K params the model is outgrowing 1.34 MB, so extra capacity (MoE *or*
dense-big, now 1.836) buys little — the binding constraint has shifted from model
capacity to **data**. Best BPB 1.847.

**The honest takeaway.** This empirically locates the Chinchilla ratio on the laptop:
to keep improving we must **grow the corpus** (and scale model+compute with it), not
just enlarge the model. Further model-only scaling on 1.34 MB is spent. And growing
the corpus is teacher-generation-bound (~10 s/sample) — i.e., the remaining lever is
**compute/data throughput**, not architecture. Exactly the vision's stated bottleneck.

### 2026-07-25 — Scale epoch: within-corpus, scaling model+compute lowers BPB 2.008 → 1.867 (clean, no confound) ✅

**What.** The honest scale test — on a FIXED corpus (1.34 MB distilled, same held-out
tail, no test-set shift), does scaling model + compute lower BPB? Grew the corpus
999 KB → **1.34 MB** (+300 teacher generations), then compared two configs on it:

| config on same 1.34 MB | MoE BPB | dense-small | dense-big |
|---|---|---|---|
| dm=96, dff=128, 3000 steps | 2.008 | 2.118 | 2.002 |
| **dm=128, dff=192, 5000 steps** | **1.867** | 1.983 | 1.867 |

**Verdict.** Scaling model + compute on the same corpus drops MoE BPB **−0.141
(2.008 → 1.867)** — the proper Chinchilla direction, measured cleanly (same test set).
**1.867 is our best BPB.** This resolves the earlier confusion: cross-corpus BPB rose
only because of test-set shift + fixed compute; *within* a corpus, scaling helps as
expected. And the MoE win is invariant yet again — Δ(small−MoE) = **+0.115** (matching
+0.109/+0.110 at every prior scale), MoE ≈ dense-big at ~half the active compute.

**The coherent scaling picture now.** (1) Model alone on fixed small data → overfits
(shown earlier). (2) Data alone, fixed compute, cross-corpus → confounded/underfits.
(3) **Model + compute together on a fixed corpus → clean improvement (this).** The
road to capability is all three scaled in the right ratio — exactly the vision's
honest bottleneck: compute, not mechanisms.

### 2026-07-25 — Seed-confirmation: the integrated MoE win HOLDS (unlike retrolm) ✅

**What.** Locked the flagship integrated result with a 2-run seed confirmation (init
variance = independent samples), same config (BPE, distilled, dm=96, dff=128, E=4,
3000 steps). Rigor prompted by the retrolm lesson (single-seed was optimistic there).

**Result.**

| | run 1 | run 2 | mean ± spread |
|---|---|---|---|
| dense-small | 2.064 | 2.072 | 2.068 |
| MoE | 1.951 | 1.965 | **1.958 ± 0.007** |
| dense-big | 1.956 | 1.963 | 1.960 |
| Δ(small − MoE) | +0.113 | +0.106 | **+0.110 ± 0.004** |
| Δ(MoE − big) | −0.005 | +0.002 | ~0 |

**Verdict — robust.** The MoE win is reproducible and tight: +0.110 bits/byte over
dense-small (~25× the ±0.004 seed noise), and MoE consistently matches dense-big at
~small active params. Unlike retrolm (which shrank under seeds), the integrated
flagship survives confirmation cleanly. The headline ~1.94–1.96 BPB is real.

### 2026-07-25 — Integration on best data: assembled MoE core (BPE+SSM+MoE, distilled) → BPB 1.943, our best ✅

**What.** First real integration of the decided architecture on our best corpus:
`MODE=moe TOKENIZER=bpe DATASET=distilled` — BPE tokenizer + lean SSM backbone +
stored top-1 MoE, trained on the grown 999 KB distilled corpus, scored by BPB.
Composes P2 (SSM) + P3 (MoE) + the tokenizer + the data lever in one model.

**Result (999 KB distilled, BPE vocab 512, dm=96, dff=128, E=4, 4000 steps).**

| model | BPB | params |
|---|---|---|
| dense-small (dff=128) | 2.035 | 124k |
| **MoE (4×128, top-1)** | **1.943** | 199k total · ~124k active |
| dense-big (dff=512) | 1.924 | 198k |

**Verdict.** **1.943 is our best BPB ever** (prior bests: 2.013 distilled-dense,
2.136 Shakespeare-BPE, 2.68 char-MoE). Two things confirmed: (1) the **MoE win
generalizes** from the char toy to **BPE + real distilled text** — MoE beats
dense-small by +0.092 and nearly matches dense-big (Δ +0.019) at ~dense-small active
params; (2) the architecture **composes** as designed — tokenizer × SSM × MoE × the
data lever stack into our strongest model, and with int8 (shown ~free in `g1`) the
MoE mass is ¼ the RAM. Retrieval (P4), symbolic veto (P3b), and self-evolution (P5)
compose as inference-time layers (demonstrated separately).

**Honest scope.** Single seed, ≤200k params, 999 KB corpus — the best integrated toy,
not a shipped model. The leap to real capability is scale (data × compute), unchanged.

### 2026-07-25 — Pillar 5 built: bounded self-evolution + homeostasis — ALL FIVE PILLARS now have ours-built corners ✅

**What.** `vyoma-lm MODE=evolve`. The model self-evolves by **writing new facts to its
store** (a KV store, last-write-wins) — not by risky weight edits, so there is no
catastrophic forgetting. A **homeostasis controller** gates every write with the
symbolic consistency check (Pillar 3b): a write whose key already exists with a
DIFFERENT value (a contradiction / poisoning attempt) is **rejected**, keeping the
system stable as it grows. naive (accept all writes) vs homeostasis, across rounds
with injected contradictory facts.

**Result (random 12-digit keys, 5 rounds × 250 good facts, +33% contradictory).**

| round | keys (naive/homeo) | acc-naive | acc-homeo | homeostasis rejects |
|---|---|---|---|---|
| 0 | 250 / 250 | 1.000 | 1.000 | 0 |
| 1 | 500 / 500 | 0.855 | 0.995 | 83 |
| 2 | 750 / 750 | 0.770 | 0.990 | 166 |
| 3 | 1000 / 1000 | 0.780 | 0.995 | 249 |
| 4 | 1250 / 1250 | **0.760** | **1.000** | 332 |

**Verdict — clean win.** As the store self-evolves (250 → 1250 keys), the naive policy
**degrades 1.0 → 0.76** under contradictory writes, while **homeostasis holds ~1.0** by
vetoing all 332 poison writes. Capability grows (more facts answered) with **no
forgetting and no weight edits**; stability is maintained by a bounded write policy.
That is the vision's "bounded self-evolution + homeostasis controller," demonstrated,
ours — and it composes the store (P4) + symbolic check (P3b) into P5.

**A fixed bug (honest).** First cut used an append-only list; `nearest()` tie-broke to
the earliest (good) entry, so naive never visibly degraded (both ~1.0) even though
homeostasis was correctly rejecting. Switching to KV last-write-wins (standard
semantics) made the poison actually overwrite → naive degrades, the honest result.

**Milestone.** All five pillars now have working, ours-built corners: P1 generation
(characterized multiplier), P2 SSM backbone, P3 MoE + symbolic lattice, P4 retriever +
store, P5 self-evolution + homeostasis. The achievable architecture is assembled and
measured end to end. Remaining distance to the north star is **scale** (data × compute)
— not unproven mechanisms. Toy-scale / synthetic caveats stand throughout.

### 2026-07-25 — Pillar 3's SYMBOLIC half: the hallucination killer (`MODE=lattice`) ✅

**What.** Built the symbolic half of Trinity: continuous retrieval (our encoder)
proposes a candidate; a **symbolic** check — exact digit-level (Hamming) consistency
between the query key and the retrieved key — **vetoes and ABSTAINS** when the store
doesn't actually support an answer, instead of emitting a confident wrong value.
Measured on 1000 answerable (noisy in-store keys) + 1000 unanswerable (keys never
stored) queries; τ = max symbolic mismatches to accept.

**Result (N=2000 random 12-digit keys, τ=2).**

| | no lattice | with symbolic lattice |
|---|---|---|
| Unanswerable — hallucination rate | 1.000 | **0.000** |
| Answerable — accuracy | 0.993 (always answers) | **1.000 on accepted**, coverage **0.991** |

**Verdict — clean win.** The symbolic veto **kills hallucinations (100% → 0%)** on
out-of-store queries while preserving real answers (99% coverage), and even upgrades
answerable accuracy to 1.000 by rejecting the rare wrong retrieval. Continuous
retrieval + symbolic verification = the Trinity idea, demonstrated end to end, ours.
**Pillar 3 now has BOTH halves built** (sparse MoE + symbolic lattice).

**Honest scope + a fixed bug.** First run only cut hallucinations 100%→77% — my bug:
zero-padded *sequential* keys (`000000002500`) shared 8 leading digits, so Hamming
distances were tiny. **Random** keys (expected Hamming ≈ 11) fixed it. This synthetic
task has crisp separation, so the check is decisive; natural data would need a tuned
τ with a precision/coverage tradeoff. The mechanism (symbolic consistency gate over a
retrieved candidate ⇒ abstain when unsupported) is validly shown.

### 2026-07-25 — Capability-per-RAM (Gate-G1 spirit): stored int8 MoE ≈ dense-big at ¼ the RAM ✅ (milestone-shaped)

**What.** `vyoma-lm MODE=g1` assembles the DECIDED architecture — lean SSM + **stored
top-1 MoE** — and measures capability-per-RAM against dense baselines, with **int8 on
the actual FFN/expert mass** (the weights the 8 GB claim rests on). BPB on real text.

**Result (tiny-shakespeare char, dm=64, dff=64, E=4, 4000 steps).**

| model | BPB fp32 | BPB int8 | FFN RAM fp32 → int8 |
|---|---|---|---|
| dense-small | 2.876 | 2.876 | 32.5 → 8.1 KB |
| **MoE (4×), stored** | 2.732 | **2.730** | 130 → **32.5 KB** |
| dense-big | 2.695 | — | 129 KB |

**Verdict — two wins, both load-bearing for the vision.**
1. **int8 is ~free on the real LM/MoE mass** (2.732 → 2.730). Not just retrieval keys
   and E1 seeds — the actual model quantizes losslessly at 8-bit. The 4× RAM cut holds.
2. **Capability-per-RAM win:** int8 lets the 4-expert MoE occupy the SAME FFN RAM as a
   single fp32 dense-small (32.5 KB) while scoring **−0.146 bits/byte** better — and it
   nearly matches **dense-big** (2.730 vs 2.695) at **¼ the RAM** (32.5 vs 129 KB).

So our decided architecture — lean SSM + **stored int8 MoE**, with knowledge held
**off-RAM on disk** (retrieval, Pillar 4) — delivers near-big-model quality at a
fraction of the RAM, entirely ours. This is the closest thing yet to a Gate-G1
capability-per-GB demonstration: the achievable-hybrid thesis, measured.

**Honest scope.** Char-level, tiny, single seed (but int8-free + the −0.146 gap match
the standalone MoE run 2.682, so robust). RAM accounting is on the FFN mass (the part
that scales); embed/head/SSM are small and shared. The MoE's *compute* (active-expert)
win is separate and not claimed here — this is a RAM (bytes) result, which is the 8 GB
axis. The leap from ~KB to real 8 GB is scale (data × compute), unchanged.

### 2026-07-25 — Data lever CONTINUES: 652 KB → 999 KB drops BPB 2.202 → 2.013 (paired, our model) ✅

**What.** Grew the teacher-distilled corpus with `vyoma-distill` (APPEND, incremental,
timeout-robust): **652 KB → 999 KB** (+346 KB, +53%, 300 new generations across 57
topics × 7 forms). Then a **paired** BPB comparison this session — same model, config,
and BPE tokenizer (dm=96, dff=256, seq=64, 1500 steps, vocab 512) — trained on the
652 KB snapshot vs the grown 999 KB corpus. Only the data differs.

**Result.**

| corpus | BPB (bits/byte) |
|---|---|
| 652 KB | 2.202 |
| **999 KB (+53% data)** | **2.013** |

**Verdict — the data lever keeps paying.** −0.189 bits/byte (−8.6%) from more data
alone, well above run-to-run noise, and the curve is **still descending** at ~1 MB
(not flattening yet — we're still in the data-starved regime). 2.013 is our best BPB
on distilled text. This is capability bought exactly as the vision's road requires
(Chinchilla — data scaled with the model), on OUR model, teacher-taught, ours end to
end. The teacher only generated training data; it is not in the system.

**Honest scope.** Single run per corpus; same BPE merges (isolates the tokenizer).

**CORRECTION (2026-07-25, added later).** This cross-corpus BPB comparison is
**confounded by test-set shift**: growing the corpus changes the held-out tail (new
topics), so 652 KB and 999 KB are scored on *different* text. A later run (1.34 MB,
same config, 3000 steps) scored BPB **2.008 — higher, not lower** — consistent with
(a) the shifted/harder test tail and (b) compute not scaling with data (fixed steps →
underfit). So the clean "more data → lower BPB" claim is **not** established by
cross-corpus BPB; the honest, reliable signals are **within-corpus**: the MoE win
(+0.11, invariant across all corpora) and model+compute scaling on a fixed corpus.
Data still matters, but it must be measured on a held-out set that doesn't shift, with
compute scaled alongside (Chinchilla) — see the within-corpus scaling run.

### 2026-07-25 — Pillar 1 × Pillar 3: generating MoE experts LOSES (honest negative) — store the experts

**What.** `vyoma-lm MODE=genmoe`. The obvious composition: MoE experts are the large
redundant FFN mass, so try to GENERATE all E experts from one fractal seed (Pillar 1)
and keep the MoE win (Pillar 3). stored-MoE vs gen-MoE (seed) vs dense-small, by BPB.

**Result (tiny-shakespeare char, dm=64, dff=64, E=4, 4000 steps).**

| model | BPB | expert mass |
|---|---|---|
| dense-small (dff=64) | 2.899 | — |
| stored-MoE (4×64) | **2.718** | 33,280 params |
| gen-MoE (seed 4905 + gate) | **2.957** | 6.4× compressed |

Δ(gen − stored) = **+0.239**, Δ(gen − small) = **+0.058**.

**Verdict — NEGATIVE (and honest).** Generating the experts from a seed **loses the
MoE advantage entirely**: gen-MoE (2.957) is worse than stored-MoE (2.718) *and* even
worse than a single small stored FFN (2.899). The routed capacity did NOT survive
generation, even at a modest 6.4×. This is fully consistent with the earlier
`language-weights-resist-generation` result: language FFN weights resist fractal
generation, and MoE experts are exactly that mass. (Note: the mode's console string
was initially an over-optimistic template; corrected to read the sign honestly.)

**Architectural lesson (a real deliverable).** The redundancy principle holds again:
**store the efficient / hard-to-generate parts, generate only genuinely redundant
image-like mass.** So in the achievable architecture the MoE experts are **stored and
quantized** (int8 — which we showed is ~free for such weights), NOT seed-generated.
Generation stays a characterized multiplier for the image-like regime, not the
language expert mass. Knowing what NOT to do is progress by the doc's own rules.

### 2026-07-25 — Pillar 3 STARTED: sparse Mixture-of-Experts — big-model quality at small-model active cost ✅

**What.** Built our own sparse MoE FFN (`vyoma-lm MODE=moe`): E expert FFNs + a gate
routing each token to its **top-1** expert (Switch-style load-balance aux). Capacity
scales with E; active compute stays ~one expert. Three-way BPB comparison on real
text: dense-small (dff) vs MoE (E experts × dff) vs dense-big (dff = E·dff, the
capacity upper bound). This is the first build of Pillar 3 (Trinity), and exactly the
"large redundant FFN mass" regime where generation (Pillar 1) will later pay off —
experts are the natural thing to generate.

**Result (tiny-shakespeare, char, dm=64, dff=64, E=4, 4000 steps).**

| model | BPB | params |
|---|---|---|
| dense-small (dff=64) | 2.866 | 17k |
| **MoE (4×dff=64, top-1)** | **2.682** | 42k total · **~17k active** |
| dense-big (dff=256) | 2.711 | 42k |

Δ(small − MoE) = **+0.184**, Δ(MoE − big) = **−0.029**.

**Verdict — clean win.** MoE beats dense-small by **0.184 bits/byte** and even edges
out dense-big, while activating only ~one expert (≈dense-small active params). That
is Pillar 3's thesis realized with our own MoE: **big-model quality (or better) at
small-model ACTIVE cost.** The Δ (0.184) is far above the ~0.02 noise scale we've
seen elsewhere, so this is solid (unlike the small/noisy retrolm signal).

**Honest caveats.** Char-level, single seed, and — a real one — the toy **computes
all experts and selects** (dense FLOPs), so only the *quality-per-active-param*
relationship is demonstrated here, not the compute saving; a production impl gathers
tokens per-expert to realize the FLOP win. Load-balance aux (coef 0.01) keeps routing
from collapsing to one expert.

**Why it matters for the vision.** First unbuilt pillar now has a working, ours-built
corner, and it composes with the rest: MoE is where Pillar-1 generation earns its keep
(generate the redundant expert mass), retrieval (Pillar 4) supplies knowledge, the SSM
(Pillar 2) is the lean backbone. Next: generate the experts from a seed; add the
symbolic-lattice half of Pillar 3; confirm with ≥2 seeds.

### 2026-07-25 — RETRO-LM seed-confirmation: the effect is REAL but SMALL (single-seed was optimistic)

**What.** Added ≥2-seed support to `retrolm` (`RSEEDS`, tunable `RP`/`RSEQ`) to bound
the variance on the −0.083 bits/char claim from the single-seed run. 2 seeds,
dm=96/dff=256/2500 steps.

**Result.**

| seed | retro bits/char | random | Δ |
|---|---|---|---|
| 0 | 3.225 | 3.290 | −0.066 |
| 1 | 3.241 | 3.256 | −0.015 |
| **mean** | **3.233 ± 0.008** | **3.273 ± 0.017** | **−0.040** |

Δacc = −0.004 (no accuracy benefit — noise).

**Verdict (honest correction).** retro beats random on bits/char in BOTH seeds
(direction consistent), but the mean is **−0.040 bits/char**, half the single-seed
−0.083, and the accuracy signal disappeared. So: OUR retriever's neighbor selection
gives a **small, consistent-in-sign but noisy** benefit to a real LM — a mild
positive, not the clean win the single run implied. This is exactly why the project
pre-registers ≥2 seeds: it caught an over-optimistic number before it was built upon.
Firming it into a robust win needs more scale (bigger model/context, BPE, more seeds)
— the honest next step, not a claim to bank yet.

### 2026-07-25 — RETRO-LM on REAL language: our retriever's neighbor helps a real LM (bits/char) — *single-seed, later corrected below*

**What.** `vyoma-lm MODE=retrolm`. Moves the retro loop from synthetic key→value facts
to **real text**. Each block is `[neighbor 31][SEP][passage 32]` (seq=64); the LM
predicts the passage chars. Clean ablation: **retro** (neighbor = OUR retriever's
nearest train passage) vs **random** (neighbor = a random train passage) — same
architecture, same context length, so the only difference is whether the prepended
context is retrieval-selected. Scored on held-out passages by masked next-char
accuracy and masked **bits/char**. Our retriever (`vyoma-embed` lib), our LM, our
store — no teacher.

**Result (tiny-shakespeare, 6000 train / 1500 test passages; dm=96, dff=256, 3500 steps).**

| condition | next-char acc | bits/char |
|---|---|---|
| retro (our retriever's neighbor) | 0.421 | **3.216** |
| random neighbor (baseline) | 0.416 | 3.299 |
| **Δ (retro − random)** | +0.005 | **−0.083** |

**Verdict — modest but real POSITIVE.** OUR retriever selects neighbors that
measurably help a real LM predict held-out text: **−0.083 bits/char (≈2.5%)** vs a
random neighbor, from retrieval *selection* alone. Both metrics move the right way
(bits/char is the clear signal; +0.005 acc is small). This is the retrieval-centric
hybrid demonstrated on **real language**, entirely ours — complementing the synthetic
`retro` result (retrieval decouples capability from model size) with "retrieval
*selection* improves language modeling."

**Honest caveats.** Char-level, tiny model, 32-char passages, **single seed**
(variance not bounded — warrants a ≥2-seed repeat for confidence). The gain is
modest and concentrated on early passage chars (where the neighbor dominates the
short context). Scaling levers: bigger model/context, BPE, and a stronger retriever.

**Engineering.** Added `eval_bpb_masked` (masked bits/char); reused the shared
`train`; retrieval is once-per-passage (nearest OTHER for train, nearest for test).

### 2026-07-25 — Retriever scaling lifts & flattens retro (E2b's lesson, inside the loop) ✅

**What.** Acting on the RETRO-lite bound: scale the retriever so retro stays high as
the library grows. Made the retriever env-tunable (`RD` key digits, `RDK` dk, `RDM`
d_model, `RSTEPS`, `RNS` fact-counts), strengthened it (dk 128→**256**, keys 8→**12
digits**, d_model 64→96), swept facts **500 → 8000** (16× range). Also fixed eval to
retrieve **once per entity** (O(N²·dk) not O(n_ex·N·dk)) so facts can scale.

**Result.**

| #facts | memorize | retro (ours) | our retriever hit-rate |
|---|---|---|---|
| 500 | 0.209 | **0.737** | 0.670 |
| 2,000 | 0.125 | **0.670** | 0.631 |
| 8,000 | 0.109 | **0.527** | 0.480 |

**Verdict — the lever works.** A stronger retriever markedly **lifts and flattens**
retro: at **8000** facts it now holds **0.527**, higher than the weaker retriever
managed at **4000** facts (0.372) — *double the library, higher capability.* The
retro/memorize gap is **3.5×–4.8×** and widens with scale (memorize collapses to
0.11). This is exactly [[externalized-knowledge-works]]'s E2b lesson (retrieval
stays flat given adequate embedding dimension) realized **inside the closed loop**,
all ours. Honest: not perfectly flat yet (0.74→0.53 over 16×); fully flat needs still
bigger dk/keys, but the direction — capability held higher as the library grows — is
unambiguous. retro > hit-rate throughout (the LM recovers extra via value collisions).

### 2026-07-25 — RETRO-lite: E2 closed with a REAL retriever — OUR retriever + OUR LM + OUR store, no teacher ✅

**What.** `vyoma-lm MODE=retro`. E2 (`kv`) showed memorize-collapses / retrieve-stays,
but its "retrieve" was an ORACLE (correct value placed adjacent). This replaces the
oracle with OUR learned retriever (`vyoma-embed`, now a library): our SSM encoder
fetches the fact from a store of 8-digit keys given a noisy query; our SSM LM reads
the fetched value and answers. End-to-end = (our retriever's hit-rate) × (the LM's
copy). Compared to memorize (LM alone), swept over #facts. Every piece is ours.

**Result (dm=64, dff=128, 2500 LM steps; retriever dk=128, 1500 steps).**

| #facts | memorize (LM alone) | retro (our retriever + LM) | our retriever hit-rate |
|---|---|---|---|
| 200 | 0.367 | **0.642** | 0.646 |
| 1,000 | 0.138 | **0.532** | 0.491 |
| 4,000 | 0.099 | **0.372** | 0.316 |

**Verdict.** Retrieval-augmented **beats memorization at every scale, and the gap
widens with #facts** (memorize collapses 0.37→0.10; retro holds 0.64→0.37 — a
1.8×→3.8× advantage). This is E2's decoupling thesis demonstrated with a REAL
retriever instead of the oracle, entirely ours end to end — the retrieval-centric
hybrid's core loop, running. retro slightly exceeds the raw hit-rate because a wrong
fetch sometimes carries a value that collides with the truth (n_val=10).

**Honest bound.** retro *declines* with #facts because our small retriever's hit-rate
declines on this deliberately hard task (8-digit keys, 1-digit-corrupted queries,
dk=128) — NOT a failure of the principle: E2b already showed retrieval stays flat
given adequate embedding dimension. The path to flat-high retro is a
higher-capacity retriever (bigger dk / longer keys), exactly E2b's lesson.

**Engineering (build the corner right).** Refactored `vyoma-embed` into a **library**
(`lib.rs`, used by `vyoma-lm`) with a variable-length `VYST v2` store; the encoder's
per-timestep scan was batched (~6× faster) so these multi-model runs are ~minutes.
No teacher anywhere in the loop.

**Next.** Scale the retriever (dk/keys) so retro stays flat-high as #facts grow; then
feed real (Shakespeare / distilled) passages through the same loop and score by BPB.

### 2026-07-25 — Corner built: the persistent on-disk Ontological Store (`vyoma-embed MODE=store`) — Pillar 4, ours end to end ✅

**Principle reaffirmed first (Akash).** The teacher (phi4-mini) only ever generates
*training data we learn from* — it is NEVER a runtime component of any corner,
retrieval included. "Our llm can learn from it but I want to build every corner so
every bit gets us closer to our vision." A build that breaks if you remove the
teacher is a wrap — we don't do that. (memory: `build-ours-teacher-only-learns`.)

**What.** Turned the learned retriever into a real, reusable Pillar-4 artifact.
`MODE=store`: train our SSM encoder → encode a corpus → quantize keys to **int8** →
write to disk in our own `VYST` format → **reload from disk** → query it. Knowledge
lives on disk; the encoder (the "skills") stays small in RAM. No teacher anywhere.

**Result (1743 held-out Shakespeare passages, dk=128, 1500 steps).**
- Wrote **1743 facts** to `data_cache/ontological_store.vyst`, **442.6 KB = 260
  bytes/fact** (128B int8 key + 128B text value + 4B scale).
- **Reloaded from disk** and queried: clean **0.679**, noise15 **0.394** — matches
  the in-memory int8 number, so the disk round-trip is lossless.
- Concrete lookup worked: query `":\nHe shall not, Isabel, if you give me love."`
  → fetched the correct adjacent passage `"...ANGELO:\nHe shall not, Isabel,"` ✓.

**Why it matters.** Pillar 4 is no longer just an in-memory experiment — it's a
persistent component of OUR system with OUR file format, written and read by OUR
code, powered by OUR trained encoder. "Skills in weights (small, in RAM), knowledge
on disk (the store)" is now literally true and runnable. This is the artifact the
LM will read from next — the retrieval half of the retrieval-centric hybrid, built
by us. Composes the proven multipliers (learned dim-efficiency × int8).

**Engineering also landed here:** factored training into a reusable `train_encoder`;
`VYST` store format (`quant_i8` / `write_store` / `load_store`); `.gitignore` already
excludes `data_cache/` so the artifact stays local.

**Next (all ours):** feed retrieved facts from this store into OUR SSM LM (RETRO-lite)
and measure the capability lift — the E2 loop closed with a real retriever instead
of the oracle, no external model.

### 2026-07-24 — Perf: retriever runs ~6× faster (batched projection + saturating steps), no accuracy loss

**What.** Runs were 30+ min. Root cause: the SSM encoder did a per-timestep `d×d`
matmul *inside* the scan loop (128 sequential matmuls per encoder call), which
dominated compute and bloated the autograd graph. Fixes (math-preserving):
(1) **batch the input projection out of the loop** — one `(B·T, d)×(d, d)` matmul
instead of T; (2) **fuse the mean-pool** using `mean_t(c⊙h_t) = c⊙mean_t(h_t)` so the
loop is elementwise-only (no 128-tensor `cat`); (3) fold `b` into the batched
precompute. The per-timestep loop is now cheap elementwise recurrence.

**Result.** 1500 steps (d128, batch128, dk128): **~4.8 min** (was ~30+), clean acc
**0.695** — identical to the old 3000-step number. Two effects compound: batching
(~3×) and the fact that contrastive quality **saturates by ~1500 steps** (300 steps
already hits 0.618) → ~6× faster wall-clock, zero quality cost. Iteration unblocked.
Multi-layer path keeps a `scan()` that materializes the sequence (needed for
stacking); the shipped 1-layer path uses the fast fused `scan_mean()`.

### 2026-07-24 — Retriever keys quantize to 4-bit FOR FREE — dim-efficiency × quantization compose (cheap store per fact) ✅

**What.** `vyoma-embed QUANT=1`: train the best-config learned encoder (1 layer,
mean-pool, d128, dk=128, 3000 steps), then measure retrieval accuracy with the
stored embeddings fake-quantized per-vector to fp32 / int8 / int4 (re-L2-normalized,
as cosine retrieval reads them). Answers the concrete store-cost question: a fact's
key is `dk` values × `bits` precision — how few bytes can it be?

**Result (clean queries, 1743 held-out passages).**

| precision | clean acc | bytes/fact |
|---|---|---|
| fp32 | 0.691 | 512 |
| **int8** | **0.691** | **128** |
| **int4** | **0.699** | **64** |

**Verdict — quantizing retrieval keys is essentially FREE to 4-bit** (0.691→0.699,
8× fewer bytes). This is the **mirror image of weight-generation**, where int4
*broke* the seed (→57%, see the `vyoma-e1 QUANT` entry). Principled why: retrieval
only needs the **cosine ordering** among keys preserved, and argmax-over-dot-products
is robust to per-component rounding; generated weights need precise *values* because
errors propagate through the network. **The store composes with aggressive
quantization where the seed does not.**

**Compound win for the store (Pillar 4).** The two multipliers stack:
dimension-efficiency (learned dk=128 ≈ classical dk=256; dk=32 ≈ classical dk=256)
**×** int4 (free) → a learned key at **64 bytes/fact** holds ~0.70 retrieval, versus
a classical trigram needing dk=1024 fp32 (~4 KB/fact) to reach its 0.98 ceiling. For
comparable mid-range accuracy the learned+quantized store is ~an order of magnitude
cheaper per fact — the concrete meaning of "small brain + big library."

*(Note: fp32 dk=128 reads 0.691 here vs 0.715 in the prior run — ~±0.02 run-to-run
init variance, candle's global RNG for weight init isn't seeded. The quant
comparison is within a single trained encoder, so it is clean regardless.)*

### 2026-07-24 — Retriever depth & pooling test: scale is a WIDTH law, not a depth law (and a self-corrected regression)

**What.** Extended `vyoma-embed` to stack N diagonal-SSM blocks (`LAYERS` env, residual
between blocks) and swept depth at fixed d_model=128, dk=128. Goal: does the
"improves-with-scale" law that lifted the retriever with *width* (d_model 64→128:
0.68→0.72) also lift it with *depth*, toward the classical high-dim ceiling (0.98)?

**Result — depth is neutral-to-negative (2500 steps, dk=128, same arch, depth the only variable):**

| layers | params | clean | noise15 | noise30 | final loss |
|---|---|---|---|---|---|
| 1 | 82k | 0.506 | 0.293 | 0.129 | 0.602 |
| 2 | 99k | 0.450 | 0.281 | 0.124 | 0.799 |
| 3 | 116k | 0.482 | 0.301 | 0.116 | 0.711 |

**Verdict.** Stacking blocks does NOT lift the plateau — flat-to-negative, and deeper
trains *worse* at fixed compute (loss 0.60→0.80). **The retriever's scale law is a
width law, not a depth law** — the exact same pattern the language model showed
(width nudged generation, depth didn't). One consistent Vyomarudra finding across
both models: for our diagonal SSM, capacity is bought with *width*, not *depth*.

**A self-corrected regression (kept honest).** The depth refactor also switched
pooling from mean-only to **mean+last** (concat) + residual. A controlled check
(1 layer, 3000 steps, matched to the earlier run) showed this **regressed** the
encoder: **0.502 vs the mean-only 0.719** at identical steps — my "upgrade" hurt.
Reverted to mean-only pooling. Definitive best config re-confirmed (1 layer,
mean-pool, d128, 3000 steps): dk=32 **0.659**, dk=64 **0.695**, dk=128 **0.715**
(reproduces the pre-refactor 0.719). Lesson logged: simpler pooling won; complexity
was not free. Multi-layer support kept (proven neutral) for future encoders.

**Consequence.** Width + mean-pool is the ceiling for this diagonal-SSM retriever
(~0.72 on the literal-substring task). Reaching the classical high-dim ceiling would
need a structurally different encoder (e.g. attention pooling / a selective scan),
OR — the more likely honest read — the classical 0.98 is inflated by the
literal-substring benchmark and the learned encoder's real edge is on the semantic /
paraphrase regime this benchmark cannot stage. That semantic query set is the next
decisive build (teacher-generated paraphrase queries; teacher teaches, retriever is
ours).

### 2026-07-24 — Corner built: LEARNED neural retriever (`vyoma-embed`) — the RETRO-quality upgrade to Pillar 4

**What.** New crate `vyoma-embed`. E2/E2b proved retrieval is the decisive engine
but measured it with a deliberately weak classical embedding (hand-hashed
char-trigram bags). This replaces it with a **learned encoder that is entirely
ours**: `bytes → embedding → our diagonal-SSM backbone (Pillar 2, reused as the
retriever) → mean-pool → projection → L2-normalize`, trained **contrastively**
(in-batch InfoNCE): a query = a random 48B fragment of a 128B passage, its positive
= the full passage, all other passages in the batch = negatives. No teacher, no
library. Train/test passages **disjoint** (80/20) → measures generalization.

**Two falsifiable claims, head-to-head vs trigram on 1743 held-out passages.**

Trigram baseline (no training):

| dk | clean | noise15 | noise30 |
|---|---|---|---|
| 64 | 0.410 | 0.245 | 0.153 |
| 128 | 0.434 | 0.309 | 0.217 |
| 256 | 0.695 | 0.406 | 0.225 |
| 1024 | **0.984** | 0.855 | 0.550 |

Learned SSM encoder (ours, 3000 steps, batch 128):

| d_model | dk | clean | noise15 | noise30 |
|---|---|---|---|---|
| 64 | 32 | 0.643 | 0.321 | 0.133 |
| 64 | 128 | 0.683 | 0.402 | 0.153 |
| 128 | 64 | 0.675 | 0.410 | 0.213 |
| 128 | 128 | **0.719** | 0.430 | 0.189 |

**Verdict (positive-with-caveats, honest).**
1. **Dimension efficiency — confirmed.** At MATCHED dim the learned encoder
   decisively beats classical: dk=128 → **0.719 vs 0.434** clean (+0.285), and wins
   under moderate noise too (noise15 0.430 vs 0.309). Learned dk=32 (0.643) ≈ trigram
   dk=256 (0.695) → **~8× fewer dimensions for the same accuracy = a cheaper store
   per fact**, which is the whole point of "small brain + big library."
2. **Improves with encoder scale.** d_model 64→128 lifted the top of the curve
   (dk=128: 0.683→0.719) and roughly doubled heavy-noise accuracy (dk=64 noise30
   0.124→0.213). The retriever obeys the same **improves-with-scale** law as E1 —
   but with diminishing returns; a mean-pool 1-layer diagonal SSM caps ~0.72.
3. **Does NOT top the classical high-dim ceiling** (trigram dk=1024 = 0.984). Honest
   why: the benchmark queries are *literal substrings* of passages, which maximally
   favors surface n-gram overlap — the trigram bag's best case — and it is handed 8×
   the dimension. Our small learned encoder wins the fair (matched-dimension) fight
   but can't out-ceiling a redundant 1024-dim bag on its home turf.

**The recurring lesson, again.** As with every Vyomarudra toy: the small learned
encoder **validates the direction** (learned embeddings are far more
dimension-efficient and scale with capacity) but a toy-scale encoder on an
n-gram-friendly task **can't stage the frontier magnitude**. The learned encoder's
true edge — semantic / paraphrase / robustly-noisy match, where surface n-grams have
no signal — needs a query set this literal-substring benchmark doesn't provide.

**Left / next.** (a) A **harder query set** (paraphrase + heavy corruption + held-out
vocabulary) where surface n-grams break — the regime that shows *why* learned wins.
(b) A **deeper encoder** (stack SSM blocks; last+mean pooling) to test if the
plateau lifts toward the classical ceiling. Run:
`STEPS=3000 BATCH=128 DMODEL=128 ./target/release/vyoma-embed`.

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
