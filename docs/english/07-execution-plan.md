# 07 — Execution Plan (post-Phase-0 pivot)

Concrete, ordered, branch-aware. This is the "how we make it happen" doc — read
with `PROGRESS.md` (results) and `06-refined-architecture.md` (the design).

## Where we are

Phase-0 verdict: generative weights are **real** on redundant image weights
(423×, improving with scale) but **tie a small model on real-language FFNs** —
the extreme compression the 300B-in-8GB north star needs did **not** appear on
language. The honest target is now the doc's stated realistic win: **a genuinely
efficient small model (3–10× capability-per-GB)** via a hybrid, not a 300B miracle.

## Decision gate — the running scale test

**Test:** does generation's edge on language reappear with a bigger, deeper target?
(`vyoma-lm`, dm=256, dff=1024, then multi-layer.)

- **If gen ≫ plain returns** → the strong bet lives; prioritize step A.
- **If gen ≈ plain persists** → strong bet parked; go straight to the hybrid (B).

## The build order (either branch converges here)

### Step A — Depth & scale test for generation *(only if the scale test hints yes)*
- Add multi-layer support to `vyoma-lm` (stack N blocks; generate all FFNs from one seed).
- Success: generated multi-layer FFN stack beats same-footprint plain by ≥3 pts on text.
- Kill: no edge at 2–4 layers → generation is a modest ~2–5× tool, not the engine. Park it.

### Step B — Externalized knowledge (Pillar 4 / E2) — **the highest-value branch-independent build**
The doc's "most practical pillar," and it helps a small model in *every* branch.
- **Experiment (clean, avoids the SSM-can't-copy confound):** fixed small model,
  synthetic key→value facts. Compare (i) **memorize** — value must live in weights;
  vs (ii) **retrieve** — the fact's value embedding is injected at the prediction
  step (trivial to consume, isolates the claim). Sweep #facts N.
- Success: memorize accuracy collapses as N grows; retrieve stays flat → **knowledge
  belongs on disk, not in weights.** That decouples capability from model size —
  the real lever for "small brain + big library."
- Then: real embedding-kNN retrieval over a disk store (RETRO-style), measure recovery.

### Step C — Quantization (proven, low-risk)
- 4-bit the stored weights (SSM core + any stored FFN + embeddings). Free ~4×.
- Compose with whatever generation ratio survived A (e.g., 4× gen × 4× quant = 16×).

### Step D — Assemble the honest hybrid & measure capability-per-GB
- Stored lean SSM backbone + (modest generated or quantized) FFN + retrieval store.
- Benchmark vs a same-RAM dense baseline (Gate G1 spirit: win on ≥60% of a small eval suite).

## Success criteria (honest, restated)

- **Realistic win (target):** 3–10× capability-per-GB vs a same-quality dense small model. Achievable.
- **Stretch:** 10×+ — needs A to resolve favorably (unlikely on current evidence).
- **North star (not a milestone):** frontier-competitive in 8 GB. Direction, not promise.

## What we will NOT do again
- Toy tasks that don't need a big FFN (they can't test generation — see PROGRESS).
- Generating lean/dynamical cores (SSM recurrence) — store those.
- Claiming the 300B-in-8GB endpoint as reached. It isn't, and the evidence says be humble.
