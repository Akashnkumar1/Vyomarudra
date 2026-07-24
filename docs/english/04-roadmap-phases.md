# 04 — The Roadmap: Phase 0 → Phase 3

**Core principle:** Every speculative idea gets a cheap test first. No mystical idea rides the critical path until an experiment proves it.

```
M0────M4──────────M12────────────M24──────────────M48
 │ E1–E5 │  MVN 1–3B   │  30–70B-equiv   │  Self-evo → 300B-equiv
 └─ G0 ──┴──── G1 ─────┴────── G2 ───────┴─── G3 ⭐
```

---

## 🔬 Phase 0 — Falsification Experiments (Months 0–4)

**Team:** 1–2 people. **Hardware:** 1 GPU or a Mac. **Scale:** ≤300M params.
Purpose: kill bad ideas fast and cheap.

### The Experiments

- **E1 — Generative Weights (THE experiment):** Train a hypernetwork to generate a 125M–300M model's weights from a <10M seed. Measure the compression-vs-quality curve. → Decides Risk #1, the project-killer.
- **E2 — Externalized Knowledge:** Take a 1–3B model, add RETRO-style retrieval from a disk store. Measure how much knowledge-task performance is recovered. → Decides Risk #3.
- **E3 — Field Dynamics:** Plain Mamba vs. Mamba + multi-scale coupling terms. Does "resonance" produce measurable gains, or is it just poetry? → Decides whether Pillar 2's addition survives.
- **E4 — Sparse Awakening:** A router that instantiates only k of N generated blocks per token. Measure quality vs. active-parameter curve.
- **E5 — Stability Tooling:** Build the Lyapunov monitor + spectral regularizer library. Everything downstream depends on this.

### Gate G0
E1 achieves ≥100x compression at ≤10% quality loss, **OR** a viable hybrid path exists (e.g., 20x generation + 4-bit storage + retrieval = credible 300B-class capability path). If neither → publish the findings and pivot. A documented negative result is still a contribution.

---

## 🏗️ Phase 1 — Minimal Viable Nexus (Months 4–12)

**Team:** small. **Hardware:** modest cluster.

- Build the 1–3B *virtual*-scale prototype: Eternal Seed + SSM-hybrid backbone + Symbolic Lattice + disk Ontological Store + Dual Stream (fast/slow).
- **Distill from an open frontier teacher** — avoids frontier-scale training compute (Risk #6).
- Build the eval harness NOW: frozen benchmark suite + stability metrics + hallucination probes.

**Deliverable:** Runs on an 8GB M-series Mac at ≥15 tok/s; matches or beats a dense 3B (Phi/Qwen class) on reasoning + coding while using ≤1.5 GB for the "model."

### Gate G1
Beats the same-RAM dense baseline on ≥60% of the eval suite.

---

## 📈 Phase 2 — Scale + Crystallization (Months 12–24)

- Scale virtual capacity to **30–70B-equivalent** inside the same 8GB envelope (this is where the compression ratio must start paying).
- Add: predictive prefetcher, knowledge **crystallization** loop (frequently-used patterns get "burned into" weights — like how riding a bike goes from conscious effort to automatic), causal what-if rollouts in the slow stream, cross-modal encoders into the shared manifold.
- Hardware co-design: Metal/MLX custom kernels for on-the-fly weight generation; block-granular caching.

### Gate G2
Matches a dense 30B (Qwen-30B class) on the suite, in 8 GB, at ≥20 tok/s.

---

## 🚀 Phase 3 — Self-Evolution + Productization (Months 24–48)

Only exists if G2 passed cleanly.

- Enable bounded per-session weight deltas (homeostasis-gated, rollback-able, eval-gated).
- Per-user personalization: each user gets their own instantiation from the shared seed.
- Serious safety program: interpretability probes on the lattice, red-teaming of the evolution loop, formal delta budgets.

### Gate G3 (North Star)
Frontier-competitive on reasoning/coding in 8 GB. 10x efficiency vs. contemporary frontier models on capability-per-GB and capability-per-watt.

---

## What To Do THIS Month (solo-feasible)

1. **Reproduce baselines:** run Mamba-2 / RWKV / a small LFM locally; get lm-eval-harness working.
2. **Start mini-E1:** fork a hypernetwork repo; target generating a 125M GPT's weights from a 5M seed. This single experiment tests the load-bearing assumption of the entire project.
3. **Start E2 cheaply:** wire a 3B model to a local vector store (even sqlite-vec) RETRO-style; measure knowledge recovery.
4. **Read the load-bearing papers:** Hypernetworks (Ha 2016) · Mamba-2 · RETRO (DeepMind) · Mixture-of-Depths · Test-Time Training layers · Dreamer v3 · EWC/continual-learning surveys · KAN.
