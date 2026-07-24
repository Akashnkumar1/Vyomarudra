# 02 — Architecture: The Five Pillars of Vyomarudra

Final synthesis of the design evolution: AetherNet v1–v2 → NexusCore v3 → Elysium v4 → Vyomarudra v5/v2.

---

## Pillar 1: Generative Weights — "The Eternal Seed" 🌱

**What:** A small network (10–100M params, ~200 MB) that **generates the big network's weights on demand**, instead of storing them.

**Analogy:** DNA. Your DNA is only ~750 MB of information, yet it builds a brain with 86 billion neurons. DNA doesn't store neurons — it stores the *rules for building* neurons.

**Real research this stands on:** Hypernetworks (Ha et al., 2016) — networks that generate other networks' weights. Proven: 10–50x compression with small quality loss.

**Our addition:** Fractal structure (Iterated Function Systems) — one generator serving all scales, pushing compression toward 100–1000x. **This is the project's biggest bet** — which is why it gets tested FIRST (Phase 0, Experiment E1).

---

## Pillar 2: SSM Backbone — "The Resonance Manifold" 🌊

**What:** Transformers' attention mechanism remembers every past token (KV cache) — memory explodes as context grows. **State Space Models (SSMs)** — Mamba, RWKV, Liquid AI's LFM family — keep a **fixed-size running state** instead. Like a river: water flows in, the state updates, memory stays constant.

**Analogy:** Transformer = keeping a word-by-word transcript of every conversation. SSM = keeping a running mental summary. Humans keep summaries too.

**Why it matters for 8GB:** Even a 1-million-token context in **constant memory**. This is already proven technology (Mamba-2; the LFM2.5 models on Ollama are this family).

**Our addition:** Multi-resolution nesting — fast/slow layers nested inside each other (micro = intuition, macro = long-term coherence) with wave-style coupling. This is the old "resonance chambers" idea, now grounded in math:

`∂ψ/∂t = D∇²ψ + F_θ(ψ) + R_φ(ψ, u(t))`

where ψ is the manifold state, F_θ a learned reaction term, R_φ learned input coupling. Discretized, this IS a generalized SSM — which is why we start from Mamba-class code, not a blank page. The open research question: does the diffusion/coupling term D∇²ψ buy anything measurable? Testable cheaply at 100M scale (Experiment E3).

---

## Pillar 3: The Trinity — Continuous + Sparse + Symbolic 🔱

Three computation modes working together:

1. **Continuous field** (intuition/creativity) — the SSM backbone. The brain's "System 1": fast, gut-feel.
2. **Sparse activation** (efficiency) — only 0.1–1% of the network active per token. Analogy: a city doesn't keep every streetlight on; lights turn on where the traffic is. Real research: Mixture-of-Experts, Mixture-of-Depths.
3. **Symbolic lattice** (precision/truth) — a discrete layer keeping facts, logic, and math **exact**, not approximate. Analogy: calculator vs. mental math. The neural net says "about 4000"; the symbolic layer says "exactly 4096". **This is the main hallucination killer.**

---

## Pillar 4: Externalized Knowledge — "The Ontological Store" 📚

The most practical pillar. The big unlock:

> **Skills live in weights. Knowledge lives on disk.**

Today's models bake facts like "Paris is the capital of France" into their weights — that's why they're so heavy. We keep facts in a **compressed knowledge store on disk** (concepts + relations + embeddings), retrieved at runtime — but not bolted-on like typical RAG. **RETRO-style** (DeepMind's proven research): retrieval happens *inside* the forward pass, so it feels like internal memory to the model.

**Analogy:** A brilliant doctor doesn't memorize every textbook. They learn *reasoning* and keep reference books nearby. Small brain + big library > big brain that memorized everything.

**Bonus — Predictive Prefetch:** the model guesses what knowledge it will need next and loads it early (like Netflix pre-buffering the next episode). Disk latency becomes invisible.

---

## Pillar 5: Bounded Self-Evolution + Homeostasis 🧬

**What:** After each session, the model makes small, safe weight updates — learning your style, your domain. Never full retraining.

**Safety mechanism (Homeostasis Controller):** Like your body regulating temperature at 37°C, a controller monitors the model's "health metrics" (stability, coherence). Every change has a **budget**, every change is **rollback-able**, and every change must pass a frozen test suite before committing. Fail = automatic undo.

**Real research:** Continual learning (EWC), Test-Time Training layers (2024). Our addition: a Lyapunov-bounded delta budget (math that guarantees the system can't spiral into chaos).

---

## Full Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│  ETERNAL SEED (~200 MB)                              │
│  DNA of the model — weight-generation rules          │
└──────────────┬──────────────────────────────────────┘
               │ generates weights on demand
┌──────────────▼──────────────────────────────────────┐
│  RESONANCE MANIFOLD (~2.5 GB active)                 │
│  SSM backbone, multi-resolution, sparse activation   │
│  ┌─────────────┐        ┌──────────────────────┐    │
│  │ FAST STREAM │  ⇄     │ SLOW STREAM          │    │
│  │ (intuition) │        │ (reasoning, simulate │    │
│  │ System 1    │        │  what-ifs) System 2  │    │
│  └─────────────┘        └──────────────────────┘    │
└───────┬─────────────────────────────┬───────────────┘
   anchors to                    retrieves from
┌───────▼──────────┐    ┌────────────▼────────────────┐
│ SYMBOLIC LATTICE │    │ ONTOLOGICAL STORE (on DISK) │
│ (~0.5 GB)        │    │ All facts & memories.       │
│ Exact facts,     │    │ Predictive prefetch streams │
│ logic, math      │    │ into RAM before needed.     │
└──────────────────┘    └─────────────────────────────┘

HOMEOSTASIS CONTROLLER: always on — stability monitor,
energy budget, self-evolution gatekeeper.
```

## 8 GB Memory Budget

| Component | Budget |
|---|---|
| Eternal Seed + hypernetwork | 0.2 GB |
| Active manifold (generated weights, working set) | 2.5 GB |
| State (SSM state is O(1) in context length!) | 1.0 GB |
| Symbolic lattice + router | 0.5 GB |
| Prefetch buffers + ontology cache | 1.5 GB |
| OS + runtime headroom | 2.3 GB |
| **Total** | **8.0 GB ✅** |

## Concept-to-Research Grounding Map

| Vyomarudra concept | Nearest real research | What we add |
|---|---|---|
| Eternal Seed | Hypernetworks (Ha 2016), implicit neural representations | Fractal/IFS parameterization; two-tier fast-approx + refine |
| Resonance Manifold | SSMs (Mamba/S4), RWKV, Liquid networks | Multi-resolution nesting; cross-scale coupling |
| Sparse awakening | MoE, Mixture-of-Depths, spiking NNs | Symbolic router choosing which generated blocks to instantiate |
| Symbolic Lattice | Neuro-symbolic systems, KG-augmented LMs | High-precision anchors that gate/correct the field |
| Ontological Store | RAG, RETRO, memorizing transformers | Knowledge ONLY external; weights are knowledge-free skill engines |
| Predictive prefetch | Expert offloading, speculative decoding | Learned prefetcher for generated blocks |
| Crystallization | Distillation, LoRA merging, fast-weight programmers | Session-level consolidation loop |
| Self-evolution | Continual learning (EWC), Test-Time Training | Lyapunov-bounded delta budget + rollback |
| Causal weaving | World models (Dreamer v3) | Cheap counterfactual rollouts in the slow stream |
| Quantum-inspired manifold | Hyperdimensional computing | Parallel branch exploration with learned collapse |
| Meta-conscious loop | o1-style test-time compute / reflection | Built into the slow stream, not prompted |

## Naming Continuity (our discussion → final doc)

| Discussion name | Final name |
|---|---|
| Procedural HyperFractal Core (AetherNet) | Eternal Seed |
| Resonance Chambers (NexusCore) | Resonance Manifold |
| Dual-Stream (Elysium) | Fast/Slow Stream |
| Ontological Compression Engine (Vyomarudra) | Ontological Store |
| Sheaf Cohomology / Categorical Resonance | Open research track — NOT on critical path (more poetry than math today) |
