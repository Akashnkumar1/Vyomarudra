# 02 — Architecture: Vyomarudra ke Paanch Pillars (Hinglish)

Humari design journey ka final synthesis: AetherNet v1–v2 → NexusCore v3 → Elysium v4 → Vyomarudra v5/v2.

---

## Pillar 1: Generative Weights — "The Eternal Seed" 🌱

**Kya hai:** Ek chhota network (10–100M params, ~200 MB) jo bade network ke weights ko **on-demand generate** karta hai — store karne ki jagah.

**Analogy:** DNA. Tera DNA sirf ~750 MB information hai, lekin usse 86 billion neurons wala brain banta hai. DNA neurons store nahi karta — neurons *banane ke rules* store karta hai.

**Real research jispe yeh khada hai:** Hypernetworks (Ha et al., 2016) — networks jo dusre networks ke weights generate karte hain. Proven: 10–50x compression with small quality loss.

**Humara addition:** Fractal structure (Iterated Function Systems) — ek hi generator sab scales pe kaam kare, jisse compression 100–1000x tak push ho. **Yeh project ka sabse bada bet hai** — isliye isko sabse PEHLE test karenge (Phase 0, Experiment E1).

---

## Pillar 2: SSM Backbone — "The Resonance Manifold" 🌊

**Kya hai:** Transformer ka attention mechanism har purane token ko yaad rakhta hai (KV cache) — context badhne pe memory explode. **State Space Models (SSM)** — Mamba, RWKV, Liquid AI ke LFM models — ek **fixed-size running state** rakhte hain. Jaise river: paani aata hai, state update hoti hai, memory constant.

**Analogy:** Transformer = har conversation ki word-by-word transcript rakhna. SSM = running mental summary rakhna. Insaan bhi summary hi rakhta hai.

**8GB ke liye kyun important:** 1 million token context bhi **constant memory** mein. Yeh already proven tech hai (Mamba-2; Ollama pe jo LFM2.5 tune dekha tha — woh isi family ka hai!).

**Humara addition:** Multi-resolution nesting — fast/slow layers ek dusre ke andar nested (micro = intuition, macro = long-term coherence), wave-style coupling ke saath. Yeh purana "resonance chambers" idea hai, ab math mein grounded:

`∂ψ/∂t = D∇²ψ + F_θ(ψ) + R_φ(ψ, u(t))`

ψ = manifold state, F_θ = learned reaction term, R_φ = learned input coupling. Discretize karo toh yeh generalized SSM hi hai — isliye hum Mamba-class code se start karenge, blank page se nahi. Open question: kya coupling term D∇²ψ measurable fayda deta hai? 100M scale pe saste mein testable (Experiment E3).

---

## Pillar 3: The Trinity — Continuous + Sparse + Symbolic 🔱

Teen computation modes ek saath:

1. **Continuous field** (intuition/creativity) — SSM backbone. Brain ka "System 1": fast, gut-feel.
2. **Sparse activation** (efficiency) — har token pe network ka sirf 0.1–1% active. Analogy: puri Delhi ki streetlights hamesha on nahi rehti; jahan traffic hai wahan on hoti hain. Real research: Mixture-of-Experts, Mixture-of-Depths.
3. **Symbolic lattice** (precision/truth) — ek discrete layer jo facts, logic, math ko **exact** rakhta hai, approximate nahi. Analogy: calculator vs mental math. Neural net "lagbhag 4000" bolta hai; symbolic layer "exactly 4096" bolta hai. **Yeh hallucination ka main killer hai.**

---

## Pillar 4: Externalized Knowledge — "The Ontological Store" 📚

Sabse practical pillar. The big unlock:

> **Skills weights mein rahen. Knowledge disk pe rahe.**

Aaj ke models mein "France ki capital Paris hai" jaisi facts weights ke andar baked hain — isliye woh itne heavy hain. Hum facts ko **disk pe** compressed knowledge store mein rakhenge (concepts + relations + embeddings), runtime pe retrieve karenge — lekin typical RAG jaisa bolted-on nahi. **RETRO-style** (DeepMind ka proven research): retrieval forward pass ke *andar* hota hai, model ko internal memory jaisa feel hota hai.

**Analogy:** Ek brilliant doctor sab textbooks ratta nahi maarta. Woh *reasoning* seekhta hai aur reference books paas rakhta hai. Chhota dimaag + badi library > bada dimaag jo sab ratta maare.

**Bonus — Predictive Prefetch:** model guess karta hai agla kya knowledge chahiye hoga aur pehle se load kar leta hai (jaise Netflix next episode pre-buffer karta hai). Disk latency invisible ho jaati hai.

---

## Pillar 5: Bounded Self-Evolution + Homeostasis 🧬

**Kya hai:** Har session ke baad model chhote, safe weight-updates karta hai — teri style, tera domain seekhta hai. Full retraining kabhi nahi.

**Safety mechanism (Homeostasis Controller):** Jaise tera body temperature ko 37°C pe regulate karta hai, ek controller model ki "health metrics" (stability, coherence) monitor karta hai. Har change ka **budget** hai, har change **rollback-able** hai, aur har change ko frozen test suite pass karna padta hai commit hone se pehle. Fail = automatic undo.

**Real research:** Continual learning (EWC), Test-Time Training layers (2024). Humara addition: Lyapunov-bounded delta budget (math jo guarantee kare ki system chaos mein na jaye).

---

## Full Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│  ETERNAL SEED (~200 MB)                              │
│  Model ka DNA — weight-generation rules              │
└──────────────┬──────────────────────────────────────┘
               │ weights on-demand generate karta hai
┌──────────────▼──────────────────────────────────────┐
│  RESONANCE MANIFOLD (~2.5 GB active)                 │
│  SSM backbone, multi-resolution, sparse activation   │
│  ┌─────────────┐        ┌──────────────────────┐    │
│  │ FAST STREAM │  ⇄     │ SLOW STREAM          │    │
│  │ (intuition) │        │ (reasoning, what-if  │    │
│  │ System 1    │        │  simulation) Sys 2   │    │
│  └─────────────┘        └──────────────────────┘    │
└───────┬─────────────────────────────┬───────────────┘
   anchor karta hai              retrieve karta hai
┌───────▼──────────┐    ┌────────────▼────────────────┐
│ SYMBOLIC LATTICE │    │ ONTOLOGICAL STORE (DISK pe) │
│ (~0.5 GB)        │    │ Saare facts & memories.     │
│ Exact facts,     │    │ Predictive prefetch pehle   │
│ logic, math      │    │ se RAM mein stream karta hai│
└──────────────────┘    └─────────────────────────────┘

HOMEOSTASIS CONTROLLER: hamesha on — stability monitor,
energy budget, self-evolution ka gatekeeper.
```

## 8 GB Memory Budget

| Component | Budget |
|---|---|
| Eternal Seed + hypernetwork | 0.2 GB |
| Active manifold (generated weights, working set) | 2.5 GB |
| State (SSM state context mein O(1) hai!) | 1.0 GB |
| Symbolic lattice + router | 0.5 GB |
| Prefetch buffers + ontology cache | 1.5 GB |
| OS + runtime headroom | 2.3 GB |
| **Total** | **8.0 GB ✅** |

## Naming Continuity (humari discussion → final)

| Discussion ka naam | Final naam |
|---|---|
| Procedural HyperFractal Core (AetherNet) | Eternal Seed |
| Resonance Chambers (NexusCore) | Resonance Manifold |
| Dual-Stream (Elysium) | Fast/Slow Stream |
| Ontological Compression Engine (Vyomarudra) | Ontological Store |
| Sheaf Cohomology / Categorical Resonance | Open research track — critical path pe NAHI (abhi poetry zyada, math kam) |
