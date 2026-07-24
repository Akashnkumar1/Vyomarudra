# 04 — The Roadmap: Phase 0 → Phase 3 (Hinglish)

**Core principle:** Har speculative idea ko sasta test PEHLE. Koi bhi mystical idea critical path pe nahi jab tak experiment prove na kare.

```
M0────M4──────────M12────────────M24──────────────M48
 │ E1–E5 │  MVN 1–3B   │  30–70B-equiv   │  Self-evo → 300B-equiv
 └─ G0 ──┴──── G1 ─────┴────── G2 ───────┴─── G3 ⭐
```

---

## 🔬 Phase 0 — Falsification Experiments (Month 0–4)

**Team:** 1–2 log. **Hardware:** 1 GPU ya Mac. **Scale:** ≤300M params.
Purpose: bure ideas ko jaldi aur saste mein maar do.

### Experiments

- **E1 — Generative Weights (THE experiment):** Ek hypernetwork train karo jo <10M seed se 125M–300M model ke weights generate kare. Compression-vs-quality curve napo. → *Risk #1 (project-killer) ka faisla.*
- **E2 — Externalized Knowledge:** Ek 1–3B model lo, disk store se RETRO-style retrieval add karo. Knowledge-task performance kitni recover hoti hai, napo. → *Risk #3 ka faisla.*
- **E3 — Field Dynamics:** Plain Mamba vs Mamba + multi-scale coupling. Kya "resonance" measurable fayda deta hai, ya sirf poetry hai? → *Pillar 2 ke addition ka faisla.*
- **E4 — Sparse Awakening:** Router jo har token pe N mein se sirf k generated blocks instantiate kare. Quality vs active-params curve napo.
- **E5 — Stability Tooling:** Lyapunov monitor + spectral regularizer library banao. Neeche ka sab kuch ispe depend karta hai.

### Gate G0
E1 mein ≥100x compression at ≤10% quality loss, **YA** viable hybrid path (jaise 20x generation + 4-bit storage + retrieval = credible 300B-class capability path). Dono nahi mile → findings publish karo aur pivot. Documented negative result bhi contribution hai.

---

## 🏗️ Phase 1 — Minimal Viable Nexus (Month 4–12)

**Team:** chhoti. **Hardware:** modest cluster.

- 1–3B *virtual*-scale prototype banao: Eternal Seed + SSM-hybrid backbone + Symbolic Lattice + disk Ontological Store + Dual Stream (fast/slow).
- **Open frontier teacher se distill karo** — frontier-scale training compute avoid hota hai (Risk #6).
- Eval harness ABHI banao: frozen benchmark suite + stability metrics + hallucination probes.

**Deliverable:** 8GB M-series Mac pe ≥15 tok/s; dense 3B (Phi/Qwen class) ko reasoning + coding pe match/beat kare, while "model" ke liye ≤1.5 GB use karte hue.

### Gate G1
Same-RAM dense baseline ko eval suite ke ≥60% pe beat kare.

---

## 📈 Phase 2 — Scale + Crystallization (Month 12–24)

- Virtual capacity ko **30–70B-equivalent** tak scale karo, same 8GB envelope mein (yahan compression ratio ko pay karna padega).
- Add karo: predictive prefetcher, knowledge **crystallization** loop (frequently-used patterns weights mein "jam" ho jayein — jaise bike chalana conscious effort se automatic ban jata hai), slow stream mein causal what-if rollouts, cross-modal encoders.
- Hardware co-design: Metal/MLX custom kernels for on-the-fly weight generation; block-granular caching.

### Gate G2
Dense 30B (Qwen-30B class) ke barabar, 8 GB mein, ≥20 tok/s pe.

---

## 🚀 Phase 3 — Self-Evolution + Productization (Month 24–48)

Yeh phase tabhi exist karta hai agar G2 cleanly pass hua.

- Bounded per-session weight deltas on karo (homeostasis-gated, rollback-able, eval-gated).
- Per-user personalization: har user ko shared seed se apna instantiation.
- Serious safety program: lattice pe interpretability probes, evolution loop ki red-teaming, formal delta budgets.

### Gate G3 (North Star)
8 GB mein reasoning/coding pe frontier-competitive. Capability-per-GB aur capability-per-watt pe contemporary frontier models se 10x efficiency.

---

## IS Month Kya Karna Hai (akele feasible)

1. **Baselines reproduce karo:** Mamba-2 / RWKV / chhota LFM locally chalao; lm-eval-harness working karo.
2. **Mini-E1 start karo:** hypernetwork repo fork karo; 5M seed se 125M GPT ke weights generate karne ka target. Yeh single experiment pure project ki load-bearing assumption test karta hai.
3. **E2 saste mein start karo:** 3B model ko local vector store (sqlite-vec bhi chalega) se RETRO-style wire karo; knowledge recovery napo.
4. **Load-bearing papers padho:** Hypernetworks (Ha 2016) · Mamba-2 · RETRO (DeepMind) · Mixture-of-Depths · Test-Time Training layers · Dreamer v3 · EWC/continual-learning surveys · KAN.
