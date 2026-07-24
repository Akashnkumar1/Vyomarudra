# 03 — Honest Risk Register (No Sugarcoating)

Every project that claims a breakthrough must first list the ways it dies. Here are ours.

| # | Risk | Severity | Mitigation | Kill signal (when to give up on this path) |
|---|------|----------|------------|--------------------------------------------|
| 1 | Weight generation tops out at 10–50x compression, not 1000x | 🔴 **Project-killer** | Test FIRST (Experiment E1). Fallback: hybrid — 20x generation + 4-bit storage + retrieval | Generated 1B model scores <70% of dense-1B at 20x compression |
| 2 | Training instability (recurrent/field systems diverge) | 🔴 High | Start from pretrained Mamba, not from scratch. Stability math (spectral normalization, Lyapunov regularizer) from day 1 | Loss divergence rate >3x baseline across seeds |
| 3 | Disk-based knowledge too slow / incoherent | 🟠 Medium | RETRO is proven — this is mostly engineering | Retrieval-augmented small model can't match 8B dense on knowledge tasks |
| 4 | Weight-generation latency kills tokens/sec | 🟠 Medium | Two-tier: fast approximation always, full generation only when needed. Cache hot blocks | <5 tok/s on M-series at 3B virtual scale |
| 5 | Self-evolution drifts / corrupts the model | 🟠 Medium | Disabled until Phase 3. Delta budget + rollback + eval-gated commits | Any regression on frozen eval suite after deltas |
| 6 | Initial training needs frontier-scale compute | 🟡 Real | **Distillation**: don't train from scratch — learn from existing frontier models (teacher → student). ~100x cheaper | — |
| 7 | Emergence is unpredictable / undebuggable | 🟡 Real | Keep the symbolic lattice as an interpretable spine; invest in probes early | — |

## Honest Expected Outcome

- **Success** = 3–7x efficiency gain over same-quality transformers. That alone would be a major result.
- **10x+** requires Risk #1 resolving favorably (the fractal compression bet).
- "Beats a frontier model in 8GB" is a **north star** (direction to walk), not a **milestone** (promise with a date).

## Why the Kill Signals Matter

A visionary project without kill signals becomes a religion. Each risk above has a concrete, measurable signal that says "this path is dead, pivot." Finding out an idea DOESN'T work — cheaply and early — is a valuable contribution too. Phase 0 exists exactly for this.

## The Lesson Baked Into All Our Design Iterations

Across AetherNet → NexusCore → Elysium → Vyomarudra, every round of "still feels like a bottleneck" taught the same thing:

> Beautiful-on-paper systems die in training stability, latency, and debuggability — not in concept. So the roadmap front-loads exactly those three.
