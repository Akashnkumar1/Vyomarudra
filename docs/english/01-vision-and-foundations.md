# 01 — Vision & Foundations

## The Vision

Run frontier-level intelligence (300B+ effective scale) on a normal 8GB consumer Mac — not by shrinking models, but by fundamentally rethinking the architecture.

---

## Foundations (Explained Simply)

### What is a parameter?

An LLM is basically one giant math function. **Parameters** (or "weights") are the numbers that decide the function's behavior — like millions of knobs on a mixing board.

- **Training** = setting the knobs to the right positions.
- **Inference** (chatting) = reading those knobs to do the calculation.

**The problem:** Every knob must sit in RAM, because every generated token needs to read them.

| Model scale | FP16 weights | 4-bit quantized | Fits in 8 GB? |
|---|---|---|---|
| 3B dense | 6 GB | ~1.8 GB | ✅ |
| 8B dense | 16 GB | ~4.5 GB | ⚠️ tight |
| 70B dense | 140 GB | ~38 GB | ❌ |
| 300B (MoE) | 600 GB | ~150 GB | ❌ |

300 billion knobs × 2 bytes = **600 GB**. We have 8 GB. That's a 75x gap.

### What is quantization, and why isn't it enough?

Quantization = storing each knob with less precision (4-bit instead of 16-bit). Like compressing a photo — small quality loss, 4x smaller size.

300B params @ 4-bit = **150 GB**. Still 19x too big. Quantization alone will never close the gap.

### The information-theory hard limit (be honest about this)

8 GB ≈ 6.8×10¹⁰ bits. Storing 300B parameters at even 1 bit each needs 3×10¹¹ bits. Therefore a **lossless** 300B model in 8 GB is mathematically impossible. Anyone who claims otherwise is selling something.

### The Key Insight (this one line IS the whole project)

Research repeatedly shows that transformer weights contain **90%+ redundancy** (pruning, distillation, low-rank literature). Those 300B knobs are NOT 300B independent choices — they contain deep patterns.

> **So we don't store the knobs. We store the RULES that generate the knobs.**

**Analogy:** Minecraft's entire world (terabytes of terrain) is not stored on disk. A tiny **seed number** + a generation algorithm is stored, and terrain is generated on demand when you walk there. We want to do the same with model weights.

### Target restated precisely

**Capability parity, not weight parity.** A system as *smart* as a 300B transformer, not as *heavy* as one. This subtle difference is the entire project.

### Why the brain says this is possible

The human brain: ~86 billion neurons, ~20 watts, and its "blueprint" (DNA) is only ~750 MB. DNA doesn't store neurons — it stores the rules for building them. Nature already proved that intelligence compresses into generative rules. We are copying nature's trick.
