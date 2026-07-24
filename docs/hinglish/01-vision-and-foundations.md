# 01 — Vision & Foundations (Hinglish)

## Vision

Frontier-level intelligence (300B+ effective scale) ko ek normal 8GB Mac pe chalana — model ko chhota karke nahi, balki architecture ko fundamentally rethink karke.

---

## Basics Samajh (simple explanation)

### Parameter kya hota hai?

Ek LLM basically ek giant math function hai. **Parameters** (ya "weights") woh numbers hain jo function ka behavior decide karte hain — jaise ek mixing board pe millions of knobs.

- **Training** = knobs ko sahi position pe set karna.
- **Inference** (chat karna) = un knobs ko *read* karke calculation karna.

**Problem:** Har knob RAM mein rakhna padta hai, kyunki har token generate karte waqt unhe read karna hota hai.

| Model scale | FP16 weights | 4-bit quantized | 8 GB mein fit? |
|---|---|---|---|
| 3B dense | 6 GB | ~1.8 GB | ✅ |
| 8B dense | 16 GB | ~4.5 GB | ⚠️ tight |
| 70B dense | 140 GB | ~38 GB | ❌ |
| 300B (MoE) | 600 GB | ~150 GB | ❌ |

300 billion knobs × 2 bytes = **600 GB**. Humare paas 8 GB hai. Yeh 75x ka gap hai.

### Quantization kya hai aur kyun kaafi nahi?

Quantization = har knob ko kam precision se store karna (16-bit ki jagah 4-bit). Jaise photo compress karna — quality thodi girti hai, size 4x kam.

300B params @ 4-bit = **150 GB**. Abhi bhi 19x door. Quantization akele kabhi kaafi nahi hoga.

### Information-theory ki hard limit (isme honest rehna zaroori hai)

8 GB ≈ 6.8×10¹⁰ bits. 300B parameters ko 1 bit each pe bhi store karne ke liye 3×10¹¹ bits chahiye. Matlab **lossless** 300B model 8 GB mein mathematically impossible hai. Jo koi ulta claim kare, woh kuch bech raha hai.

### The Key Insight (yeh ek line pura project hai)

Research baar-baar dikhati hai ki transformer weights mein **90%+ redundancy** hai (pruning, distillation, low-rank papers). Woh 300B knobs actually 300B independent choices NAHI hain — unme deep patterns hain.

> **Toh hum knobs store nahi karenge. Hum woh RULES store karenge jo knobs GENERATE karte hain.**

**Analogy:** Minecraft ki puri duniya (terabytes ka terrain) disk pe stored nahi hoti. Ek chhota **seed number** + generation algorithm stored hota hai, aur terrain on-demand generate hota hai jab tu wahan pahunchta hai. Hum model weights ke saath yehi karenge.

### Target precisely bola jaye toh

**Capability parity, not weight parity.** 300B transformer jitna *smart* system, na ki utna *heavy*. Yeh subtle difference hi pura project hai.

### Brain kyun kehta hai yeh possible hai

Insaan ka brain: ~86 billion neurons, sirf ~20 watts, aur uska "blueprint" (DNA) sirf ~750 MB hai. DNA neurons store nahi karta — neurons *banane ke rules* store karta hai. Nature ne already prove kar diya ki intelligence generative rules mein compress hoti hai. Hum nature ki trick copy kar rahe hain.
