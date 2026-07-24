# 03 — Honest Risk Register (Hinglish, No Sugarcoating)

Har project jo breakthrough claim karta hai, usko pehle list karna chahiye ki woh kaise MAR sakta hai. Yeh humari list hai.

| # | Risk | Kitna dangerous | Plan | Kill signal (kab is path ko chodna hai) |
|---|------|----------------|------|------------------------------------------|
| 1 | Weight generation 1000x compress na ho paye (sirf 10–50x tak jaye) | 🔴 **Project-killer** | Sabse PEHLE test karo (Experiment E1). Fallback: hybrid — 20x generation + 4-bit storage + retrieval | Generated 1B model, dense-1B ke 70% se kam score kare 20x compression pe |
| 2 | Training unstable ho (recurrent/field systems diverge karte hain) | 🔴 High | Pretrained Mamba se start, scratch se nahi. Stability math (spectral norm, Lyapunov regularizer) day-1 se | Loss divergence rate baseline se 3x zyada across seeds |
| 3 | Disk knowledge slow/incoherent ho | 🟠 Medium | RETRO proven hai — yeh mostly engineering hai | Retrieval-augmented chhota model knowledge tasks pe 8B dense ko match na kar paye |
| 4 | Weight generation itni slow ho ki tokens/sec mar jaye | 🟠 Medium | Two-tier: fast approximation hamesha, full generation sirf jab zaroori. Hot blocks cache | M-series pe <5 tok/s at 3B virtual scale |
| 5 | Self-evolution model ko drift/corrupt kar de | 🟠 Medium | Phase 3 tak DISABLED. Delta budget + rollback + eval-gate | Deltas ke baad frozen eval suite pe koi bhi regression |
| 6 | Training ke liye frontier-level compute chahiye | 🟡 Real | **Distillation**: scratch se mat sikhao — existing frontier models se sikhwao (teacher → student). ~100x sasta | — |
| 7 | Emergence unpredictable/undebuggable ho | 🟡 Real | Symbolic lattice ko interpretable spine rakho; probes mein early invest karo | — |

## Sachchi Expectation

- **Success** = same-quality transformers se 3–7x efficiency gain. Yeh akela bhi major result hoga.
- **10x+** tabhi jab Risk #1 favorably resolve ho (fractal compression wala bet).
- "Frontier model ko 8GB mein beat" = **north star** (chalne ki direction), **milestone nahi** (date ke saath promise nahi).

## Kill Signals Kyun Important Hain

Bina kill signals ke visionary project ek religion ban jata hai. Upar har risk ke saath concrete, measurable signal hai jo bolta hai "yeh path dead hai, pivot karo." Kisi idea ka kaam NA karna bhi — saste mein, jaldi pata chal jaye toh — valuable contribution hai. Phase 0 exactly isi ke liye hai.

## Humari Saari Design Iterations ka Lesson

AetherNet → NexusCore → Elysium → Vyomarudra — har round ka "abhi bhi bottleneck lag raha hai" ne same cheez sikhaayi:

> Beautiful-on-paper systems concept mein nahi marte — training stability, latency, aur debuggability mein marte hain. Isliye roadmap exactly in teeno ko front-load karta hai.
