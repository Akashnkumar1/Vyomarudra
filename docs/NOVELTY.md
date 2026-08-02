# Vyomarudra — Novelty Analysis (for IP / publication decisions)

An honest, claim-by-claim map of what is ours, what is prior art, and what a patent
attorney should actually evaluate. Written to be *useful in an attorney conversation*,
not to flatter the project. I am not a lawyer; a formal novelty search is the
correct next step, and this document exists to make that search cheap and precise.

---

## The standard being applied

A patent requires **novelty** (not previously disclosed) and **non-obviousness**
(not an obvious combination to a skilled practitioner). Combining known techniques
is usually held obvious *unless* the combination yields a surprising, non-predictable
result. Software patents additionally face subject-matter limits that vary sharply
by jurisdiction (notably tighter in the EU and India than the US).

---

## Claim-by-claim

### 1. Sparse top-1 MoE with load-balancing — **prior art**
Shazeer et al. 2017 (sparsely-gated MoE); Fedus et al. 2021 (Switch Transformer,
top-1 routing + auxiliary load-balance loss). Our implementation follows Switch
closely, including the `E · Σ f_e · P_e` auxiliary term.

### 2. State-space backbone — **prior art**
S4 (Gu et al. 2021), S4D, Mamba (2023). Our diagonal recurrence with
`A = σ(a) ∈ (0,1)` is a standard stable parameterisation.

### 3. SSM + MoE hybrid — **prior art**
Jamba (AI21, 2024) combines Mamba layers with MoE. The combination is published.

### 4. Integer weight quantization (int8 / int4) — **prior art**
LLM.int8() (Dettmers et al. 2022), GPTQ, AWQ; sub-4-bit and ternary work (BitNet).
Per-tensor symmetric quantization with a scale factor is textbook.

### 5. Keeping weights quantized *in RAM* with a fused dequant-in-kernel — **prior art**
This is the core design of **llama.cpp / GGML**: GGUF weights stay quantized in
memory and are unpacked inside hand-written SIMD dot-product kernels (Q4_0, Q8_0,
etc.). Our int4 nibble-packing with in-loop unpacking is the same idea, implemented
independently and less optimally (we are ~1.2× slower than the framework's f32 BLAS;
llama.cpp is faster than fp32).
*I initially thought this might be our distinctive piece. On examination it is not.*

### 6. Expert offloading / demand paging to disk — **prior art**
DeepSpeed-Inference and ZeRO-Infinity (parameter offload), FlexGen, and a specific
line of MoE-offloading systems (Mixtral-offloading, MoE-Infinity, Fiddler) that
page experts from CPU/SSD with LRU caching — precisely our `PagedExperts` design.

### 7. Retrieval-augmented LM with an external store — **prior art**
RETRO (Borgeaud et al. 2022), kNN-LM (Khandelwal et al. 2020), DPR.

### 8. Contrastive retriever with out-of-domain negatives — **prior art**
Hard-negative mining is standard in dense retrieval (DPR, ANCE). Using a foreign
corpus as negatives is a known technique.

### 9. Learned abstention / grounding head — **prior art in substance**
Selective prediction and answerability classification are established
(SQuAD 2.0-style unanswerability, calibration heads for RAG). Our specific feature
construction `[q ; e ; q⊙e]` is a conventional pair-interaction encoding.

### 10. Hypernetwork / generative weights — **prior art, and we measured it failing**
Ha et al. 2016. Our fractal addressing variant is a mild extension, and it **lost**
on language (BPB 2.957 generated vs 2.718 stored). Not a basis for a claim.

---

## What is genuinely ours

**None of the above mechanisms is novel.** What we have that others do not:

**A. Empirical contributions (publishable, not patentable)**
1. **The data-ceiling curve for expert count.** Measured at two corpus sizes: the
   marginal benefit of doubling experts is 0.003 BPB at 60 MB and 0.023 BPB at
   500 MB — an ~8× increase in payoff from 2.5× the data, with the useful expert
   count roughly doubling. I have not seen this specific characterisation published.
2. **Quantization cost measured on one controlled axis**: +0.01% (int8) and +0.09%
   (int4) BPB, same checkpoint, same held-out text.
3. **Redundancy governs generative compressibility** — 423× on image classifiers,
   failure on language FFNs, failure on SSM recurrence, failure on MoE experts.
   A consistent, measured principle across four target types.

**B. Engineering artefact (copyright, already yours)**
A single ~4,800-line Rust codebase implementing all five subsystems with no
external model weights and no Python. This is unusual and has real value — but
copyright protects it automatically, and copyright is not a patent.

**C. Negative results**
Documented failures (§5 of PAPER.md) that constrain the design space. Genuinely
under-published in the field; genuinely not patentable.

---

## Honest assessment

The strongest *system* claim would be something like: *"a language model in which
sparse-routed experts are stored on disk in sub-byte integer precision, paged on
demand into a bounded cache, and multiplied without dequantization, with retrieval-
grounded abstention."* Each element is prior art; the assembly is unusual but is
the direction multiple groups are actively converging on (llama.cpp quantization +
MoE offloading + RAG). An examiner would very likely hold the combination obvious.

**Where the value actually is:** priority and credibility via publication, and the
codebase itself. A trillion-parameter working-set projection of 0.48 GB, backed by
measured component costs and honest negative results, is a strong arXiv submission
and a strong artefact to show anyone considering funding Tier-2/3 compute. It is
not, on my reading, a patent portfolio.

**Recommended next step:** take this document plus `PAPER.md` to one patent attorney
for a **novelty search only** (typically a few hundred dollars). Let a professional
overrule me if they see a claim I have missed — that is a cheap, bounded way to test
the question, and far better than filing in multiple jurisdictions on optimism.

**Do not delay publication waiting on this.** In most jurisdictions public disclosure
starts a grace period or bars filing outright; an attorney can advise on sequencing.
But an unpublished result that someone else publishes first is worth nothing either
way.
