# Vyomarudra: A Sparse, Disk-Resident Language Model Architecture with a Sub-Gigabyte Working Set

**Draft — v0.1**

---

## Abstract

We present Vyomarudra, a language-model architecture implemented from scratch in
Rust that decouples model *capacity* from *resident memory*. The system combines a
diagonal state-space backbone, a sparse top-1 mixture-of-experts feedforward layer
whose experts are held on disk in integer precision and demand-paged into a small
LRU cache, an externalised knowledge store with a learned retriever, and a learned
grounding classifier that abstains when the store does not support a query. We
implement our own int8/int4 matrix kernels because the host tensor framework
provides no signed-integer dtype, which otherwise forces a 4× memory expansion at
load time.

On real web text (FineWeb) we measure: (i) expert count can be scaled 16× (4→64)
for a 0.65% increase in active parameters; (ii) int8 and int4 weight quantization
cost **+0.01%** and **+0.09%** bits-per-byte respectively; (iii) the combination
yields a projected **0.48 GB** working set for a ~999B-parameter configuration,
against 1.98 GB for the same configuration in fp32. We further report a
reproducible *data ceiling*: additional experts stop improving quality past ~32
experts on 60 MB of text and ~64 experts on 500 MB, with the marginal benefit of
doubling expert count growing ~8× when the corpus grows 2.5×.

We report negative results with equal prominence, including the failure of
generative (hypernetwork) weight synthesis on language, the failure of an
uncalibrated similarity threshold as a grounding signal, and the failure of
instruction tuning to produce generalising behaviour at small scale.

---

## 1. Introduction

The dominant constraint on running large language models locally is not arithmetic
but **resident memory**: parameters must be in RAM to be multiplied. A trillion
parameters cannot be resident in 8 GB — at one bit per parameter that is still
116 GB. Any claim of "trillion-parameter capability in 8 GB" must therefore rest on
one of two mechanisms: generating parameters from a compact seed, or keeping most
parameters out of RAM.

We evaluated both. Generative weight synthesis (§5.1) succeeds on redundant image
classifiers (423× compression) but **fails on language**, so we discarded it as a
load-bearing mechanism. The second path — sparse activation plus demand paging —
succeeds, and forms the basis of the architecture reported here.

**Contributions.**
1. An end-to-end architecture in which expert parameters are disk-resident,
   integer-quantized, and paged on demand, with a measured sub-gigabyte working set
   at trillion-parameter scale.
2. Custom int8/int4 kernels that keep quantized weights quantized *in memory*,
   with measured quality cost of +0.01% / +0.09% BPB.
3. An empirical characterisation of the *data ceiling* on expert count, measured at
   two corpus sizes.
4. A learned grounding head that improves supported/unsupported discrimination from
   74.0% (tuned cosine threshold) to 86.6% balanced accuracy.
5. A set of documented negative results that constrain the design space.

---

## 2. Architecture

Five components, each implemented from scratch (≈4,800 lines of Rust; no external
model weights are used at any point).

**2.1 Backbone.** A diagonal state-space layer with a Lyapunov-stable recurrence
`h_t = σ(a)·h_{t-1} + B·u_t`, read out as `y = C·h + D·u`. Input projections are
batched across timesteps; only the recurrence remains sequential.

**2.2 Sparse experts.** Each block routes every token to its top-1 expert of `E`
via a learned gate, with a Switch-style load-balance auxiliary loss. Training uses
*gathered* computation: tokens are grouped by routed expert and each expert runs
once on its own tokens (§4.4).

**2.3 Disk-resident expert store.** Experts are serialised to a container format
(`VYST`) as int8. At inference a pager seeks and reads only the routed expert's
byte range, decodes into an LRU cache of configurable capacity, and never
materialises the full expert set. Capacity therefore scales with *disk*, not RAM.

**2.4 Externalised knowledge.** A contrastively-trained encoder (the same SSM
backbone) embeds passages into an on-disk store with int8 keys. Retrieval is cosine
nearest-neighbour. Crucially, the encoder is trained with **out-of-domain
negatives** drawn from a foreign corpus (§5.2).

**2.5 Grounding head.** A small MLP over `[q ; e ; q⊙e]` — query embedding,
retrieved match, and their elementwise interaction — predicts whether the store
supports the query, replacing a scalar similarity threshold.

---

## 3. Quantized kernels

The host framework (candle) exposes `U8, U32, I64, BF16, F16, F32, F64` — no signed
int8. Quantized weights therefore had to be dequantized to f32 on load, giving int8
storage but 4× that in RAM, which defeats the purpose.

We implement the kernel directly: f32 activations, integer weights, f32 accumulator,
with the scale applied once to the completed dot product. int4 packs two signed
4-bit values per byte, unpacked in the inner loop. Two optimisations proved
necessary: four independent accumulators (a single accumulator serialises on a
floating-point dependency chain and blocks vectorisation) and thread-level
parallelism over output rows, computing a transposed result so threads own disjoint
contiguous slices.

---

## 4. Results

All language results use byte-level BPE (vocab 4096) and are reported in
bits-per-byte (BPB) on held-out text. Lower is better.

### 4.1 Capacity scales, active compute does not

FineWeb, 60 MB, 30k steps, d_model=384, d_ff=512:

| experts | BPB | total params | active params |
|---|---|---|---|
| 4 | 1.768 | 4.73 M | 3.547 M |
| 16 | 1.701 | 9.46 M | 3.552 M |
| 32 | 1.687 | 15.78 M | 3.558 M |
| 64 | 1.684 | 28.40 M | 3.570 M |

A 6× increase in total parameters costs **0.65%** additional active parameters.

### 4.1b Depth versus width at matched parameters

500 MB corpus, 40k steps, ~53.6 M parameters in every row:

| config | BPB |
|---|---|
| 64 experts × 1 layer (28.4 M) | 1.917 |
| 128 experts × 1 layer | 1.923 |
| **64 experts × 2 layers** | **1.898** |
| 32 experts × 4 layers | 2.002 |

Depth is non-monotonic with an optimum at 2 layers for this budget. Width alone is
exhausted (128 vs 64 experts differ by 0.001); shallow-and-wide is not optimal.

### 4.2 The data ceiling

Extending to a 500 MB corpus (40k steps):

| experts | BPB | marginal gain |
|---|---|---|
| 32 | 1.947 | — |
| 64 | 1.924 | 0.023 |
| 128 | 1.923 | 0.001 |

Comparing the *marginal gain of doubling experts* across corpora — the only
quantity comparable across different held-out sets — gives:

| corpus | gain from doubling experts | ceiling |
|---|---|---|
| 60 MB | 0.003 (32→64) | ~32 experts |
| 500 MB | 0.023 (32→64) | ~64 experts |

A 2.5× larger corpus increased the marginal benefit of identical extra capacity by
approximately **8×** and roughly doubled the useful expert count. Capacity is only
convertible into quality in proportion to available data.

> **Methodological note.** Absolute BPB is *not* comparable across corpora: the
> held-out split is the tail of the corpus, so changing the corpus changes the test
> distribution. We report within-corpus differences only. An earlier version of this
> work compared absolute BPB across corpora and drew an incorrect conclusion.

### 4.3 Quantization is nearly free

Identical checkpoint, identical held-out text:

| weights | BPB | cost |
|---|---|---|
| f32 (reference) | 2.1721 | — |
| int8 | 2.1723 | +0.01% |
| int4 | 2.1742 | +0.09% |

### 4.4 Sparse training compute

Applying every expert to every token and masking is correct but wasteful. Gathered
computation (300 steps, d_model=64):

| experts | dense | gathered | BPB dense | BPB gathered |
|---|---|---|---|---|
| 4 | 18.6 s | 11.7 s | 2.482 | 2.468 |
| 32 | 78.2 s | 13.7 s | 2.474 | 2.470 |

Dense cost scales linearly in expert count; gathered cost is approximately flat.

### 4.5 Working set

Projection for d_model=4096, d_ff=14336, vocab=32k, 8500 experts (~999B parameters),
LRU cache of 2 experts, using measured per-component costs:

| configuration | working set |
|---|---|
| all f32 | 1.98 GB |
| int8 experts + int8 backbone | 0.59 GB |
| int4 experts + int8 backbone | **0.48 GB** |

Resident memory is approximately invariant to expert count: 241B → 999B parameters
moves the working set from 1.88 GB to 1.98 GB (f32 experts), because additional
experts occupy disk rather than RAM.

### 4.6 Externalised knowledge

Synthetic key→value facts, fixed model capacity:

| #facts | memorise | retrieve |
|---|---|---|
| 200 | 98.1% | 97.2% |
| 1,000 | 15.4% | 97.3% |
| 16,000 | 11.7% | 97.0% |

Memorisation collapses to chance as facts exceed model capacity; retrieval is flat.

### 4.7 Grounding

| gate | balanced accuracy |
|---|---|
| tuned cosine threshold | 74.0% |
| learned head | **86.6%** |

Held-out, 800 samples. Training the retriever with out-of-domain negatives was a
precondition: without them, no statistic separated in-domain from out-of-domain
queries at all (§5.2).

---

## 5. Negative results

We report these because they constrain the design space more sharply than the
positive results.

**5.1 Generative weight synthesis fails on language.** A fractal/hypernetwork seed
achieves 423× compression on MNIST-class targets and *improves* with target size.
On language feedforward weights the same mechanism merely ties a same-footprint
dense layer; on state-space recurrence weights it is strictly worse than a small
stored model; and generating MoE experts produced BPB 2.957 against 2.718 for stored
experts and 2.899 for a single small dense FFN. The governing variable is target
redundancy: image classifiers are redundant, language and recurrence weights are
not. We therefore store and quantize experts rather than generating them.

**5.2 An uncalibrated similarity threshold is not a grounding signal.** With a
retriever trained only on in-domain negatives, an out-of-domain prompt scored *higher*
(cos 0.682) than a correct in-domain match (0.643), and a calibration-free z-score
against the store distribution also failed (3.26σ vs 3.31σ). The cause is not the
statistic but the training objective: in-batch negatives teach "which passage",
never "is this my domain". Out-of-domain negatives make the question learnable.

**5.3 Small-sample calibration flatters.** Seven hand-picked probes suggested clean
separation; a 400-sample calibration showed overlapping distributions and 74%
balanced accuracy. Separately, a grounding head trained on fixed-offset fragments
scored 88.9% held-out but rejected genuine in-domain prompts in live use; retraining
with varied positives *lowered* the held-out figure to 86.6% while fixing behaviour.

**5.4 Instruction tuning learns format, not semantics, at small scale.** With 126
instruction pairs and 462K parameters, the model correctly emits turn markers and
stops appropriately, but out-of-distribution prompts retrieve unrelated memorised
answers ("What is a submarine?" → an answer about fossils).

**5.5 Depth has an optimum, and a premature conclusion was published.** Until late
in this work the entire model was one block deep. With multi-layer support, at a
matched ~53.6 M parameters: 2 layers × 64 experts reaches **1.898 BPB** (best result
in this work), 1 layer × 128 experts 1.923, and 4 layers × 32 experts 2.002. The
curve is non-monotonic — some depth helps, more hurts at fixed optimisation budget.

We record a process failure alongside it: the conclusion "depth loses to width" was
written and committed from the 4-layer point alone while the 2-layer run was still
training, and had to be retracted. Single-sample curves should not be published,
particularly when the remaining samples are already in flight.

---

## 6. Related work and prior art

The individual mechanisms here are **not novel**, and we make no such claim:

- **Mixture-of-experts with top-1 routing and load-balancing:** Shazeer et al.
  (2017); Switch Transformer, Fedus et al. (2021).
- **State-space sequence models:** S4 (Gu et al., 2021), S4D, Mamba (2023).
- **Post-training integer quantization:** LLM.int8() (Dettmers et al., 2022), GPTQ,
  AWQ; sub-4-bit and ternary variants (BitNet).
- **Parameter offloading and layer streaming:** ZeRO-Infinity / DeepSpeed-Inference,
  FlexGen, and expert-offloading systems for Mixtral-class models.
- **Retrieval-augmented language modelling:** RETRO (Borgeaud et al., 2022),
  kNN-LM (Khandelwal et al., 2020), DPR.
- **Hypernetworks:** Ha et al. (2016).

Our contribution is the **integrated system and its measurement**: a single
codebase in which these mechanisms are co-designed (quantized paging with a matching
integer kernel, retrieval with a learned abstention head), and a set of controlled
measurements — including the data-ceiling curve and the quantization cost — taken on
one architecture under one methodology.

---

## 7. Limitations

1. **Scale.** The largest model trained here is 53.6 M parameters. The 999B working-set
   figure is a **projection** from measured per-component costs, not a trained system.
2. **Quality.** Generated text is grammatical but not semantically coherent; the
   model does not follow instructions.
3. **Single-seed results.** Several numbers are single runs. Where seeds were run,
   they are reported (the MoE advantage is +0.110 ± 0.004 over two seeds).
4. **Projection assumptions.** §4.5 assumes an LRU cache of 2 experts and ignores
   activation and KV memory, which are workload-dependent.
5. **Kernel performance.** Our int8 path is ~1.2× slower than the framework's f32
   BLAS after threading (1.81 vs 1.50 ms/token); it is not SIMD-optimised.
6. **Disk cost.** A 999B int4 configuration requires ~0.9 TB of storage. The
   parameters are not eliminated — they are relocated.

---

## 8. Conclusion

Resident memory and model capacity can be decoupled: with top-1 routing, integer
quantization, and demand paging, a trillion-parameter configuration projects to a
0.48 GB working set, and the quantization required costs under 0.1% in bits-per-byte.
The binding constraint on such a system is not memory engineering but **data and
training compute** — capacity converts to quality only in proportion to the corpus,
as our data-ceiling measurements show.

---

## Reproducibility

All code, measurements, and negative results: https://github.com/Akashnkumar1/Vyomarudra
Every figure in this paper corresponds to a logged entry in `docs/PROGRESS.md` with
the exact command used.
