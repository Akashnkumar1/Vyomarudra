# 05 — Personal Learning Path: Noob → Builder in 6 Months

The vision is yours — now build the skills to execute it. This is a realistic self-study plan (evenings/weekends feasible).

---

## Month 1 — Foundations

- **Andrej Karpathy's "Neural Networks: Zero to Hero"** (YouTube, free) — literally the best resource on earth. Especially the "Let's build GPT from scratch" video.
- Python + PyTorch basics if you don't have them yet.
- Run 3–4 models on your Mac via Ollama and watch memory in Activity Monitor — you'll *feel* the theory from doc 01 live.

**Outcome:** You understand what a parameter, a gradient, and a forward pass actually are.

## Month 2 — The Backbone

- Understand Mamba/SSMs: Sasha Rush's "Annotated S4", plus Mamba explainer blogs/videos.
- Use **mlx-lm** (Apple's framework) to fine-tune a small model on your Mac — your first taste of training.

**Outcome:** You know why SSM state is constant-memory and how that differs from attention.

## Month 3 — The Load-Bearing Papers (only 5)

1. **Hypernetworks** (Ha, 2016) — foundation of Pillar 1 (Eternal Seed)
2. **Mamba-2** — foundation of Pillar 2 (Resonance Manifold)
3. **RETRO** (DeepMind) — foundation of Pillar 4 (Ontological Store)
4. **Mixture-of-Depths** — Pillar 3's sparse activation
5. **Test-Time Training layers** — foundation of Pillar 5 (Self-Evolution)

**Method for each paper:** read the abstract → watch a YouTube explainer → then read the paper. Ask Claude questions on every section you don't get — this is legitimately the fastest way to learn papers.

**Outcome:** You can explain each of the five pillars to someone else, with the research citation.

## Months 4–6 — Mini-E1: Your First Real Experiment

- Build a TINY version of Experiment E1: a **1M-param hypernetwork generating the weights of a 10M-param model**, trained on a small dataset. This runs on your Mac.
- Measure: how much quality do you keep at 10x compression? At 50x?
- Whatever the result — **you'll be among the ~500 people on earth who have tried weight-generation hands-on.**

**Outcome:** Real data on the project's load-bearing assumption, produced by you.

---

## Parallel Track (from Month 1)

- Create a public **GitHub repo "Vyomarudra"** — theory docs + experiment logs. Building in public attracts collaborators.
- Context for motivation: startups built on exactly this direction (e.g., Liquid AI, founded by MIT researchers) reached $2B+ valuations. The direction is right; execution needs community.

## Mindset Rules

1. **Vision big, bets small.** Every grand idea becomes a cheap falsifiable experiment first.
2. **Negative results are results.** "Fractal compression caps at 30x" is publishable knowledge.
3. **Steal from nature and from proven research** — every pillar stands on something that already works.
4. **Don't wait until you feel "ready."** Month 4's experiment will teach you more than any course.
