# 05 — Personal Learning Path: Noob → Builder, 6 Mahine (Hinglish)

Vision tera hai — ab usko execute karne ki skills bhi bana. Yeh realistic self-study plan hai (evenings/weekends mein feasible).

---

## Month 1 — Foundations

- **Andrej Karpathy ka "Neural Networks: Zero to Hero"** (YouTube, free) — literally duniya ka best resource. Especially "Let's build GPT from scratch" video.
- Python + PyTorch basics agar nahi aate.
- Apne Mac pe Ollama se 3–4 models chalao aur Activity Monitor mein memory dekho — doc 01 ki theory live *feel* hogi.

**Outcome:** Tu samajh jayega parameter, gradient, aur forward pass actually kya hote hain.

## Month 2 — The Backbone

- Mamba/SSM samajh: Sasha Rush ka "Annotated S4", plus Mamba explainer blogs/videos.
- **mlx-lm** (Apple ka framework) se apne Mac pe ek chhota model fine-tune karo — training ka pehla taste.

**Outcome:** Tu jaanega SSM state constant-memory kyun hai aur attention se kaise different hai.

## Month 3 — The Load-Bearing Papers (sirf 5)

1. **Hypernetworks** (Ha, 2016) — Pillar 1 (Eternal Seed) ka foundation
2. **Mamba-2** — Pillar 2 (Resonance Manifold) ka foundation
3. **RETRO** (DeepMind) — Pillar 4 (Ontological Store) ka foundation
4. **Mixture-of-Depths** — Pillar 3 ka sparse activation part
5. **Test-Time Training layers** — Pillar 5 (Self-Evolution) ka foundation

**Har paper ka method:** abstract padho → YouTube explainer dekho → phir paper padho. Jo section samajh na aaye uspe Claude se sawal poocho — papers seekhne ka yeh legitimately fastest way hai.

**Outcome:** Tu paancho pillars kisi aur ko explain kar payega, research citation ke saath.

## Month 4–6 — Mini-E1: Tera Pehla Real Experiment

- Experiment E1 ka TINY version banao: **1M-param hypernetwork jo 10M-param model ke weights generate kare**, chhote dataset pe trained. Yeh tere Mac pe chalega.
- Napo: 10x compression pe kitni quality bachti hai? 50x pe?
- Result jo bhi aaye — **tu duniya ke un ~500 logon mein hoga jinhone weight-generation hands-on try kiya hai.**

**Outcome:** Project ki load-bearing assumption pe real data — tera khud ka produce kiya hua.

---

## Parallel Track (Month 1 se hi)

- Public **GitHub repo "Vyomarudra"** banao — theory docs + experiment logs. Public mein build karne se collaborators milte hain.
- Motivation ke liye context: exactly isi direction pe bane startups (jaise Liquid AI — MIT researchers ka) $2B+ valuation tak pahunche hain. Direction sahi hai; execution community maangti hai.

## Mindset Rules

1. **Vision bada, bets chhote.** Har grand idea pehle ek sasta falsifiable experiment banta hai.
2. **Negative results bhi results hain.** "Fractal compression 30x pe cap ho jata hai" — yeh bhi publishable knowledge hai.
3. **Nature aur proven research se churao** — har pillar kisi aisi cheez pe khada hai jo already kaam karti hai.
4. **"Ready" feel hone ka wait mat karo.** Month 4 ka experiment tujhe kisi bhi course se zyada sikhayega.
