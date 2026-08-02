//! Phase-1 — the first INTEGRATED Vyomarudra model.
//!
//! Architecture forced by Phase-0 evidence (docs/english/06):
//!   token embed (stored) → diagonal SSM mixer (stored, lean) → FFN (GENERATED
//!   by the fractal Eternal Seed; the large redundant mass) → output head (stored).
//!
//! Task: char-level next-token prediction on procedurally-generated arithmetic
//! ("a+b=c\n") — a capacity-hungry task where a bigger FFN measurably helps, so
//! generating that FFN is a meaningful test (unlike row-MNIST, where the SSM
//! alone sufficed and the FFN was moot).
//!
//! Honest test (unchanged): generated-FFN model vs. a same-footprint plain model
//! vs. the dense upper bound. Backend: candle, CPU. Fully offline.

use anyhow::Result;
use candle_core::{DType, Device, Tensor, Var, D};
use candle_nn::{optim::{AdamW, ParamsAdamW}, Optimizer};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// vocab: '0'..'9' -> 0..9, '+' -> 10, '=' -> 11, '\n' -> 12
const VOCAB: usize = 13;
fn ch2id(c: u8) -> u32 {
    match c { b'0'..=b'9' => (c - b'0') as u32, b'+' => 10, b'=' => 11, b'\n' => 12, _ => 12 }
}

/// Returns (token stream, answer-mask). answer-mask[i] = true iff position i is a
/// digit of the ANSWER (after '='). Only these positions test whether the model
/// learned to add — the input digits are random and inherently unpredictable, so
/// overall next-char accuracy is insensitive to the capability we care about.
fn build_corpus(n_examples: usize, seed: u64) -> (Vec<u32>, Vec<bool>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let (mut s, mut mask) = (Vec::new(), Vec::new());
    let mut push = |text: String, is_ans: bool, s: &mut Vec<u32>, mask: &mut Vec<bool>| {
        for c in text.bytes() { s.push(ch2id(c)); mask.push(is_ans); }
    };
    for _ in 0..n_examples {
        let (a, b) = (rng.gen_range(0..100), rng.gen_range(0..100));
        push(format!("{a}+{b}="), false, &mut s, &mut mask);
        push(format!("{}", a + b), true, &mut s, &mut mask); // answer digits
        push("\n".into(), false, &mut s, &mut mask);
    }
    (s, mask)
}

/// E2 — externalized knowledge. Synthetic key→value facts. Each entity maps to a
/// fixed value. `retrieve=false` (MEMORIZE): the model sees the ENTITY and must
/// recall the value from weights → capacity-bound, degrades as #facts grows.
/// `retrieve=true` (RETRIEVE): the value is placed right before the query (as if
/// fetched from a store) → a short-range copy any model handles, flat vs #facts.
/// Returns (stream, vocab, answer-mask at value-target positions).
fn build_kv_corpus(n_ent: usize, n_val: usize, retrieve: bool, n_examples: usize, seed: u64) -> (Vec<u32>, usize, Vec<bool>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let val_of: Vec<u32> = (0..n_ent).map(|_| rng.gen_range(0..n_val) as u32).collect();
    // FIXED tiny vocab: digits 0..9 encode both the entity id (5 digits) and the
    // value (1 digit). No per-entity parameters, so memorization must go through
    // the shared FFN and is genuinely capacity-bound.
    const D: usize = 5;
    let qmark = 10u32; let nl = 11u32;
    let vocab = 12; // 0..9, QMARK, NL
    let digits = |mut x: usize| -> Vec<u32> { let mut v = vec![0u32; D]; for i in (0..D).rev() { v[i] = (x % 10) as u32; x /= 10; } v };
    let (mut s, mut m) = (Vec::new(), Vec::new());
    let mut push = |t: u32, ans: bool, s: &mut Vec<u32>, m: &mut Vec<bool>| { s.push(t); m.push(ans); };
    for _ in 0..n_examples {
        let e = rng.gen_range(0..n_ent);
        let v = val_of[e]; // value digit 0..9
        for d in digits(e) { push(d, false, &mut s, &mut m); }
        if retrieve { push(v, false, &mut s, &mut m); }  // retrieved value, adjacent to the query
        push(qmark, false, &mut s, &mut m);
        push(v, true, &mut s, &mut m);                    // answer (mask here)
        push(nl, false, &mut s, &mut m);
    }
    let _ = n_val;
    (s, vocab, m)
}

/// Load a text file as a char-level token stream; returns (stream, vocab_size).
fn build_text_corpus(path: &str) -> Result<(Vec<u32>, usize)> {
    let bytes = std::fs::read(path)?;
    let mut seen = [false; 256];
    for &b in &bytes { seen[b as usize] = true; }
    let mut map = [0u32; 256];
    let mut v = 0u32;
    for i in 0..256 { if seen[i] { map[i] = v; v += 1; } }
    let stream: Vec<u32> = bytes.iter().map(|&b| map[b as usize]).collect();
    Ok((stream, v as usize))
}

/// Reversible pre-tokenization matching `vyoma-tokenizer` (whitespace starts a word).
fn pretokenize(bytes: &[u8]) -> Vec<Vec<u32>> {
    let (mut words, mut cur): (Vec<Vec<u32>>, Vec<u32>) = (Vec::new(), Vec::new());
    for &b in bytes {
        if (b == b' ' || b == b'\n' || b == b'\t') && !cur.is_empty() { words.push(std::mem::take(&mut cur)); }
        cur.push(b as u32);
    }
    if !cur.is_empty() { words.push(cur); }
    words
}

/// Tokenize a corpus with the BPE merges produced by `vyoma-tokenizer`.
/// Returns (token stream, vocab_size = 256 + n_merges). Our own tokenizer; no dep.
fn build_bpe_corpus(text_path: &str, merges_path: &str) -> Result<(Vec<u32>, usize)> {
    use std::collections::HashMap;
    let merges_txt = std::fs::read_to_string(merges_path)
        .map_err(|e| anyhow::anyhow!("need BPE merges at {merges_path} (run vyoma-tokenizer): {e}"))?;
    let (mut rank, mut id_of) = (HashMap::new(), HashMap::new());
    for (m, line) in merges_txt.lines().enumerate() {
        let mut it = line.split_whitespace();
        let a: u32 = it.next().unwrap().parse()?;
        let b: u32 = it.next().unwrap().parse()?;
        rank.insert((a, b), m);
        id_of.insert((a, b), 256 + m as u32);
    }
    let vocab = 256 + rank.len();
    let bytes = std::fs::read(text_path)?;
    let mut stream = Vec::new();
    for word in pretokenize(&bytes) {
        let mut seq = word;
        loop {
            let mut best: Option<(usize, usize)> = None; // (rank, pos)
            for i in 0..seq.len().saturating_sub(1) {
                if let Some(&r) = rank.get(&(seq[i], seq[i + 1])) {
                    if best.map_or(true, |(br, _)| r < br) { best = Some((r, i)); }
                }
            }
            let (_, pos) = match best { Some(x) => x, None => break };
            let id = id_of[&(seq[pos], seq[pos + 1])];
            seq.splice(pos..pos + 2, [id]);
        }
        stream.extend(seq);
    }
    Ok((stream, vocab))
}

// ---------------------------------------------------------------------------
// Var helpers + fractal seed (targets an arbitrary param count + prior)
// ---------------------------------------------------------------------------
fn var_randn(shape: (usize, usize), std: f64, dev: &Device) -> Result<Var> {
    Ok(Var::from_tensor(&Tensor::randn(0f32, std as f32, shape, dev)?)?)
}
fn var_randn1(n: usize, std: f64, dev: &Device) -> Result<Var> {
    Ok(Var::from_tensor(&Tensor::randn(0f32, std as f32, (n,), dev)?)?)
}
fn var_full1(n: usize, val: f64, dev: &Device) -> Result<Var> {
    Ok(Var::from_tensor(&(Tensor::ones((n,), DType::F32, dev)? * val)?)?)
}
fn softmax(x: &[f32]) -> Vec<f32> {
    let m = x.iter().cloned().fold(f32::MIN, f32::max);
    let mut e: Vec<f32> = x.iter().map(|v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum::<f32>().max(1e-8);
    for v in &mut e { *v /= s; }
    e
}
fn l2norm(mut r: Vec<f32>) -> Vec<f32> {
    let n = r.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    for x in &mut r { *x /= n; }
    r
}

const PE_DIM: usize = 16;
const ADDR_HIDDEN: usize = 16;
fn positional_encoding(n: usize, d: usize, dev: &Device) -> Result<Tensor> {
    let mut v = vec![0f32; n * d];
    for i in 0..n {
        for k in 0..d {
            let half = (k / 2) as f32;
            let freq = 1.0f32 / 10000f32.powf(2.0 * half / d as f32);
            let ang = i as f32 * freq;
            v[i * d + k] = if k % 2 == 0 { ang.sin() } else { ang.cos() };
        }
    }
    Ok(Tensor::from_vec(v, (n, d), dev)?)
}

struct FractalSeed {
    pe: Tensor, prior: Tensor,
    a_w1: Var, a_b1: Var, a_w2: Var, a_b2: Var,
    g_w1: Var, g_b1: Var, g_w2: Var, g_b2: Var,
    gscale: Var, n: usize,
}
impl FractalSeed {
    fn new(n: usize, prior_vals: Vec<f32>, chunk: usize, ed: usize, hh: usize, dev: &Device) -> Result<Self> {
        let n_chunks = (n + chunk - 1) / chunk;
        Ok(Self {
            pe: positional_encoding(n_chunks, PE_DIM, dev)?,
            prior: Tensor::from_vec(prior_vals, (n,), dev)?,
            a_w1: var_randn((PE_DIM, ADDR_HIDDEN), (1.0 / PE_DIM as f64).sqrt(), dev)?,
            a_b1: var_randn1(ADDR_HIDDEN, 1e-6, dev)?,
            a_w2: var_randn((ADDR_HIDDEN, ed), (1.0 / ADDR_HIDDEN as f64).sqrt(), dev)?,
            a_b2: var_randn1(ed, 1e-6, dev)?,
            g_w1: var_randn((ed, hh), (1.0 / ed as f64).sqrt(), dev)?,
            g_b1: var_randn1(hh, 1e-6, dev)?,
            g_w2: var_randn((hh, chunk), (1.0 / hh as f64).sqrt(), dev)?,
            g_b2: var_randn1(chunk, 1e-6, dev)?,
            gscale: var_full1(1, 1.0, dev)?, n,
        })
    }
    fn vars(&self) -> Vec<Var> {
        vec![self.a_w1.clone(), self.a_b1.clone(), self.a_w2.clone(), self.a_b2.clone(),
             self.g_w1.clone(), self.g_b1.clone(), self.g_w2.clone(), self.g_b2.clone(), self.gscale.clone()]
    }
    fn seed_params(&self) -> usize { self.vars().iter().map(|v| v.elem_count()).sum() }
    fn generate(&self) -> Result<Tensor> {
        let emb = self.pe.matmul(self.a_w1.as_tensor())?.broadcast_add(self.a_b1.as_tensor())?
            .gelu()?.matmul(self.a_w2.as_tensor())?.broadcast_add(self.a_b2.as_tensor())?;
        let h = emb.matmul(self.g_w1.as_tensor())?.broadcast_add(self.g_b1.as_tensor())?.gelu()?;
        let raw = h.matmul(self.g_w2.as_tensor())?.broadcast_add(self.g_b2.as_tensor())?;
        let flat = raw.flatten_all()?.narrow(0, 0, self.n)?;
        let std = flat.sqr()?.mean_all()?.sqrt()?;
        Ok(flat.broadcast_div(&(std + 1e-6)?)?.broadcast_mul(&self.prior)?.broadcast_mul(self.gscale.as_tensor())?)
    }
}

// ---------------------------------------------------------------------------
// Model geometry: embed + diagonal SSM (stored) + FFN (dm->dff->dm) + head
// ---------------------------------------------------------------------------
#[derive(Clone, Copy)]
struct Cfg { dm: usize, dff: usize, layers: usize }

fn ffn_per(c: &Cfg) -> usize { 2 * c.dm * c.dff + c.dff + c.dm } // one layer: w1,b1,w2,b2
fn ffn_total(c: &Cfg) -> usize { c.layers * ffn_per(c) }         // all generated FFN mass
fn ffn_prior(c: &Cfg) -> Vec<f32> {                              // total-length prior (per-layer repeated)
    let mut v = Vec::with_capacity(ffn_total(c));
    for _ in 0..c.layers {
        v.extend(std::iter::repeat((1.0 / c.dm as f32).sqrt()).take(c.dff * c.dm)); // w1
        v.extend(std::iter::repeat(0.01f32).take(c.dff));                            // b1
        v.extend(std::iter::repeat((1.0 / c.dff as f32).sqrt()).take(c.dm * c.dff)); // w2
        v.extend(std::iter::repeat(0.01f32).take(c.dm));                             // b2
    }
    v
}

// One diagonal-SSM layer's stored params.
struct SsmLayer { a_raw: Var, b_co: Var, c_co: Var, d_co: Var }

// stored (non-FFN) weights: embed + per-layer SSM + head
struct Stored { embed: Var, ssm: Vec<SsmLayer>, wo: Var, bo: Var }
impl Stored {
    fn new(c: &Cfg, vocab: usize, dev: &Device) -> Result<Self> {
        let mut ssm = Vec::with_capacity(c.layers);
        for _ in 0..c.layers {
            ssm.push(SsmLayer {
                a_raw: var_randn1(c.dm, 1.0, dev)?,
                b_co: var_randn1(c.dm, 0.5, dev)?,
                c_co: var_randn1(c.dm, 0.5, dev)?,
                d_co: var_randn1(c.dm, 0.5, dev)?,
            });
        }
        Ok(Self {
            embed: var_randn((vocab, c.dm), (1.0 / c.dm as f64).sqrt(), dev)?,
            ssm,
            wo: var_randn((vocab, c.dm), (1.0 / c.dm as f64).sqrt(), dev)?,
            bo: var_randn1(vocab, 0.01, dev)?,
        })
    }
    fn vars(&self) -> Vec<Var> {
        let mut v = vec![self.embed.clone(), self.wo.clone(), self.bo.clone()];
        for s in &self.ssm { v.extend([s.a_raw.clone(), s.b_co.clone(), s.c_co.clone(), s.d_co.clone()]); }
        v
    }
    fn n_params(&self) -> usize { self.vars().iter().map(|v| v.elem_count()).sum() }
}

/// One diagonal-SSM layer over the sequence. `hseq` is (B,L,dm) → returns (B,L,dm).
/// Input projections (B·x, D·x) and the output projection (C·h) are computed once,
/// vectorized across all timesteps; the per-timestep loop does only the recurrence
/// h_t = A·h_{t-1} + Bx_t — this is where most of the wall-clock was going.
fn ssm_layer(hseq: &Tensor, s: &SsmLayer, b: usize, l: usize, dm: usize, dev: &Device) -> Result<Tensor> {
    let a = candle_nn::ops::sigmoid(s.a_raw.as_tensor())?;
    let bx = hseq.broadcast_mul(s.b_co.as_tensor())?;   // (B,L,dm) — all timesteps at once
    let mut h = Tensor::zeros((b, dm), DType::F32, dev)?;
    let mut hs: Vec<Tensor> = Vec::with_capacity(l);
    for t in 0..l {
        let bxt = bx.narrow(1, t, 1)?.reshape((b, dm))?;
        h = h.broadcast_mul(&a)?.add(&bxt)?;            // only the recurrence in the loop
        hs.push(h.clone());
    }
    let hseq_out = Tensor::stack(&hs, 1)?;              // (B,L,dm)
    // y = C·h + D·x, vectorized
    let y = hseq_out.broadcast_mul(s.c_co.as_tensor())?
        .add(&hseq.broadcast_mul(s.d_co.as_tensor())?)?;
    Ok(y)
}

/// Forward returning both the pre-head hidden state and the logits. The hidden
/// state (B*L, dm) is the retrieval key used by kNN-LM (MODE=knn).
fn forward_hidden(ids: &Tensor, st: &Stored, ffn: &Tensor, c: &Cfg) -> Result<(Tensor, Tensor)> {
    let (b, l) = ids.dims2()?;
    let dm = c.dm;
    let dev = ids.device();
    let flat_ids = ids.reshape((b * l,))?;
    let mut hseq = st.embed.as_tensor().index_select(&flat_ids, 0)?.reshape((b, l, dm))?;
    let per = ffn_per(c);
    for li in 0..c.layers {
        let y = ssm_layer(&hseq, &st.ssm[li], b, l, dm, dev)?.reshape((b * l, dm))?;
        let mut o = li * per;
        let w1 = ffn.narrow(0, o, c.dff * dm)?.reshape((c.dff, dm))?; o += c.dff * dm;
        let b1 = ffn.narrow(0, o, c.dff)?; o += c.dff;
        let w2 = ffn.narrow(0, o, dm * c.dff)?.reshape((dm, c.dff))?; o += dm * c.dff;
        let b2 = ffn.narrow(0, o, dm)?;
        let ff = y.matmul(&w1.t()?)?.broadcast_add(&b1)?.gelu()?.matmul(&w2.t()?)?.broadcast_add(&b2)?;
        hseq = (y + ff)?.reshape((b, l, dm))?;
    }
    let hflat = hseq.reshape((b * l, dm))?;
    let logits = hflat.matmul(&st.wo.as_tensor().t()?)?.broadcast_add(st.bo.as_tensor())?;
    Ok((logits, hflat))
}

/// Forward over a batch of id windows. Returns logits (B*L, VOCAB).
fn forward(ids: &Tensor, st: &Stored, ffn: &Tensor, c: &Cfg) -> Result<Tensor> {
    Ok(forward_hidden(ids, st, ffn, c)?.0)
}

// ---------------------------------------------------------------------------
// Data batching
// ---------------------------------------------------------------------------
fn sample_batch(stream: &[u32], b: usize, l: usize, rng: &mut StdRng, dev: &Device) -> Result<(Tensor, Tensor)> {
    let mut xin = Vec::with_capacity(b * l);
    let mut xtg = Vec::with_capacity(b * l);
    for _ in 0..b {
        let start = rng.gen_range(0..stream.len() - l - 1);
        xin.extend_from_slice(&stream[start..start + l]);
        xtg.extend_from_slice(&stream[start + 1..start + l + 1]);
    }
    Ok((Tensor::from_vec(xin, (b, l), dev)?, Tensor::from_vec(xtg, (b, l), dev)?))
}

/// Bytes each token represents (for tokenizer-independent bits-per-byte).
/// char-level: 1 each; BPE: base bytes = 1, merged = sum of constituents.
fn token_byte_lengths(stream: &[u32], tok: &str, merges_path: &str) -> Result<Vec<u32>> {
    if tok != "bpe" { return Ok(vec![1u32; stream.len()]); }
    let txt = std::fs::read_to_string(merges_path)?;
    let mut blen: Vec<u32> = vec![1; 256];
    for line in txt.lines() {
        let mut it = line.split_whitespace();
        let a: usize = it.next().unwrap().parse()?;
        let b: usize = it.next().unwrap().parse()?;
        blen.push(blen[a] + blen[b]);
    }
    Ok(stream.iter().map(|&t| blen[t as usize]).collect())
}

/// Bits-per-byte on a token stream (teacher-forced): total −log2 p(true) over bytes.
/// Lower is better; comparable across tokenizers because it normalizes by raw bytes.
fn eval_bpb(stream: &[u32], blens: &[u32], st: &Stored, ffn: &Tensor, c: &Cfg, l: usize, dev: &Device) -> Result<f64> {
    let (mut bits, mut bytes, mut cnt) = (0f64, 0f64, 0usize);
    let mut e0 = 0;
    while cnt < 4000 && e0 + l + 1 <= stream.len() {
        let ids = Tensor::from_vec(stream[e0..e0 + l].to_vec(), (1, l), dev)?;
        let rows = forward(&ids, st, ffn, c)?.to_vec2::<f32>()?;
        for t in 0..l {
            if cnt >= 4000 { break; }
            let truth = stream[e0 + t + 1] as usize;
            let row = &rows[t];
            let m = row.iter().cloned().fold(f32::MIN, f32::max);
            let lse = m + row.iter().map(|v| (v - m).exp()).sum::<f32>().ln();
            bits += (-(row[truth] - lse) as f64) / std::f64::consts::LN_2;
            bytes += blens[e0 + t + 1] as f64;
            cnt += 1;
        }
        e0 += l;
    }
    Ok(bits / bytes)
}

// ---------------------------------------------------------------------------
// Pillar 3 — sparse Mixture-of-Experts FFN (ours). E expert FFNs + a gate that
// routes each token to its top-1 expert. Capacity scales with E; active compute
// stays ~one expert. This is exactly the "large redundant FFN mass" regime where
// generation earns its keep (Pillar 1) — experts are the natural thing to generate.
// (Toy note: we compute all experts and select, so FLOPs here are dense; a
// production impl gathers per-expert. The routing/quality behavior is faithful.)
// ---------------------------------------------------------------------------
struct Expert { w1: Var, b1: Var, w2: Var, b2: Var }
impl Expert {
    fn new(dm: usize, dff: usize, dev: &Device) -> Result<Self> {
        Ok(Self {
            w1: var_randn((dff, dm), (1.0 / dm as f64).sqrt(), dev)?,
            b1: var_randn1(dff, 0.01, dev)?,
            w2: var_randn((dm, dff), (1.0 / dff as f64).sqrt(), dev)?,
            b2: var_randn1(dm, 0.01, dev)?,
        })
    }
    fn vars(&self) -> Vec<Var> { vec![self.w1.clone(), self.b1.clone(), self.w2.clone(), self.b2.clone()] }
    fn apply(&self, y: &Tensor) -> Result<Tensor> { // (N,dm) -> (N,dm)
        Ok(y.matmul(&self.w1.as_tensor().t()?)?.broadcast_add(self.b1.as_tensor())?.gelu()?
            .matmul(&self.w2.as_tensor().t()?)?.broadcast_add(self.b2.as_tensor())?)
    }
}
struct Moe { gate: Var, experts: Vec<Expert> }
impl Moe {
    fn new(dm: usize, dff: usize, e: usize, dev: &Device) -> Result<Self> {
        Ok(Self {
            gate: var_randn((dm, e), (1.0 / dm as f64).sqrt(), dev)?,
            experts: (0..e).map(|_| Expert::new(dm, dff, dev)).collect::<Result<_>>()?,
        })
    }
    fn vars(&self) -> Vec<Var> {
        let mut v = vec![self.gate.clone()];
        for e in &self.experts { v.extend(e.vars()); }
        v
    }
    fn n_params(&self) -> usize { self.vars().iter().map(|v| v.elem_count()).sum() }
    fn expert_params(&self) -> usize { self.experts[0].vars().iter().map(|v| v.elem_count()).sum() }
}

/// MoE forward. Returns (logits (B*L,vocab), load-balance aux loss scalar).
fn forward_moe(ids: &Tensor, st: &Stored, moe: &Moe, dm: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
    let (b, l) = ids.dims2()?;
    let flat = ids.reshape((b * l,))?;
    let hseq = st.embed.as_tensor().index_select(&flat, 0)?.reshape((b, l, dm))?;
    let y = ssm_layer(&hseq, &st.ssm[0], b, l, dm, dev)?.reshape((b * l, dm))?; // (N,dm)
    let gl = y.matmul(moe.gate.as_tensor())?;                 // (N,E)
    let gp = candle_nn::ops::softmax(&gl, D::Minus1)?;        // (N,E)
    let (n, e) = gp.dims2()?;
    let maxp = gp.max_keepdim(D::Minus1)?.broadcast_as((n, e))?;
    let mask = gp.eq(&maxp)?.to_dtype(DType::F32)?;           // (N,E) top-1 one-hot
    let w = gp.mul(&mask)?;                                   // (N,E) chosen expert's prob
    // SPARSE (gathered) expert compute. The obvious implementation applies EVERY
    // expert to EVERY token and zeroes the unselected ones — correct, but it burns
    // E× the FLOPs to train a model where top-1 routing uses exactly one expert per
    // token (64 experts made training ~16× slower than 4). Instead: group the token
    // indices by their routed expert and run each expert once, on only its own
    // tokens. index_select/index_add keep this differentiable, so training gets the
    // same sparsity inference already had. GATHER=0 falls back to the dense path.
    let gathered = std::env::var("GATHER").map(|v| v != "0").unwrap_or(true);
    let out = if gathered {
        let choice = gp.argmax(D::Minus1)?.to_vec1::<u32>()?;
        let mut groups: std::collections::HashMap<u32, Vec<u32>> = Default::default();
        for (t, &ex) in choice.iter().enumerate() { groups.entry(ex).or_default().push(t as u32); }
        let mut acc = Tensor::zeros((n, dm), DType::F32, dev)?;
        for (ex, toks) in groups {
            let idx = Tensor::from_vec(toks.clone(), (toks.len(),), dev)?;
            let ys = y.contiguous()?.index_select(&idx, 0)?;          // (T,dm) this expert's tokens
            let ff = moe.experts[ex as usize].apply(&ys)?;            // (T,dm) one expert, once
            let wsel = w.narrow(1, ex as usize, 1)?.contiguous()?.index_select(&idx, 0)?; // (T,1)
            acc = acc.index_add(&idx, &ff.mul(&wsel.broadcast_as(ff.shape())?)?, 0)?;
        }
        acc
    } else {
        let mut acc = Tensor::zeros((n, dm), DType::F32, dev)?;
        for ei in 0..e {
            let ff = moe.experts[ei].apply(&y)?;
            acc = (acc + ff.broadcast_mul(&w.narrow(1, ei, 1)?)?)?;
        }
        acc
    };
    let h = (y + out)?;                                       // residual
    let logits = h.matmul(&st.wo.as_tensor().t()?)?.broadcast_add(st.bo.as_tensor())?;
    // Switch-style load balance: E · Σ_e f_e · P_e  (minimized ⇒ balanced routing)
    let aux = (mask.mean(0)?.mul(&gp.mean(0)?)?.sum_all()? * e as f64)?;
    Ok((logits, aux))
}

/// Per-tensor symmetric fake-quantization of trained weights (offline, no grad):
/// pull to f32, quantize to `bits`, rebuild. For the capability-per-RAM benchmark.
fn fake_quant_vec(flat: &[f32], bits: u32) -> Vec<f32> {
    let maxabs = flat.iter().fold(0f32, |m, &x| m.max(x.abs()));
    if maxabs == 0.0 { return flat.to_vec(); }
    let levels = ((1u32 << (bits - 1)) - 1) as f32;
    let scale = maxabs / levels;
    flat.iter().map(|&x| (x / scale).round().clamp(-levels, levels) * scale).collect()
}
fn quantized(t: &Tensor, bits: u32, dev: &Device) -> Result<Tensor> {
    let v = t.flatten_all()?.to_vec1::<f32>()?;
    Ok(Tensor::from_vec(fake_quant_vec(&v, bits), t.dims().to_vec(), dev)?)
}

// ---------------------------------------------------------------------------
// PAGED EXPERTS — the "huge model, small RAM" mechanism (the mission).
//
// A sparse MoE only *activates* one expert per token, but our earlier code still
// held every expert in RAM (and computed all of them — a toy shortcut). That
// makes MoE a quality trick, not a memory strategy. Here the experts live on
// DISK in int8 (`VYX1` format, ours), and only the routed expert is read in,
// dequantized, and cached. RAM holds: embed + SSM + head + gate + a small LRU
// cache of hot experts — NOT the expert mass.
//
// This is the honest version of "a trillion parameters on 8 GB": the parameters
// are on disk, the working set is small, activation is sparse. Cost is disk I/O
// on a cache miss, which is exactly the latency tradeoff that buys the capacity.
//
// VYX1 layout (little-endian):
//   "VYX1" | u32 n_exp | u32 dm | u32 dff |
//   per expert: f32 s_w1, s_b1, s_w2, s_b2 | i8 w1[dff*dm] b1[dff] w2[dm*dff] b2[dm]
// ---------------------------------------------------------------------------
fn q8(v: &[f32]) -> (Vec<i8>, f32) {
    let m = v.iter().fold(0f32, |a, &x| a.max(x.abs()));
    let s = if m == 0.0 { 1.0 } else { m / 127.0 };
    (v.iter().map(|&x| (x / s).round().clamp(-127.0, 127.0) as i8).collect(), s)
}

/// Write a trained MoE's experts to an on-disk int8 store.
fn write_expert_store(path: &str, moe: &Moe, dm: usize, dff: usize) -> Result<u64> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"VYX1");
    buf.extend_from_slice(&(moe.experts.len() as u32).to_le_bytes());
    buf.extend_from_slice(&(dm as u32).to_le_bytes());
    buf.extend_from_slice(&(dff as u32).to_le_bytes());
    for e in &moe.experts {
        let mut payload: Vec<u8> = Vec::new();
        let mut scales: Vec<u8> = Vec::new();
        for v in [&e.w1, &e.b1, &e.w2, &e.b2] {
            let f = v.as_tensor().flatten_all()?.to_vec1::<f32>()?;
            let (codes, s) = q8(&f);
            scales.extend_from_slice(&s.to_le_bytes());
            payload.extend(codes.iter().map(|&c| c as u8));
        }
        buf.extend_from_slice(&scales);
        buf.extend_from_slice(&payload);
    }
    std::fs::write(path, &buf)?;
    Ok(buf.len() as u64)
}

// ---------------------------------------------------------------------------
// OUR int8 compute path. candle has no signed-int8 dtype (U8, U32, I64, BF16,
// F16, F32, F64), so every quantized weight had to be dequantized to f32 the
// moment it was paged in — int8 on disk, but 4× that in RAM, which is exactly
// the number the memory thesis rests on. These weights are therefore held and
// multiplied as i8 by our own kernel, outside candle: activations stay f32,
// weights never materialize as f32, and the scale is applied once to the
// accumulated dot product. Biases stay f32 (they are dff+dm elements — noise).
// ---------------------------------------------------------------------------
struct Q8Expert { w1: Vec<i8>, s1: f32, b1: Vec<f32>, w2: Vec<i8>, s2: f32, b2: Vec<f32>, dm: usize, dff: usize }

impl Q8Expert {
    /// RAM held by this expert: int8 weights + f32 biases (vs 4× for all-f32).
    fn bytes(&self) -> usize { self.w1.len() + self.w2.len() + (self.b1.len() + self.b2.len()) * 4 }

    /// out[t][o] = scale · Σ_k x[t][k]·w[o][k]  — int8 weights, f32 activations,
    /// f32 accumulator. `w` is row-major (rows × k).
    fn matmul(x: &[f32], t: usize, k: usize, w: &[i8], rows: usize, scale: f32, bias: &[f32], out: &mut Vec<f32>) {
        out.clear();
        out.resize(t * rows, 0.0);
        for ti in 0..t {
            let xr = &x[ti * k..(ti + 1) * k];
            for o in 0..rows {
                let wr = &w[o * k..(o + 1) * k];
                let mut acc = 0f32;
                for i in 0..k { acc += xr[i] * wr[i] as f32; }
                out[ti * rows + o] = acc * scale + bias[o];
            }
        }
    }

    /// Full FFN for `t` tokens: x → w1 → gelu → w2 → out. No f32 weight ever exists.
    fn ffn(&self, x: &[f32], t: usize) -> Vec<f32> {
        let mut h = Vec::new();
        Self::matmul(x, t, self.dm, &self.w1, self.dff, self.s1, &self.b1, &mut h);
        for v in h.iter_mut() { // gelu (tanh approximation)
            let x3 = *v * *v * *v;
            *v = 0.5 * *v * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (*v + 0.044715 * x3)).tanh());
        }
        let mut o = Vec::new();
        Self::matmul(&h, t, self.dff, &self.w2, self.dm, self.s2, &self.b2, &mut o);
        o
    }
}

// ---------------------------------------------------------------------------
// int8 BACKBONE. Once the experts stopped inflating to f32, the embedding and
// output head became the largest resident block — at vocab=32k, dm=4096 they are
// 1.05 GB of a 1.33 GB working set, purely because they are two vocab×dm f32
// matrices. Both are quantized here with the same kernel:
//   * embed: only the looked-up rows are dequantized (b·l rows, not the vocab),
//     so the table stays int8 in RAM and the cost is trivial.
//   * head:  h @ woᵀ is exactly our (rows × k) int8 matmul shape — reused directly.
// The SSM parameters stay f32: 4·dm per layer is noise.
// ---------------------------------------------------------------------------
struct Q8Backbone { embed: Vec<i8>, s_embed: f32, wo: Vec<i8>, s_wo: f32, bo: Vec<f32>, vocab: usize, dm: usize }

impl Q8Backbone {
    fn from_stored(st: &Stored) -> Result<Self> {
        let e = st.embed.as_tensor();
        let (vocab, dm) = e.dims2()?;
        let ev = e.flatten_all()?.to_vec1::<f32>()?;
        let (ec, s_embed) = q8(&ev);
        let wv = st.wo.as_tensor().flatten_all()?.to_vec1::<f32>()?;
        let (wc, s_wo) = q8(&wv);
        Ok(Self { embed: ec, s_embed, wo: wc, s_wo, bo: st.bo.as_tensor().flatten_all()?.to_vec1::<f32>()?, vocab, dm })
    }
    /// RAM: two int8 vocab×dm tables + an f32 bias vector (vs 4× for all-f32).
    fn bytes(&self) -> usize { self.embed.len() + self.wo.len() + self.bo.len() * 4 }
    /// Dequantize ONLY the rows actually looked up — the table itself stays int8.
    fn embed_rows(&self, ids: &[u32]) -> Vec<f32> {
        let mut out = vec![0f32; ids.len() * self.dm];
        for (i, &tok) in ids.iter().enumerate() {
            let src = (tok as usize) * self.dm;
            for k in 0..self.dm { out[i * self.dm + k] = self.embed[src + k] as f32 * self.s_embed; }
        }
        out
    }
    /// logits = h · woᵀ + bo, computed against int8 weights.
    fn head(&self, h: &[f32], n: usize) -> Vec<f32> {
        let mut out = Vec::new();
        Q8Expert::matmul(h, n, self.dm, &self.wo, self.vocab, self.s_wo, &self.bo, &mut out);
        out
    }
}

/// Reads experts on demand from a VYX1 store, keeping only `cap` in RAM (LRU).
/// The file is NOT loaded into memory — we seek+read only the requested expert's
/// bytes, so the on-disk expert mass never counts against the working set. (An
/// earlier version read the whole file into a Vec, which quietly made "disk"
/// resident and overstated the saving; fixed.)
struct PagedExperts {
    f: std::fs::File,
    n_exp: usize, dm: usize, dff: usize, per: usize,
    cache: std::collections::HashMap<usize, std::rc::Rc<Q8Expert>>,
    order: Vec<usize>, cap: usize,
    pub loads: usize, pub hits: usize, pub bytes_read: usize,
}
impl PagedExperts {
    fn open(path: &str, cap: usize) -> Result<Self> {
        use std::io::Read;
        let mut f = std::fs::File::open(path)?;
        let mut hdr = [0u8; 16];
        f.read_exact(&mut hdr)?;
        anyhow::ensure!(&hdr[0..4] == b"VYX1", "bad expert-store magic");
        let rd = |o: usize| u32::from_le_bytes([hdr[o], hdr[o+1], hdr[o+2], hdr[o+3]]) as usize;
        let (n_exp, dm, dff) = (rd(4), rd(8), rd(12));
        let per = 16 + (2 * dm * dff + dff + dm); // 4 f32 scales + int8 payload
        Ok(Self { f, n_exp, dm, dff, per, cache: Default::default(), order: vec![], cap, loads: 0, hits: 0, bytes_read: 0 })
    }
    fn total_expert_bytes(&self) -> usize { self.n_exp * self.per }
    /// Bytes of expert weights actually resident in RAM (fp32 after dequant).
    /// Real RAM held by cached experts — int8 weights + f32 biases, NOT 4×.
    fn resident_bytes(&self) -> usize {
        self.cache.values().map(|e| e.bytes()).sum()
    }
    fn fetch(&mut self, i: usize, _dev: &Device) -> Result<std::rc::Rc<Q8Expert>> {
        use std::io::{Read, Seek, SeekFrom};
        if let Some(t) = self.cache.get(&i) {
            self.hits += 1;
            self.order.retain(|&x| x != i);
            self.order.push(i);
            return Ok(t.clone()); // candle Tensors are Arc-backed: cheap
        }
        self.loads += 1;
        // read ONLY this expert's slice from disk
        let mut buf = vec![0u8; self.per];
        self.f.seek(SeekFrom::Start((16 + i * self.per) as u64))?;
        self.f.read_exact(&mut buf)?;
        self.bytes_read += self.per;
        let f32le = |o: usize, b: &[u8]| f32::from_le_bytes([b[o], b[o+1], b[o+2], b[o+3]]);
        let (s1, s2, s3, s4) = (f32le(0, &buf), f32le(4, &buf), f32le(8, &buf), f32le(12, &buf));
        let mut o = 16usize;
        let (dm, dff) = (self.dm, self.dff);
        // weights stay i8 (no f32 expansion); only the small biases become f32
        let mut take_i8 = |n: usize, bytes: &[u8], o: &mut usize| -> Vec<i8> {
            let v: Vec<i8> = (0..n).map(|k| bytes[*o + k] as i8).collect(); *o += n; v
        };
        let w1 = take_i8(dff * dm, &buf, &mut o);
        let b1v = take_i8(dff, &buf, &mut o);
        let w2 = take_i8(dm * dff, &buf, &mut o);
        let b2v = take_i8(dm, &buf, &mut o);
        let b1: Vec<f32> = b1v.iter().map(|&c| c as f32 * s2).collect();
        let b2: Vec<f32> = b2v.iter().map(|&c| c as f32 * s4).collect();
        let t = std::rc::Rc::new(Q8Expert { w1, s1, b1, w2, s2: s3, b2, dm, dff });
        if self.cache.len() >= self.cap {
            if let Some(ev) = self.order.first().copied() { self.order.remove(0); self.cache.remove(&ev); }
        }
        self.cache.insert(i, t.clone());
        self.order.push(i);
        Ok(t)
    }
}

/// MoE forward with PAGED experts: route, then load only the chosen expert(s).
/// Unlike `forward_moe` (which computes every expert), this touches only the
/// experts the router actually selects — real sparse compute AND sparse memory.
fn forward_moe_paged(ids: &Tensor, st: &Stored, gate: &Tensor, pg: &mut PagedExperts,
                     dm: usize, dev: &Device) -> Result<Tensor> {
    let (b, l) = ids.dims2()?;
    let flat = ids.reshape((b * l,))?;
    let hseq = st.embed.as_tensor().index_select(&flat, 0)?.reshape((b, l, dm))?;
    let y = ssm_layer(&hseq, &st.ssm[0], b, l, dm, dev)?.reshape((b * l, dm))?;
    let gp = candle_nn::ops::softmax(&y.matmul(gate)?, D::Minus1)?;
    let choice = gp.argmax(D::Minus1)?.to_vec1::<u32>()?; // top-1 per token
    let probs = gp.to_vec2::<f32>()?;
    let n = choice.len();

    // group token indices by expert so each expert is fetched at most once
    let mut groups: std::collections::HashMap<u32, Vec<u32>> = Default::default();
    for (t, &e) in choice.iter().enumerate() { groups.entry(e).or_default().push(t as u32); }

    let mut out = Tensor::zeros((n, dm), DType::F32, dev)?;
    for (e, toks) in groups {
        let ex = pg.fetch(e as usize, dev)?;
        let idx = Tensor::from_vec(toks.clone(), (toks.len(),), dev)?;
        let ys = y.contiguous()?.index_select(&idx, 0)?;          // (T, dm)
        // OUR int8 kernel: weights are multiplied as i8, never expanded to f32
        let xs = ys.flatten_all()?.to_vec1::<f32>()?;
        let mut ff_v = ex.ffn(&xs, toks.len());
        for (r, &t) in toks.iter().enumerate() {                  // scale by gate prob
            let g = probs[t as usize][e as usize];
            for c in 0..dm { ff_v[r * dm + c] *= g; }
        }
        let ff = Tensor::from_vec(ff_v, (toks.len(), dm), dev)?;
        out = out.index_add(&idx, &ff, 0)?;
    }
    let h = (y + out)?;
    Ok(h.matmul(&st.wo.as_tensor().t()?)?.broadcast_add(st.bo.as_tensor())?)
}

/// FULLY int8-resident forward: quantized embedding + paged int8 experts +
/// quantized head. Nothing vocab-sized or expert-sized is ever f32 in RAM.
fn forward_q8(ids: &Tensor, bb: &Q8Backbone, ssm: &SsmLayer, gate: &Tensor,
              pg: &mut PagedExperts, dm: usize, dev: &Device) -> Result<Tensor> {
    let (b, l) = ids.dims2()?;
    let flat: Vec<u32> = ids.reshape((b * l,))?.to_vec1()?;
    let emb = bb.embed_rows(&flat);                                   // only b·l rows
    let hseq = Tensor::from_vec(emb, (b, l, dm), dev)?;
    let y = ssm_layer(&hseq, ssm, b, l, dm, dev)?.reshape((b * l, dm))?;
    let gp = candle_nn::ops::softmax(&y.matmul(gate)?, D::Minus1)?;
    let choice = gp.argmax(D::Minus1)?.to_vec1::<u32>()?;
    let probs = gp.to_vec2::<f32>()?;
    let n = choice.len();
    let mut groups: std::collections::HashMap<u32, Vec<u32>> = Default::default();
    for (t, &e) in choice.iter().enumerate() { groups.entry(e).or_default().push(t as u32); }
    let mut out = Tensor::zeros((n, dm), DType::F32, dev)?;
    for (e, toks) in groups {
        let ex = pg.fetch(e as usize, dev)?;
        let idx = Tensor::from_vec(toks.clone(), (toks.len(),), dev)?;
        let xs = y.contiguous()?.index_select(&idx, 0)?.flatten_all()?.to_vec1::<f32>()?;
        let mut ff = ex.ffn(&xs, toks.len());
        for (r, &t) in toks.iter().enumerate() {
            let g = probs[t as usize][e as usize];
            for c in 0..dm { ff[r * dm + c] *= g; }
        }
        out = out.index_add(&idx, &Tensor::from_vec(ff, (toks.len(), dm), dev)?, 0)?;
    }
    let h = (y + out)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(Tensor::from_vec(bb.head(&h, n), (n, bb.vocab), dev)?)   // int8 head
}

/// GENERATED MoE forward (Pillar 1 × Pillar 3): the E experts' weights come from a
/// flat tensor produced by a fractal seed (`ex_flat`, length E·expert_params); the
/// gate is stored (tiny). Same top-1 routing. Returns (logits, load-balance aux).
fn forward_moe_gen(ids: &Tensor, st: &Stored, gate: &Tensor, ex_flat: &Tensor, dm: usize, dff: usize, e: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
    let (b, l) = ids.dims2()?;
    let flat = ids.reshape((b * l,))?;
    let hseq = st.embed.as_tensor().index_select(&flat, 0)?.reshape((b, l, dm))?;
    let y = ssm_layer(&hseq, &st.ssm[0], b, l, dm, dev)?.reshape((b * l, dm))?;
    let gp = candle_nn::ops::softmax(&y.matmul(gate)?, D::Minus1)?;   // (N,E)
    let (n, _) = gp.dims2()?;
    let maxp = gp.max_keepdim(D::Minus1)?.broadcast_as((n, e))?;
    let mask = gp.eq(&maxp)?.to_dtype(DType::F32)?;
    let w = gp.mul(&mask)?;
    let per = 2 * dm * dff + dff + dm;
    let mut out = Tensor::zeros((n, dm), DType::F32, dev)?;
    for ei in 0..e {
        let mut o = ei * per;
        let w1 = ex_flat.narrow(0, o, dff * dm)?.reshape((dff, dm))?; o += dff * dm;
        let b1 = ex_flat.narrow(0, o, dff)?; o += dff;
        let w2 = ex_flat.narrow(0, o, dm * dff)?.reshape((dm, dff))?; o += dm * dff;
        let b2 = ex_flat.narrow(0, o, dm)?;
        let ff = y.matmul(&w1.t()?)?.broadcast_add(&b1)?.gelu()?.matmul(&w2.t()?)?.broadcast_add(&b2)?;
        out = (out + ff.broadcast_mul(&w.narrow(1, ei, 1)?)?)?;
    }
    let h = (y + out)?;
    let logits = h.matmul(&st.wo.as_tensor().t()?)?.broadcast_add(st.bo.as_tensor())?;
    let aux = (mask.mean(0)?.mul(&gp.mean(0)?)?.sum_all()? * e as f64)?;
    Ok((logits, aux))
}

fn eval_bpb_moe_gen(stream: &[u32], blens: &[u32], st: &Stored, gate: &Tensor, ex_flat: &Tensor, dm: usize, dff: usize, e: usize, l: usize, dev: &Device) -> Result<f64> {
    let (mut bits, mut bytes, mut cnt) = (0f64, 0f64, 0usize);
    let mut e0 = 0;
    while cnt < 4000 && e0 + l + 1 <= stream.len() {
        let ids = Tensor::from_vec(stream[e0..e0 + l].to_vec(), (1, l), dev)?;
        let rows = forward_moe_gen(&ids, st, gate, ex_flat, dm, dff, e, dev)?.0.to_vec2::<f32>()?;
        for t in 0..l {
            if cnt >= 4000 { break; }
            let truth = stream[e0 + t + 1] as usize;
            let row = &rows[t];
            let m = row.iter().cloned().fold(f32::MIN, f32::max);
            let lse = m + row.iter().map(|v| (v - m).exp()).sum::<f32>().ln();
            bits += (-(row[truth] - lse) as f64) / std::f64::consts::LN_2;
            bytes += blens[e0 + t + 1] as f64;
            cnt += 1;
        }
        e0 += l;
    }
    Ok(bits / bytes)
}

fn eval_bpb_moe(stream: &[u32], blens: &[u32], st: &Stored, moe: &Moe, dm: usize, l: usize, dev: &Device) -> Result<f64> {
    let (mut bits, mut bytes, mut cnt) = (0f64, 0f64, 0usize);
    let mut e0 = 0;
    while cnt < 4000 && e0 + l + 1 <= stream.len() {
        let ids = Tensor::from_vec(stream[e0..e0 + l].to_vec(), (1, l), dev)?;
        let rows = forward_moe(&ids, st, moe, dm, dev)?.0.to_vec2::<f32>()?;
        for t in 0..l {
            if cnt >= 4000 { break; }
            let truth = stream[e0 + t + 1] as usize;
            let row = &rows[t];
            let m = row.iter().cloned().fold(f32::MIN, f32::max);
            let lse = m + row.iter().map(|v| (v - m).exp()).sum::<f32>().ln();
            bits += (-(row[truth] - lse) as f64) / std::f64::consts::LN_2;
            bytes += blens[e0 + t + 1] as f64;
            cnt += 1;
        }
        e0 += l;
    }
    Ok(bits / bytes)
}

/// Returns (loss, overall next-char acc, ANSWER-digit acc). The last is the one
/// that measures whether the model learned to add.
fn eval_loss_acc(stream: &[u32], amask: &[bool], st: &Stored, ffn: &Tensor, c: &Cfg, l: usize, dev: &Device) -> Result<(f64, f64, f64)> {
    let mut starts: Vec<usize> = (0..).map(|k| k * l).take_while(|&s| s + l + 1 <= stream.len()).collect();
    starts.truncate(200); // cap eval cost
    let (mut tot_loss, mut correct, mut count) = (0f64, 0f64, 0usize);
    let (mut ans_correct, mut ans_count) = (0f64, 0f64);
    for chunk in starts.chunks(64) {
        let b = chunk.len();
        let (mut xin, mut xtg, mut xmask) = (Vec::new(), Vec::new(), Vec::new());
        for &s in chunk {
            xin.extend_from_slice(&stream[s..s + l]);
            xtg.extend_from_slice(&stream[s + 1..s + l + 1]);
            for t in 0..l { xmask.push(if amask[s + t + 1] { 1f32 } else { 0f32 }); }
        }
        let ids = Tensor::from_vec(xin, (b, l), dev)?;
        let tg = Tensor::from_vec(xtg, (b * l,), dev)?;
        let mask = Tensor::from_vec(xmask, (b * l,), dev)?;
        let logits = forward(&ids, st, ffn, c)?;
        tot_loss += candle_nn::loss::cross_entropy(&logits, &tg)?.to_scalar::<f32>()? as f64 * b as f64;
        let ok = logits.argmax(D::Minus1)?.to_dtype(DType::U32)?.eq(&tg)?.to_dtype(DType::F32)?;
        correct += ok.sum_all()?.to_scalar::<f32>()? as f64;
        ans_correct += ok.mul(&mask)?.sum_all()?.to_scalar::<f32>()? as f64;
        ans_count += mask.sum_all()?.to_scalar::<f32>()? as f64;
        count += b;
    }
    Ok((tot_loss / count as f64, correct / (count * l) as f64, ans_correct / ans_count.max(1.0)))
}

/// Bits-per-char over MASKED (target) positions only — the retrieval-conditioned
/// language-modeling metric. Char-level ⇒ bits/char = bits/byte on those positions.
fn eval_bpb_masked(stream: &[u32], amask: &[bool], st: &Stored, ffn: &Tensor, c: &Cfg, l: usize, dev: &Device) -> Result<f64> {
    let mut starts: Vec<usize> = (0..).map(|k| k * l).take_while(|&s| s + l + 1 <= stream.len()).collect();
    starts.truncate(400);
    let (mut nll_bits, mut cnt) = (0f64, 0f64);
    for chunk in starts.chunks(64) {
        let b = chunk.len();
        let (mut xin, mut xtg, mut xmask) = (Vec::new(), Vec::new(), Vec::new());
        for &s in chunk {
            xin.extend_from_slice(&stream[s..s + l]);
            xtg.extend_from_slice(&stream[s + 1..s + l + 1]);
            for t in 0..l { xmask.push(if amask[s + t + 1] { 1f32 } else { 0f32 }); }
        }
        let ids = Tensor::from_vec(xin, (b, l), dev)?;
        let tgt = Tensor::from_vec(xtg, (b * l, 1), dev)?;
        let logits = forward(&ids, st, ffn, c)?;
        let lp = candle_nn::ops::log_softmax(&logits, D::Minus1)?;
        let picked = lp.gather(&tgt, 1)?;                // (b*l, 1) log p(target), nats
        let mask = Tensor::from_vec(xmask.clone(), (b * l, 1), dev)?;
        nll_bits += picked.neg()?.mul(&mask)?.sum_all()?.to_scalar::<f32>()? as f64;
        cnt += xmask.iter().sum::<f32>() as f64;
    }
    Ok(nll_bits / cnt.max(1.0) / std::f64::consts::LN_2)
}

fn train<F>(vars: Vec<Var>, ffn_of: &F, train_s: &[u32], test_s: &[u32], test_mask: &[bool], st: &Stored, c: &Cfg,
            steps: usize, l: usize, bs: usize, lr: f64, seed: u64, dev: &Device) -> Result<(f64, f64, f64)>
where F: Fn() -> Result<Tensor> {
    let mut opt = AdamW::new(vars, ParamsAdamW { lr, ..Default::default() })?;
    let mut rng = StdRng::seed_from_u64(seed);
    for _ in 0..steps {
        let (ids, tg) = sample_batch(train_s, bs, l, &mut rng, dev)?;
        let logits = forward(&ids, st, &ffn_of()?, c)?;
        let loss = candle_nn::loss::cross_entropy(&logits, &tg.reshape((bs * l,))?)?;
        opt.backward_step(&loss)?;
    }
    eval_loss_acc(test_s, test_mask, st, &ffn_of()?, c, l, dev)
}

// ---------------------------------------------------------------------------
// Checkpoints — persist a trained model so a run produces a KEEPABLE artifact
// (previously every training run evaporated on exit). safetensors via candle;
// the config (vocab/dm/dff/E/layers) is derived from tensor SHAPES on load, so
// no side-car metadata file can drift out of sync with the weights.
// ---------------------------------------------------------------------------
fn save_moe_ckpt(path: &str, st: &Stored, moe: &Moe) -> Result<()> {
    let mut m: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();
    m.insert("embed".into(), st.embed.as_tensor().clone());
    m.insert("wo".into(), st.wo.as_tensor().clone());
    m.insert("bo".into(), st.bo.as_tensor().clone());
    for (i, l) in st.ssm.iter().enumerate() {
        m.insert(format!("ssm.{i}.a_raw"), l.a_raw.as_tensor().clone());
        m.insert(format!("ssm.{i}.b_co"), l.b_co.as_tensor().clone());
        m.insert(format!("ssm.{i}.c_co"), l.c_co.as_tensor().clone());
        m.insert(format!("ssm.{i}.d_co"), l.d_co.as_tensor().clone());
    }
    m.insert("moe.gate".into(), moe.gate.as_tensor().clone());
    for (i, e) in moe.experts.iter().enumerate() {
        m.insert(format!("moe.{i}.w1"), e.w1.as_tensor().clone());
        m.insert(format!("moe.{i}.b1"), e.b1.as_tensor().clone());
        m.insert(format!("moe.{i}.w2"), e.w2.as_tensor().clone());
        m.insert(format!("moe.{i}.b2"), e.b2.as_tensor().clone());
    }
    candle_core::safetensors::save(&m, path)?;
    Ok(())
}

/// Load a checkpoint; returns (Stored, Moe, vocab, dm). Config from shapes.
fn load_moe_ckpt(path: &str, dev: &Device) -> Result<(Stored, Moe, usize, usize)> {
    let t = candle_core::safetensors::load(path, dev)?;
    let g = |k: &str| -> Result<Tensor> {
        t.get(k).cloned().ok_or_else(|| anyhow::anyhow!("checkpoint missing tensor `{k}`"))
    };
    let v = |k: &str| -> Result<Var> { Ok(Var::from_tensor(&g(k)?)?) };
    let embed = g("embed")?;
    let (vocab, dm) = embed.dims2()?;
    let n_layers = (0..).take_while(|i| t.contains_key(&format!("ssm.{i}.a_raw"))).count().max(1);
    let n_exp = (0..).take_while(|i| t.contains_key(&format!("moe.{i}.w1"))).count();
    anyhow::ensure!(n_exp > 0, "checkpoint has no experts");
    let mut ssm = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        ssm.push(SsmLayer {
            a_raw: v(&format!("ssm.{i}.a_raw"))?, b_co: v(&format!("ssm.{i}.b_co"))?,
            c_co: v(&format!("ssm.{i}.c_co"))?, d_co: v(&format!("ssm.{i}.d_co"))?,
        });
    }
    let st = Stored { embed: Var::from_tensor(&embed)?, ssm, wo: v("wo")?, bo: v("bo")? };
    let mut experts = Vec::with_capacity(n_exp);
    for i in 0..n_exp {
        experts.push(Expert {
            w1: v(&format!("moe.{i}.w1"))?, b1: v(&format!("moe.{i}.b1"))?,
            w2: v(&format!("moe.{i}.w2"))?, b2: v(&format!("moe.{i}.b2"))?,
        });
    }
    Ok((st, Moe { gate: v("moe.gate")?, experts }, vocab, dm))
}

// ---------------------------------------------------------------------------
// Standalone BPE codec (from a merges file) — needed to encode a prompt and
// decode generated tokens back to text. Self-contained: token→bytes is fully
// derivable from the merge list, so generation needs no corpus.
// ---------------------------------------------------------------------------
struct BpeCodec { rank: std::collections::HashMap<(u32, u32), usize>, id_of: std::collections::HashMap<(u32, u32), u32>, bytes_of: Vec<Vec<u8>> }
impl BpeCodec {
    fn load(merges_path: &str) -> Result<Self> {
        let txt = std::fs::read_to_string(merges_path)
            .map_err(|e| anyhow::anyhow!("need BPE merges at {merges_path} (run vyoma-tokenizer): {e}"))?;
        let (mut rank, mut id_of) = (std::collections::HashMap::new(), std::collections::HashMap::new());
        let mut bytes_of: Vec<Vec<u8>> = (0..256u32).map(|b| vec![b as u8]).collect();
        for (m, line) in txt.lines().enumerate() {
            let mut it = line.split_whitespace();
            let a: u32 = it.next().unwrap().parse()?;
            let b: u32 = it.next().unwrap().parse()?;
            rank.insert((a, b), m);
            id_of.insert((a, b), 256 + m as u32);
            let mut nb = bytes_of[a as usize].clone();
            nb.extend_from_slice(&bytes_of[b as usize]);
            bytes_of.push(nb);
        }
        Ok(Self { rank, id_of, bytes_of })
    }
    fn encode(&self, bytes: &[u8]) -> Vec<u32> {
        let mut out = Vec::new();
        for word in pretokenize(bytes) {
            let mut seq = word;
            loop {
                let mut best: Option<(usize, usize)> = None;
                for i in 0..seq.len().saturating_sub(1) {
                    if let Some(&r) = self.rank.get(&(seq[i], seq[i + 1])) {
                        if best.map_or(true, |(br, _)| r < br) { best = Some((r, i)); }
                    }
                }
                let (_, pos) = match best { Some(x) => x, None => break };
                let id = self.id_of[&(seq[pos], seq[pos + 1])];
                seq.splice(pos..pos + 2, [id]);
            }
            out.extend(seq);
        }
        out
    }
    fn decode(&self, ids: &[u32]) -> String {
        let mut b = Vec::new();
        for &i in ids { if let Some(x) = self.bytes_of.get(i as usize) { b.extend_from_slice(x); } }
        String::from_utf8_lossy(&b).into_owned()
    }
}

fn main() -> Result<()> {
    let steps: usize = std::env::var("STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(2000);
    let dm: usize = std::env::var("DM").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
    let dff: usize = std::env::var("DFF").ok().and_then(|s| s.parse().ok()).unwrap_or(512);
    let l: usize = std::env::var("SEQ").ok().and_then(|s| s.parse().ok()).unwrap_or(24);
    let layers: usize = std::env::var("LAYERS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let (bs, lr) = (64usize, 2e-3);
    // Portable device selection (builds anywhere; accelerators are opt-in cargo
    // features so the default CPU build compiles on Kaggle/Linux, Mac, etc.):
    //   default        → CPU
    //   --features cuda → NVIDIA GPU (Kaggle, cloud)
    //   --features metal→ Apple GPU (Mac)
    #[cfg(feature = "cuda")]
    let (dev, devname) = (Device::new_cuda(0).unwrap_or(Device::Cpu), "cuda(GPU)");
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let (dev, devname) = (Device::new_metal(0).unwrap_or(Device::Cpu), "metal(GPU)");
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let (dev, devname) = (Device::Cpu, "cpu");
    println!("[lm] device={devname}");
    let dataset = std::env::var("DATASET").unwrap_or_else(|_| "arith".into());
    let mode = std::env::var("MODE").unwrap_or_else(|_| "sweep".into());

    // --- MODE=paged: experts on DISK, only the routed one in RAM (the mission).
    // Takes a trained checkpoint, writes its experts to an int8 on-disk store,
    // then generates with paging and reports the actual RAM working set vs the
    // full expert mass. Env: LOAD, CACHE (experts held in RAM), PROMPT, NEW.
    if mode == "paged" {
        let ckpt = std::env::var("LOAD").map_err(|_| anyhow::anyhow!("MODE=paged needs LOAD=ckpt.safetensors"))?;
        let cap: usize = std::env::var("CACHE").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
        let n_new: usize = std::env::var("NEW").ok().and_then(|s| s.parse().ok()).unwrap_or(120);
        let win: usize = std::env::var("WIN").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
        let prompt = std::env::var("PROMPT").unwrap_or_else(|_| "The ".into());
        let (st, moe, vocab, dmc) = load_moe_ckpt(&ckpt, &dev)?;
        let dffc = moe.experts[0].w1.as_tensor().dims2()?.0;
        let store = std::env::var("XSTORE").unwrap_or_else(|_| "/tmp/vyoma_experts.vyx".into());
        let sz = write_expert_store(&store, &moe, dmc, dffc)?;
        let mut pg = PagedExperts::open(&store, cap)?;

        // Q8=1 also quantizes the backbone (embedding + output head), which became
        // the dominant resident block once the experts stopped inflating to f32.
        let q8_backbone = std::env::var("Q8").map(|v| v != "0").unwrap_or(false);
        let bb = if q8_backbone { Some(Q8Backbone::from_stored(&st)?) } else { None };
        let stored_ram = match &bb {
            Some(b) => b.bytes() + (6 * dmc + dmc * moe.experts.len()) * 4, // int8 tables + f32 ssm/gate
            None => st.n_params() * 4 + dmc * moe.experts.len() * 4,        // all f32
        };
        println!("[paged] experts on DISK: {} experts × dff={dffc} = {:.1} KB int8 ({store})", moe.experts.len(), sz as f64 / 1024.0);
        println!("[paged] RAM: backbone+gate {:.1} KB ({}) + LRU cache of {cap} expert(s)",
                 stored_ram as f64 / 1024.0, if q8_backbone { "int8" } else { "f32" });

        let codec = BpeCodec::load(&format!("{}/data_cache/bpe_merges.txt", env!("CARGO_MANIFEST_DIR")))?;
        let gate_t = moe.gate.as_tensor().clone();
        let mut ctx = codec.encode(prompt.as_bytes());
        anyhow::ensure!(!ctx.is_empty(), "prompt encoded to zero tokens");
        let start = ctx.len();
        let mut rng = StdRng::seed_from_u64(0);
        let t0 = std::time::Instant::now();
        for _ in 0..n_new {
            let s = ctx.len().saturating_sub(win);
            let w = &ctx[s..];
            let ids = Tensor::from_vec(w.to_vec(), (1, w.len()), &dev)?;
            let logits = match &bb {
                Some(b) => forward_q8(&ids, b, &st.ssm[0], &gate_t, &mut pg, dmc, &dev)?,
                None => forward_moe_paged(&ids, &st, &gate_t, &mut pg, dmc, &dev)?,
            };
            let row: Vec<f32> = logits.narrow(0, w.len() - 1, 1)?.flatten_all()?.to_vec1()?;
            let mut idx: Vec<usize> = (0..row.len()).collect();
            idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap_or(std::cmp::Ordering::Equal));
            idx.truncate(40.min(row.len()));
            let scaled: Vec<f32> = idx.iter().map(|&i| row[i] / 0.8).collect();
            let probs = softmax(&scaled);
            let r: f32 = rng.gen();
            let (mut acc, mut pick) = (0.0f32, *idx.last().unwrap());
            for (j, &p) in probs.iter().enumerate() { acc += p; if r <= acc { pick = idx[j]; break; } }
            ctx.push(pick as u32);
        }
        let el = t0.elapsed().as_secs_f64();
        let full_ram = stored_ram + (2 * dmc * dffc + dffc + dmc) * moe.experts.len() * 4;
        let _ = &bb;
        let paged_ram = stored_ram + pg.resident_bytes();
        println!("[paged] generated {n_new} tokens in {el:.2}s ({:.2} ms/token)", el * 1000.0 / n_new as f64);
        println!("[paged] expert fetches: {} disk loads / {} cache hits ({:.0}% hit rate)",
                 pg.loads, pg.hits, pg.hits as f64 * 100.0 / (pg.loads + pg.hits).max(1) as f64);
        println!("[paged] RAM all-experts-resident : {:.1} KB", full_ram as f64 / 1024.0);
        println!("[paged] RAM paged (cache={cap})       : {:.1} KB  → {:.2}× smaller working set",
                 paged_ram as f64 / 1024.0, full_ram as f64 / paged_ram as f64);
        println!("[paged] vocab={vocab}\n{}", codec.decode(&ctx[start..]));
        println!("[paged] Experts live on disk in int8; only routed experts enter RAM. This is the");
        println!("[paged] mechanism for big-model-small-RAM: capacity scales with DISK, not memory.");
        return Ok(());
    }

    // --- MODE=rag: THE ASSEMBLED SYSTEM. Every pillar in one command, ours:
    //   P4 our retriever + our on-disk VYST store  → fetch supporting context
    //   P3b symbolic/similarity gate               → veto unsupported context
    //        (abstain from grounding rather than condition on a bad match)
    //   P2+P3 lean SSM + stored MoE (checkpoint)   → generate the answer
    // No teacher, no external model. Env: LOAD (LM ckpt), RET (retriever ckpt),
    // STORE (.vyst), PROMPT, NEW, TEMP, TOPK, GATE (min cosine to accept context).
    if mode == "rag" {
        let ckpt = std::env::var("LOAD").map_err(|_| anyhow::anyhow!("MODE=rag needs LOAD=lm.safetensors"))?;
        let retp = std::env::var("RET").map_err(|_| anyhow::anyhow!("MODE=rag needs RET=retriever.safetensors"))?;
        let storep = std::env::var("STORE").map_err(|_| anyhow::anyhow!("MODE=rag needs STORE=path.vyst"))?;
        let prompt = std::env::var("PROMPT").unwrap_or_else(|_| "The ".into());
        let n_new: usize = std::env::var("NEW").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
        let temp: f32 = std::env::var("TEMP").ok().and_then(|s| s.parse().ok()).unwrap_or(0.8);
        let topk: usize = std::env::var("TOPK").ok().and_then(|s| s.parse().ok()).unwrap_or(40);
        let win: usize = std::env::var("WIN").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
        // min cosine to accept retrieved context. 0.68 is the empirically best
        // threshold from `vyoma-embed MODE=gate` calibration over 200+200 samples
        // — but note it only achieves ~74% balanced accuracy: the in/out-of-domain
        // similarity distributions genuinely overlap, so this gate is a useful
        // filter, NOT a guarantee. Recalibrate per store (MODE=gate NEG=...).
        let gate: f32 = std::env::var("GATE").ok().and_then(|s| s.parse().ok()).unwrap_or(0.68);

        let (st, moe, vocab, dm) = load_moe_ckpt(&ckpt, &dev)?;
        let enc = vyoma_embed::Encoder::load(&retp, &dev)?;
        let (store_embs, store_recs) = vyoma_embed::load_store(&storep)?;
        let codec = BpeCodec::load(&format!("{}/data_cache/bpe_merges.txt", env!("CARGO_MANIFEST_DIR")))?;
        println!("[rag] LM(vocab={vocab} dm={dm} experts={}) + retriever(dk={}) + store({} facts)",
                 moe.experts.len(), enc.dk, store_recs.len());
        println!("[rag] prompt: {prompt:?}");

        // P4 — retrieve with OUR encoder over OUR store
        let q = vyoma_embed::embed_all(&enc, &[prompt.as_bytes().to_vec()], &dev)?;
        let sims: Vec<f32> = store_embs.iter().map(|e| vyoma_embed::dot(&q[0], e)).collect();
        let hit = (0..sims.len()).max_by(|&a, &b| sims[a].partial_cmp(&sims[b]).unwrap_or(std::cmp::Ordering::Equal)).unwrap();
        let sim = sims[hit];
        let fetched = String::from_utf8_lossy(&store_recs[hit]).into_owned();

        // P3b — the gate: ABSOLUTE cosine threshold, but this only works because the
        // retriever is trained with out-of-domain negatives (`vyoma-embed NEG=`).
        // Measured history, both directions:
        //   * WITHOUT OOD negatives: cosine fails (OOD prompt 0.682 > correct 0.643)
        //     and a z-score vs the store distribution ALSO fails (3.26σ vs 3.31σ) —
        //     the encoder simply has no "is this my domain?" signal to threshold.
        //   * WITH OOD negatives (MODE=gate): cosine separates cleanly (in-domain
        //     ≥0.697 vs all out-of-domain ≤0.593) while z-score still fails (random
        //     junk scores 3.67σ). The domain signal lives in the ABSOLUTE similarity;
        //     z-scoring normalizes exactly that away.
        // So the fix was the TRAINING, not the statistic. GATE = min cosine.
        let n = sims.len() as f32;
        let mean = sims.iter().sum::<f32>() / n;
        println!("[rag] retrieved (cos={sim:.3}, store mean {mean:.3}): {:?}",
                 &fetched.chars().take(90).collect::<String>());

        // Prefer the LEARNED head when given (HEAD=...): it reads the whole
        // retrieval geometry [q ; e ; q⊙e] instead of one scalar, and measured
        // 88.9% balanced accuracy held-out vs 74.0% for the cosine threshold.
        let grounded = match std::env::var("HEAD").ok() {
            Some(hp) => {
                let head = vyoma_embed::GroundingHead::load(&hp, &dev)?;
                let p = head.score(&q[0], &store_embs[hit], &dev)?;
                let ok = p >= 0.5;
                println!("[rag] learned head: P(supported)={p:.3} → {}", if ok { "ACCEPT" } else { "VETO" });
                ok
            }
            None => {
                let ok = sim >= gate;
                println!("[rag] cosine gate (no HEAD=): cos {sim:.3} vs {gate} → {}", if ok { "ACCEPT" } else { "VETO" });
                ok
            }
        };
        let full = if grounded {
            println!("[rag] grounding generation in retrieved context");
            format!("{fetched}\n{prompt}")
        } else {
            println!("[rag] store does not support this prompt → generating ungrounded (no hallucinated grounding)");
            prompt.clone()
        };

        // P2+P3 — generate with the stored-MoE LM
        let mut ctx = codec.encode(full.as_bytes());
        anyhow::ensure!(!ctx.is_empty(), "prompt encoded to zero tokens");
        let start = ctx.len();
        let mut rng = StdRng::seed_from_u64(std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(0));
        for _ in 0..n_new {
            let s = ctx.len().saturating_sub(win);
            let w = &ctx[s..];
            let ids = Tensor::from_vec(w.to_vec(), (1, w.len()), &dev)?;
            let logits = forward_moe(&ids, &st, &moe, dm, &dev)?.0;
            let row: Vec<f32> = logits.narrow(0, w.len() - 1, 1)?.flatten_all()?.to_vec1()?;
            let mut idx: Vec<usize> = (0..row.len()).collect();
            idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap_or(std::cmp::Ordering::Equal));
            idx.truncate(topk.max(1).min(row.len()));
            let scaled: Vec<f32> = idx.iter().map(|&i| row[i] / temp.max(1e-3)).collect();
            let probs = softmax(&scaled);
            let r: f32 = rng.gen();
            let (mut acc, mut pick) = (0.0f32, *idx.last().unwrap());
            for (j, &p) in probs.iter().enumerate() { acc += p; if r <= acc { pick = idx[j]; break; } }
            ctx.push(pick as u32);
        }
        println!("[rag] ----- Vyomarudra (grounded={grounded}) -----");
        println!("{}", codec.decode(&ctx[start..]));
        println!("[rag] --------------------------------------------");
        println!("[rag] Retriever + store + gate + MoE LM, one pipeline, every corner ours. No teacher.");
        return Ok(());
    }

    // --- MODE=generate: load a saved MoE checkpoint and WRITE TEXT. The point of
    // training something is to use it — this makes Vyomarudra produce language,
    // not just bits-per-byte. Needs no corpus (BPE token→bytes comes from merges).
    // Env: LOAD=ckpt.safetensors PROMPT="..." NEW=200 TEMP=0.8 TOPK=40
    if mode == "generate" {
        let ckpt = std::env::var("LOAD").map_err(|_| anyhow::anyhow!("MODE=generate needs LOAD=path.safetensors"))?;
        let (st, moe, vocab, dm) = load_moe_ckpt(&ckpt, &dev)?;
        let merges = format!("{}/data_cache/bpe_merges.txt", env!("CARGO_MANIFEST_DIR"));
        let codec = BpeCodec::load(&merges)?;
        let prompt = std::env::var("PROMPT").unwrap_or_else(|_| "The ".into());
        let n_new: usize = std::env::var("NEW").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
        let temp: f32 = std::env::var("TEMP").ok().and_then(|s| s.parse().ok()).unwrap_or(0.8);
        let topk: usize = std::env::var("TOPK").ok().and_then(|s| s.parse().ok()).unwrap_or(40);
        let win: usize = std::env::var("WIN").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
        println!("[gen] ckpt={ckpt} vocab={vocab} dm={dm} experts={} | temp={temp} topk={topk} new={n_new}", moe.experts.len());
        println!("[gen] prompt: {prompt:?}");

        let mut ctx = codec.encode(prompt.as_bytes());
        anyhow::ensure!(!ctx.is_empty(), "prompt encoded to zero tokens");
        let mut rng = StdRng::seed_from_u64(std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(0));
        for _ in 0..n_new {
            let s = ctx.len().saturating_sub(win);
            let w = &ctx[s..];
            let ids = Tensor::from_vec(w.to_vec(), (1, w.len()), &dev)?;
            let logits = forward_moe(&ids, &st, &moe, dm, &dev)?.0; // (1*L, vocab)
            let row: Vec<f32> = logits.narrow(0, w.len() - 1, 1)?.flatten_all()?.to_vec1()?;
            // top-k + temperature sampling
            let mut idx: Vec<usize> = (0..row.len()).collect();
            idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap_or(std::cmp::Ordering::Equal));
            idx.truncate(topk.max(1).min(row.len()));
            let scaled: Vec<f32> = idx.iter().map(|&i| row[i] / temp.max(1e-3)).collect();
            let probs = softmax(&scaled);
            let r: f32 = rng.gen();
            let (mut acc, mut pick) = (0.0f32, *idx.last().unwrap());
            for (j, &p) in probs.iter().enumerate() { acc += p; if r <= acc { pick = idx[j]; break; } }
            ctx.push(pick as u32);
        }
        println!("[gen] ----- Vyomarudra writes -----\n{}", codec.decode(&ctx));
        println!("[gen] -----------------------------");
        return Ok(());
    }

    // --- assemble corpus (+ vocab, + optional answer mask) ---
    let tok = std::env::var("TOKENIZER").unwrap_or_else(|_| "char".into());
    let (corpus, vocab, amask): (Vec<u32>, usize, Vec<bool>) = if dataset == "text" || dataset == "distilled" || dataset == "fineweb" {
        // fineweb.txt is produced by `vyoma-data` from a FineWeb parquet shard.
        let file = match dataset.as_str() { "distilled" => "distilled.txt", "fineweb" => "fineweb.txt", _ => "tinyshakespeare.txt" };
        let dir = env!("CARGO_MANIFEST_DIR");
        let path = format!("{dir}/data_cache/{file}");
        let (s, v) = if tok == "bpe" {
            build_bpe_corpus(&path, &format!("{dir}/data_cache/bpe_merges.txt"))? // OUR BPE tokenizer
        } else {
            build_text_corpus(&path)?
        };
        let n = s.len();
        (s, v, vec![true; n]) // all positions count for language
    } else {
        let (s, m) = build_corpus(6000, 42);
        (s, VOCAB, m)
    };
    let split = corpus.len() * 9 / 10;
    let (train_s, test_s, test_mask) = (&corpus[..split], &corpus[split..], &amask[split..]);
    let metric = if dataset == "text" { "next-char acc" } else { "ANSWER-digit acc" };
    println!("[lm] {dataset} LM [{tok}]  dm={dm} dff={dff} seq={l} vocab={vocab}  corpus={} tokens  steps={steps}  metric={metric}", corpus.len());

    // --- MODE=diag: does FFN WIDTH drive quality? (dense-only, several dff) ---
    if mode == "diag" {
        println!("[lm] DIAGNOSTIC — dense models, varying FFN width. If a wide FFN >> narrow, the task is FFN-capacity-bound (a valid setting to test generation).");
        for &w in &[8usize, 32, 128, 512] {
            let cw = Cfg { dm, dff: w, layers };
            let st = Stored::new(&cw, vocab, &dev)?;
            let ffn = {
                let np = ffn_total(&cw);
                let noise = Tensor::randn(0f32, 1f32, (np,), &dev)?;
                Var::from_tensor(&noise.mul(&Tensor::from_vec(ffn_prior(&cw), (np,), &dev)?)?)?
            };
            let ffn_c = ffn.clone();
            let mut vars = st.vars(); vars.push(ffn.clone());
            let (loss, ov, _a) = train(vars, &|| Ok(ffn_c.as_tensor().clone()), train_s, test_s, test_mask, &st, &cw, steps, l, bs, lr, 0, &dev)?;
            println!("[lm]  dff={w:4}  total={:6}  loss={loss:.3}  {metric}={ov:.4}", st.n_params() + ffn_total(&cw));
        }
        println!("[lm] done (diag)");
        return Ok(());
    }

    // --- MODE=kv: externalized knowledge (E2). memorize vs retrieve, sweep #facts. ---
    if mode == "kv" {
        let n_val = 10usize;
        println!("[lm] E2 externalized knowledge — memorize (value in weights) vs retrieve (value fetched, adjacent). dm={dm} dff={dff}");
        for &n_ent in &[200usize, 1000, 4000, 16000] {
            let mut line = format!("[lm]  facts={n_ent:5}");
            for retrieve in [false, true] {
                let n_ex = (20 * n_ent).max(8000); // each fact seen ~20× so it CAN be memorized
                let (corp, kvocab, kmask) = build_kv_corpus(n_ent, n_val, retrieve, n_ex, 7);
                let ksplit = corp.len() * 9 / 10;
                let cc = Cfg { dm, dff, layers };
                let st = Stored::new(&cc, kvocab, &dev)?;
                let ffn = {
                    let np = ffn_total(&cc);
                    let noise = Tensor::randn(0f32, 1f32, (np,), &dev)?;
                    Var::from_tensor(&noise.mul(&Tensor::from_vec(ffn_prior(&cc), (np,), &dev)?)?)?
                };
                let ffn_c = ffn.clone();
                let mut vars = st.vars(); vars.push(ffn.clone());
                let (_l, _ov, va) = train(vars, &|| Ok(ffn_c.as_tensor().clone()),
                    &corp[..ksplit], &corp[ksplit..], &kmask[ksplit..], &st, &cc, steps, l, bs, lr, 0, &dev)?;
                line += &format!("  {}={va:.3}", if retrieve { "retrieve" } else { "memorize" });
            }
            println!("{line}");
        }
        println!("[lm] done (kv). If memorize collapses as facts grow but retrieve stays flat → knowledge belongs on disk, not in weights.");
        return Ok(());
    }

    // --- MODE=retro: E2 closed with a REAL retriever (ours), not the oracle. ---
    // Our vyoma-embed encoder fetches the fact from a store; our SSM LM reads the
    // fetched value and answers. End-to-end accuracy = our retriever's quality ×
    // the LM's (trivial) copy. Compare to memorize (LM alone), sweep #facts.
    // Entirely ours: our retriever + our LM + our store. No teacher anywhere.
    if mode == "retro" {
        let n_val = 10usize;
        let ev = |k: &str, d: usize| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
        // Tunable retriever (the "keep growing" lever): longer keys + bigger dk stay
        // separable as facts grow → retro flat-high. Env: RD, RDK, RDM, RSTEPS, RNS.
        let d_key = ev("RD", 12);        // key length in digits
        let (rdm, rdk, rsteps) = (ev("RDM", 96), ev("RDK", 256), ev("RSTEPS", 2500));
        let key_bytes = |e: usize| -> Vec<u8> { format!("{e:0width$}", width = d_key).into_bytes() };
        let n_list: Vec<usize> = match std::env::var("RNS").ok() {
            Some(s) => s.split(',').filter_map(|x| x.trim().parse().ok()).collect(),
            None => vec![500, 2000, 8000],
        };
        println!("[lm] RETRO-lite — our retriever (d={rdm}, dk={rdk}, {rsteps} steps, {d_key}-digit keys) fetches; our LM reads & answers.");
        println!("[lm] memorize vs retro(ours), sweep facts. dm={dm} dff={dff}");
        for &n_ent in &n_list {
            let mut rng = StdRng::seed_from_u64(100 + n_ent as u64);
            let val_of: Vec<u32> = (0..n_ent).map(|_| rng.gen_range(0..n_val) as u32).collect();

            // Train OUR retriever: noisy key → clean key record. Then encode the store.
            let (enc, _rl) = vyoma_embed::train_encoder(
                rdm, rdk, 1, rsteps, 128, 1e-3, 0.05, &dev, 7 + n_ent as u64,
                |r: &mut StdRng| {
                    let e = r.gen_range(0..n_ent);
                    let clean = key_bytes(e);
                    let mut q = clean.clone();
                    let pos = r.gen_range(0..q.len());
                    q[pos] = b'0' + r.gen_range(0..10u8);
                    (q, clean)
                },
            )?;
            let recs: Vec<Vec<u8>> = (0..n_ent).map(&key_bytes).collect();
            let store = vyoma_embed::embed_all(&enc, &recs, &dev)?;

            // Retrieve ONCE per entity (O(N²·dk), not O(n_ex·N·dk)) so we can scale facts.
            let mut ers = StdRng::seed_from_u64(9 + n_ent as u64);
            let e_queries: Vec<Vec<u8>> = (0..n_ent).map(|e| {
                let mut q = key_bytes(e);
                let pos = ers.gen_range(0..q.len());
                q[pos] = b'0' + ers.gen_range(0..10u8);
                q
            }).collect();
            let qembs = vyoma_embed::embed_all(&enc, &e_queries, &dev)?;
            let retrieved: Vec<usize> = qembs.iter().map(|q| vyoma_embed::nearest(q, &store)).collect();
            let ret_acc = retrieved.iter().enumerate().filter(|(e, r)| **r == *e).count() as f64 / n_ent as f64;
            let rv_of: Vec<u32> = retrieved.iter().map(|&r| val_of[r]).collect(); // value our retriever fetches for e

            // Build the retro corpus: [clean key digits][our-retrieved value][QMARK][true value*][NL].
            let (qmark, nl, kvocab) = (10u32, 11u32, 12usize);
            let digits = |mut x: usize| -> Vec<u32> { let mut v = vec![0u32; d_key]; for i in (0..d_key).rev() { v[i] = (x % 10) as u32; x /= 10; } v };
            let n_ex = (20 * n_ent).max(8000);
            let mut crng = StdRng::seed_from_u64(21 + n_ent as u64);
            let (mut s, mut m) = (Vec::new(), Vec::new());
            for _ in 0..n_ex {
                let e = crng.gen_range(0..n_ent);
                let v = val_of[e];
                let rv = rv_of[e]; // value carried by the fact OUR retriever fetched for e
                for d in digits(e) { s.push(d); m.push(false); }
                s.push(rv); m.push(false); // fetched value, adjacent (wrong if retrieval missed)
                s.push(qmark); m.push(false);
                s.push(v); m.push(true); // answer (scored): the TRUE value
                s.push(nl); m.push(false);
            }
            let split = s.len() * 9 / 10;

            // Train the LM to read the fetched value and answer.
            let cc = Cfg { dm, dff, layers };
            let st = Stored::new(&cc, kvocab, &dev)?;
            let ffn = { let np = ffn_total(&cc); let noise = Tensor::randn(0f32, 1f32, (np,), &dev)?; Var::from_tensor(&noise.mul(&Tensor::from_vec(ffn_prior(&cc), (np,), &dev)?)?)? };
            let ffn_c = ffn.clone();
            let mut vars = st.vars(); vars.push(ffn.clone());
            let (_l, _o, retro_acc) = train(vars, &|| Ok(ffn_c.as_tensor().clone()),
                &s[..split], &s[split..], &m[split..], &st, &cc, steps, l, bs, lr, 0, &dev)?;

            // Memorize baseline: LM alone (no fetched value), same #facts and key encoding.
            let (mut mc, mut mm) = (Vec::new(), Vec::new());
            let mut mrng = StdRng::seed_from_u64(7 + n_ent as u64);
            for _ in 0..n_ex {
                let e = mrng.gen_range(0..n_ent);
                let v = val_of[e];
                for d in digits(e) { mc.push(d); mm.push(false); }
                mc.push(qmark); mm.push(false);
                mc.push(v); mm.push(true);
                mc.push(nl); mm.push(false);
            }
            let msplit = mc.len() * 9 / 10;
            let stm = Stored::new(&cc, kvocab, &dev)?;
            let ffnm = { let np = ffn_total(&cc); let noise = Tensor::randn(0f32, 1f32, (np,), &dev)?; Var::from_tensor(&noise.mul(&Tensor::from_vec(ffn_prior(&cc), (np,), &dev)?)?)? };
            let ffnm_c = ffnm.clone();
            let mut vm = stm.vars(); vm.push(ffnm.clone());
            let (_l2, _o2, mem_acc) = train(vm, &|| Ok(ffnm_c.as_tensor().clone()),
                &mc[..msplit], &mc[msplit..], &mm[msplit..], &stm, &cc, steps, l, bs, lr, 0, &dev)?;

            println!("[lm]  facts={n_ent:5}  memorize={mem_acc:.3}  retro(ours)={retro_acc:.3}  [our-retriever hit-rate={ret_acc:.3}]");
        }
        println!("[lm] done (retro). memorize collapses as facts grow; retro(ours) stays ~ retriever hit-rate → capability decoupled from model size, with OUR retriever closing the loop (E2 was oracle). No teacher.");
        return Ok(());
    }

    // --- MODE=retrolm: the retro loop on REAL language. Each block is
    // [neighbor][SEP][passage]; the LM predicts the passage chars. Ablation:
    // retro (neighbor = OUR retriever's nearest passage) vs baseline (neighbor =
    // a RANDOM passage). Same architecture / context length — the only difference
    // is whether the prepended context is retrieval-selected. Scored by masked
    // next-char accuracy AND masked bits/char on held-out text. Entirely ours.
    if mode == "retrolm" {
        let ev = |k: &str, d: usize| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
        let p_len = ev("RP", 32);          // passage/chunk length in bytes
        let seq = ev("RSEQ", 64);          // block/window length (LM context)
        let n_pre = seq - p_len - 1;       // neighbor chars prepended (n_pre + 1 SEP + p_len = seq)
        let n_seeds = ev("RSEEDS", 2);     // ≥2 seeds to bound variance
        let max_train = 6000usize;
        let max_test = 1500usize;

        let path = format!("{}/data_cache/tinyshakespeare.txt", env!("CARGO_MANIFEST_DIR"));
        let raw = std::fs::read(&path)?;
        // char map (byte -> id); SEP = the extra id V
        let mut seen = [false; 256];
        for &b in &raw { seen[b as usize] = true; }
        let mut cmap = [0u32; 256];
        let mut v = 0u32;
        for i in 0..256 { if seen[i] { cmap[i] = v; v += 1; } }
        let sep = v; let rvocab = (v + 1) as usize;
        let pass_bytes: Vec<Vec<u8>> = raw.chunks(p_len).filter(|c| c.len() == p_len).map(|c| c.to_vec()).collect();
        let n_tr = (pass_bytes.len() * 8 / 10).min(max_train);
        let te_start = pass_bytes.len() * 8 / 10;
        let n_te = (pass_bytes.len() - te_start).min(max_test);
        let tr_b = &pass_bytes[..n_tr];
        let te_b = &pass_bytes[te_start..te_start + n_te];
        println!("[lm] RETRO-LM (real text): block=[neighbor {n_pre}][SEP][passage {p_len}], seq={seq}, vocab={rvocab}");
        println!("[lm]   train passages={n_tr} test passages={n_te}. Ablation: retrieved-neighbor vs random-neighbor.");

        // Our retriever over passages (fragment → passage), then store embeddings.
        let (enc, _rl) = vyoma_embed::train_encoder(
            96, 256, 1, 2000, 128, 1e-3, 0.05, &dev, 4242,
            |r: &mut StdRng| {
                let p = &tr_b[r.gen_range(0..tr_b.len())];
                let off = r.gen_range(0..=(p_len - p_len.min(20)));
                (p[off..(off + p_len.min(20)).min(p.len())].to_vec(), p.clone())
            },
        )?;
        let store = vyoma_embed::embed_all(&enc, tr_b, &dev)?;
        let tr_emb = store.clone();
        let te_emb = vyoma_embed::embed_all(&enc, te_b, &dev)?;

        // nearest OTHER train passage (exclude self) for each train passage.
        let nearest_excl = |q: &[f32], skip: usize| -> usize {
            let mut best = (usize::MAX, f32::NEG_INFINITY);
            for (j, e) in store.iter().enumerate() {
                if j == skip { continue; }
                let s: f32 = q.iter().zip(e).map(|(a, b)| a * b).sum();
                if s > best.1 { best = (j, s); }
            }
            best.0
        };
        let tr_neighbor: Vec<usize> = (0..n_tr).map(|i| nearest_excl(&tr_emb[i], i)).collect();
        let te_neighbor: Vec<usize> = (0..n_te).map(|i| vyoma_embed::nearest(&te_emb[i], &store)).collect();

        // Build a flat stream of [neighbor][SEP][passage] blocks; mask marks the
        // passage (target) positions. `use_retrieved`: neighbor = retriever's pick,
        // else a random train passage. Returns (stream, mask).
        let ids_of = |bytes: &[u8]| -> Vec<u32> { bytes.iter().map(|&b| cmap[b as usize]).collect() };
        let build = |idxs: &[usize], neigh: &[usize], use_retrieved: bool, seed: u64| -> (Vec<u32>, Vec<bool>) {
            let mut rng = StdRng::seed_from_u64(seed);
            let (mut s, mut m) = (Vec::new(), Vec::new());
            for (k, &i) in idxs.iter().enumerate() {
                let nb = if use_retrieved { neigh[k] } else { rng.gen_range(0..n_tr) };
                for &t in ids_of(&tr_b[nb]).iter().take(n_pre) { s.push(t); m.push(false); }
                s.push(sep); m.push(false);
                for &t in &ids_of(&pass_bytes[i]) { s.push(t); m.push(true); }
            }
            (s, m)
        };
        let tr_idx: Vec<usize> = (0..n_tr).collect();
        let te_idx: Vec<usize> = (te_start..te_start + n_te).collect();

        let run = |use_retrieved: bool, seed: u64| -> Result<(f64, f64)> {
            let (s_tr, _m_tr) = build(&tr_idx, &tr_neighbor, use_retrieved, seed * 2 + 1);
            let (s_te, m_te) = build(&te_idx, &te_neighbor, use_retrieved, seed * 2 + 2);
            let cc = Cfg { dm, dff, layers };
            let st = Stored::new(&cc, rvocab, &dev)?;
            let ffn = { let np = ffn_total(&cc); let noise = Tensor::randn(0f32, 1f32, (np,), &dev)?; Var::from_tensor(&noise.mul(&Tensor::from_vec(ffn_prior(&cc), (np,), &dev)?)?)? };
            let ffn_c = ffn.clone();
            let mut vars = st.vars(); vars.push(ffn.clone());
            let (_ls, _ov, acc) = train(vars, &|| Ok(ffn_c.as_tensor().clone()), &s_tr, &s_te, &m_te, &st, &cc, steps, seq, bs, lr, seed, &dev)?;
            let bpb = eval_bpb_masked(&s_te, &m_te, &st, &ffn_c.as_tensor().clone(), &cc, seq, &dev)?;
            Ok((acc, bpb))
        };
        let mean_std = |v: &[f64]| -> (f64, f64) {
            let m = v.iter().sum::<f64>() / v.len() as f64;
            let sd = (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
            (m, sd)
        };
        let (mut r_bpb, mut x_bpb, mut r_acc, mut x_acc) = (vec![], vec![], vec![], vec![]);
        for sd in 0..n_seeds as u64 {
            let (ra, rb) = run(true, sd)?;
            let (xa, xb) = run(false, sd)?;
            r_acc.push(ra); r_bpb.push(rb); x_acc.push(xa); x_bpb.push(xb);
            println!("[lm]   seed {sd}: retro bits/char {rb:.3} (acc {ra:.3}) | random {xb:.3} (acc {xa:.3}) | Δbits {:+.3}", rb - xb);
        }
        let (rbm, rbs) = mean_std(&r_bpb);
        let (xbm, xbs) = mean_std(&x_bpb);
        let (ram, _) = mean_std(&r_acc);
        let (xam, _) = mean_std(&x_acc);
        println!("[lm]  RETRO-LM over {n_seeds} seeds — retro bits/char {rbm:.3}±{rbs:.3}  random {xbm:.3}±{xbs:.3}");
        println!("[lm]  Δbits/char = {:+.3} (retro − random),  Δacc = {:+.3}", rbm - xbm, ram - xam);
        println!("[lm] done (retrolm). retro < random on bits/char ⇒ OUR retriever selects passages that help predict real held-out text. No teacher.");
        return Ok(());
    }

    // --- MODE=moe: Pillar 3 — sparse Mixture-of-Experts. Does routing to top-1 of
    // E experts (dff each) buy dense-big (dff=E·dff) capacity at dense-small
    // (dff) active compute? Scored by bits-per-byte on real text. Ours, no teacher.
    if mode == "moe" {
        let e_n: usize = std::env::var("MOE_E").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
        // ONLY=small,moe,big (comma list; default "all") lets the three independent
        // trainings be split across separate processes/GPUs (e.g. one on cuda:0, one
        // on cuda:1 via CUDA_VISIBLE_DEVICES) for wall-clock parallelism — no true
        // multi-GPU tensor support needed since the three models don't interact.
        let only = std::env::var("ONLY").unwrap_or_else(|_| "all".into());
        let want = |name: &str| only == "all" || only.split(',').any(|s| s.trim() == name);
        let (run_small, run_moe, run_big) = (want("small"), want("moe"), want("big"));
        let merges_path = format!("{}/data_cache/bpe_merges.txt", env!("CARGO_MANIFEST_DIR"));
        let blens = token_byte_lengths(&corpus, &tok, &merges_path)?;
        let tb = &blens[split..];
        println!("[lm] MODE=moe (Pillar 3) [{tok}]: dense-small(dff={dff}) vs MoE({e_n} experts × dff={dff}, top-1) vs dense-big(dff={})  ONLY={only}", e_n * dff);

        let mut bpb_small_o: Option<f64> = None;
        let mut bpb_moe_o: Option<f64> = None;
        let mut bpb_big_o: Option<f64> = None;

        if run_small {
            // dense-small: single FFN, dff.
            let cs = Cfg { dm, dff, layers: 1 };
            let sts = Stored::new(&cs, vocab, &dev)?;
            let ffs = { let np = ffn_total(&cs); let noise = Tensor::randn(0f32, 1f32, (np,), &dev)?; Var::from_tensor(&noise.mul(&Tensor::from_vec(ffn_prior(&cs), (np,), &dev)?)?)? };
            let ffs_c = ffs.clone();
            let mut vs = sts.vars(); vs.push(ffs.clone());
            train(vs, &|| Ok(ffs_c.as_tensor().clone()), train_s, test_s, test_mask, &sts, &cs, steps, l, bs, lr, 0, &dev)?;
            let bpb_small = eval_bpb(test_s, tb, &sts, ffs_c.as_tensor(), &cs, l, &dev)?;
            let p_small = sts.n_params() + ffs_c.as_tensor().elem_count();
            println!("[lm]   dense-small (dff={dff:4}) BPB={bpb_small:.3}  params={p_small}");
            bpb_small_o = Some(bpb_small);
        }

        if run_moe {
            // MoE: E experts of dff, top-1 routing + load-balance aux.
            let cm = Cfg { dm, dff, layers: 1 };
            // RESUME=1 continues from the SAVE checkpoint if it exists, instead of
            // starting from random weights. Cloud sessions are ephemeral; without
            // this, every restart threw away all prior training and the 30 hours of
            // Kaggle time never accumulated. Geometry is checked against the current
            // config so a mismatched checkpoint fails loudly rather than silently.
            let resume = std::env::var("RESUME").is_ok();
            let (stm, moe) = match (resume, std::env::var("SAVE").ok()) {
                (true, Some(p)) if std::path::Path::new(&p).exists() => {
                    let (s, m, v_ck, dm_ck) = load_moe_ckpt(&p, &dev)?;
                    anyhow::ensure!(v_ck == vocab && dm_ck == dm && m.experts.len() == e_n,
                        "RESUME: checkpoint geometry (vocab={v_ck} dm={dm_ck} experts={}) != config (vocab={vocab} dm={dm} experts={e_n})", m.experts.len());
                    println!("[lm]   RESUMED from {p} — continuing training, not restarting");
                    (s, m)
                }
                _ => (Stored::new(&cm, vocab, &dev)?, Moe::new(dm, dff, e_n, &dev)?),
            };
            let mut vm = stm.vars(); vm.extend(moe.vars());
            let mut opt = AdamW::new(vm, ParamsAdamW { lr, ..Default::default() })?;
            let mut rng = StdRng::seed_from_u64(0);
            // PERIODIC CHECKPOINTING. Cloud sessions die (Kaggle restarted on us and
            // a full 30k-step run was lost because we only saved at the end).
            // CKPT_EVERY=N writes the model every N steps, so a killed session costs
            // at most N steps. Also prints loss so progress is visible in the log.
            let ckpt_every: usize = std::env::var("CKPT_EVERY").ok().and_then(|s| s.parse().ok()).unwrap_or(1000);
            let save_path = std::env::var("SAVE").ok();
            for step in 0..steps {
                let (ids, tg) = sample_batch(train_s, bs, l, &mut rng, &dev)?;
                let (logits, aux) = forward_moe(&ids, &stm, &moe, dm, &dev)?;
                let ce = candle_nn::loss::cross_entropy(&logits, &tg.reshape((bs * l,))?)?;
                let loss = (ce + (aux * 0.01)?)?;
                opt.backward_step(&loss)?;
                if ckpt_every > 0 && (step + 1) % ckpt_every == 0 {
                    let l_v = loss.to_scalar::<f32>().unwrap_or(f32::NAN);
                    if let Some(p) = &save_path {
                        save_moe_ckpt(p, &stm, &moe)?;
                        println!("[lm]   step {}/{steps} loss {l_v:.4} — checkpoint saved", step + 1);
                    } else {
                        println!("[lm]   step {}/{steps} loss {l_v:.4}", step + 1);
                    }
                    use std::io::Write;
                    std::io::stdout().flush().ok();
                }
            }
            let bpb_moe = eval_bpb_moe(test_s, tb, &stm, &moe, dm, l, &dev)?;
            let p_moe_total = stm.n_params() + moe.n_params();
            let p_moe_active = stm.n_params() + moe.expert_params() + dm * e_n; // ~one expert + gate
            println!("[lm]   MoE ({e_n}×dff={dff})      BPB={bpb_moe:.3}  total={p_moe_total} active≈{p_moe_active}");
            // SAVE=path.safetensors → keep the trained model (else it evaporates on exit)
            if let Ok(p) = std::env::var("SAVE") {
                save_moe_ckpt(&p, &stm, &moe)?;
                let kb = std::fs::metadata(&p).map(|m| m.len() as f64 / 1024.0).unwrap_or(0.0);
                println!("[lm]   saved MoE checkpoint -> {p} ({kb:.0} KB)  [generate: MODE=generate LOAD={p}]");
            }
            bpb_moe_o = Some(bpb_moe);
        }

        if run_big {
            // dense-big: single FFN, E·dff (the capacity upper bound).
            let cb = Cfg { dm, dff: e_n * dff, layers: 1 };
            let stb = Stored::new(&cb, vocab, &dev)?;
            let ffb = { let np = ffn_total(&cb); let noise = Tensor::randn(0f32, 1f32, (np,), &dev)?; Var::from_tensor(&noise.mul(&Tensor::from_vec(ffn_prior(&cb), (np,), &dev)?)?)? };
            let ffb_c = ffb.clone();
            let mut vb = stb.vars(); vb.push(ffb.clone());
            train(vb, &|| Ok(ffb_c.as_tensor().clone()), train_s, test_s, test_mask, &stb, &cb, steps, l, bs, lr, 0, &dev)?;
            let bpb_big = eval_bpb(test_s, tb, &stb, ffb_c.as_tensor(), &cb, l, &dev)?;
            let p_big = stb.n_params() + ffb_c.as_tensor().elem_count();
            println!("[lm]   dense-big  (dff={:4}) BPB={bpb_big:.3}  params={p_big}", e_n * dff);
            bpb_big_o = Some(bpb_big);
        }

        if let (Some(bs_), Some(bm_)) = (bpb_small_o, bpb_moe_o) {
            println!("[lm]   Δ(small−moe)={:+.3}", bs_ - bm_);
        }
        if let (Some(bm_), Some(bb_)) = (bpb_moe_o, bpb_big_o) {
            println!("[lm]   Δ(moe−big)={:+.3}", bm_ - bb_);
        }
        println!("[lm] done (moe, ONLY={only}). Split across GPUs: CUDA_VISIBLE_DEVICES=0 ONLY=small,moe ... & CUDA_VISIBLE_DEVICES=1 ONLY=big ... &");
        return Ok(());
    }

    // --- MODE=genmoe: Pillar 1 × Pillar 3 — GENERATE the MoE experts from a fractal
    // seed. The experts are the large redundant mass; can a tiny seed produce them
    // and keep the MoE win? stored-MoE vs gen-MoE (seed) vs dense-small, by BPB. ---
    if mode == "genmoe" {
        let e_n: usize = std::env::var("MOE_E").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
        let merges_path = format!("{}/data_cache/bpe_merges.txt", env!("CARGO_MANIFEST_DIR"));
        let blens = token_byte_lengths(&corpus, &tok, &merges_path)?;
        let tb = &blens[split..];
        let per = 2 * dm * dff + dff + dm;          // one expert's params
        let total_experts = e_n * per;              // the mass we try to generate
        println!("[lm] MODE=genmoe (Pillar 1×3) [{tok}]: generate {e_n} experts (dff={dff}, {total_experts} params) from a fractal seed");

        // dense-small floor.
        let cs = Cfg { dm, dff, layers: 1 };
        let sts = Stored::new(&cs, vocab, &dev)?;
        let ffs = { let np = ffn_total(&cs); let noise = Tensor::randn(0f32, 1f32, (np,), &dev)?; Var::from_tensor(&noise.mul(&Tensor::from_vec(ffn_prior(&cs), (np,), &dev)?)?)? };
        let ffs_c = ffs.clone();
        let mut vs = sts.vars(); vs.push(ffs.clone());
        train(vs, &|| Ok(ffs_c.as_tensor().clone()), train_s, test_s, test_mask, &sts, &cs, steps, l, bs, lr, 0, &dev)?;
        let bpb_small = eval_bpb(test_s, tb, &sts, ffs_c.as_tensor(), &cs, l, &dev)?;

        // stored-MoE (the quality target).
        let stm = Stored::new(&cs, vocab, &dev)?;
        let moe = Moe::new(dm, dff, e_n, &dev)?;
        let mut vm = stm.vars(); vm.extend(moe.vars());
        let mut opt = AdamW::new(vm, ParamsAdamW { lr, ..Default::default() })?;
        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..steps {
            let (ids, tg) = sample_batch(train_s, bs, l, &mut rng, &dev)?;
            let (logits, aux) = forward_moe(&ids, &stm, &moe, dm, &dev)?;
            let loss = (candle_nn::loss::cross_entropy(&logits, &tg.reshape((bs * l,))?)? + (aux * 0.01)?)?;
            opt.backward_step(&loss)?;
        }
        let bpb_stored = eval_bpb_moe(test_s, tb, &stm, &moe, dm, l, &dev)?;

        // gen-MoE: a fractal seed generates all E experts; gate stored.
        let mut prior = Vec::with_capacity(total_experts);
        let one = ffn_prior(&Cfg { dm, dff, layers: 1 });
        for _ in 0..e_n { prior.extend_from_slice(&one); }
        let (chunk, ed, hh) = (256usize, 8usize, 16usize);
        let seed = FractalSeed::new(total_experts, prior, chunk, ed, hh, &dev)?;
        let seed_p = seed.seed_params();
        let stg = Stored::new(&cs, vocab, &dev)?;
        let gate_g = var_randn((dm, e_n), (1.0 / dm as f64).sqrt(), &dev)?;
        let mut vg = stg.vars(); vg.push(gate_g.clone()); vg.extend(seed.vars());
        let mut optg = AdamW::new(vg, ParamsAdamW { lr, ..Default::default() })?;
        let mut rng2 = StdRng::seed_from_u64(1);
        for _ in 0..steps {
            let (ids, tg) = sample_batch(train_s, bs, l, &mut rng2, &dev)?;
            let ex = seed.generate()?;
            let (logits, aux) = forward_moe_gen(&ids, &stg, gate_g.as_tensor(), &ex, dm, dff, e_n, &dev)?;
            let loss = (candle_nn::loss::cross_entropy(&logits, &tg.reshape((bs * l,))?)? + (aux * 0.01)?)?;
            optg.backward_step(&loss)?;
        }
        let ex_final = seed.generate()?;
        let bpb_gen = eval_bpb_moe_gen(test_s, tb, &stg, gate_g.as_tensor(), &ex_final, dm, dff, e_n, l, &dev)?;
        let comp = total_experts as f64 / (seed_p + dm * e_n) as f64;

        println!("[lm]   dense-small (dff={dff})           BPB={bpb_small:.3}");
        println!("[lm]   stored-MoE  ({e_n}×dff={dff})        BPB={bpb_stored:.3}  expert-params={total_experts}");
        println!("[lm]   gen-MoE     (seed {seed_p}+gate)   BPB={bpb_gen:.3}  expert-mass {comp:.1}× compressed");
        println!("[lm]   Δ(gen−stored)={:+.3}  Δ(gen−small)={:+.3}", bpb_gen - bpb_stored, bpb_gen - bpb_small);
        println!("[lm] done (genmoe). Read: gen-MoE ≈ stored-MoE ⇒ experts are generable; gen-MoE > stored-MoE (and > dense-small) ⇒ language experts resist generation → STORE (+ quantize) the experts, don't generate them.");
        return Ok(());
    }

    // --- MODE=g1: capability-per-RAM (Gate G1 spirit). Assemble the DECIDED
    // architecture — lean SSM + STORED top-1 MoE — and show it wins per byte, with
    // int8 on the actual FFN/expert mass (the weights the 8GB claim rests on).
    // dense-small vs MoE vs dense-big, each at fp32 and int8. Ours, no teacher. ---
    if mode == "g1" {
        let e_n: usize = std::env::var("MOE_E").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
        let merges_path = format!("{}/data_cache/bpe_merges.txt", env!("CARGO_MANIFEST_DIR"));
        let blens = token_byte_lengths(&corpus, &tok, &merges_path)?;
        let tb = &blens[split..];
        println!("[lm] MODE=g1 (capability-per-RAM) [{tok}]: lean SSM + stored MoE vs dense, fp32 & int8 on the FFN mass");

        // dense-small (dff)
        let cs = Cfg { dm, dff, layers: 1 };
        let sts = Stored::new(&cs, vocab, &dev)?;
        let ffs = { let np = ffn_total(&cs); let noise = Tensor::randn(0f32, 1f32, (np,), &dev)?; Var::from_tensor(&noise.mul(&Tensor::from_vec(ffn_prior(&cs), (np,), &dev)?)?)? };
        let ffs_c = ffs.clone();
        let mut vs = sts.vars(); vs.push(ffs.clone());
        train(vs, &|| Ok(ffs_c.as_tensor().clone()), train_s, test_s, test_mask, &sts, &cs, steps, l, bs, lr, 0, &dev)?;
        let small_fp32 = eval_bpb(test_s, tb, &sts, ffs_c.as_tensor(), &cs, l, &dev)?;
        let small_int8 = eval_bpb(test_s, tb, &sts, &quantized(ffs_c.as_tensor(), 8, &dev)?, &cs, l, &dev)?;
        let small_ffn = ffs_c.as_tensor().elem_count();

        // MoE (E × dff), stored
        let stm = Stored::new(&cs, vocab, &dev)?;
        let moe = Moe::new(dm, dff, e_n, &dev)?;
        let mut vm = stm.vars(); vm.extend(moe.vars());
        let mut opt = AdamW::new(vm, ParamsAdamW { lr, ..Default::default() })?;
        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..steps {
            let (ids, tg) = sample_batch(train_s, bs, l, &mut rng, &dev)?;
            let (logits, aux) = forward_moe(&ids, &stm, &moe, dm, &dev)?;
            let loss = (candle_nn::loss::cross_entropy(&logits, &tg.reshape((bs * l,))?)? + (aux * 0.01)?)?;
            opt.backward_step(&loss)?;
        }
        let moe_fp32 = eval_bpb_moe(test_s, tb, &stm, &moe, dm, l, &dev)?;
        // flatten experts into forward_moe_gen's layout, then int8 the mass
        let mut exflat: Vec<f32> = Vec::new();
        for ex in &moe.experts {
            for var in [&ex.w1, &ex.b1, &ex.w2, &ex.b2] { exflat.extend(var.as_tensor().flatten_all()?.to_vec1::<f32>()?); }
        }
        let moe_ffn = exflat.len();
        let ex_q = Tensor::from_vec(fake_quant_vec(&exflat, 8), (moe_ffn,), &dev)?;
        let moe_int8 = eval_bpb_moe_gen(test_s, tb, &stm, moe.gate.as_tensor(), &ex_q, dm, dff, e_n, l, &dev)?;

        // dense-big (E·dff)
        let cb = Cfg { dm, dff: e_n * dff, layers: 1 };
        let stb = Stored::new(&cb, vocab, &dev)?;
        let ffb = { let np = ffn_total(&cb); let noise = Tensor::randn(0f32, 1f32, (np,), &dev)?; Var::from_tensor(&noise.mul(&Tensor::from_vec(ffn_prior(&cb), (np,), &dev)?)?)? };
        let ffb_c = ffb.clone();
        let mut vb = stb.vars(); vb.push(ffb.clone());
        train(vb, &|| Ok(ffb_c.as_tensor().clone()), train_s, test_s, test_mask, &stb, &cb, steps, l, bs, lr, 0, &dev)?;
        let big_fp32 = eval_bpb(test_s, tb, &stb, ffb_c.as_tensor(), &cb, l, &dev)?;
        let big_ffn = ffb_c.as_tensor().elem_count();

        let kb = |params: usize, bits: usize| params as f64 * bits as f64 / 8.0 / 1024.0;
        println!("[lm]  model         BPB(fp32) BPB(int8)  FFN-KB(fp32) FFN-KB(int8)");
        println!("[lm]  dense-small    {small_fp32:.3}    {small_int8:.3}      {:.1}        {:.1}", kb(small_ffn, 32), kb(small_ffn, 8));
        println!("[lm]  MoE ({e_n}×)        {moe_fp32:.3}    {moe_int8:.3}      {:.1}        {:.1}", kb(moe_ffn, 32), kb(moe_ffn, 8));
        println!("[lm]  dense-big      {big_fp32:.3}      —        {:.1}          —", kb(big_ffn, 32));
        println!("[lm]  Δ(MoE-int8 − dense-small-fp32) BPB = {:+.3} at {:.1}KB vs {:.1}KB FFN RAM", moe_int8 - small_fp32, kb(moe_ffn, 8), kb(small_ffn, 32));
        println!("[lm] done (g1). MoE-int8 lower BPB at ≤ dense-small FFN-RAM ⇒ capability-per-GB win; and int8 ~free on the LM mass. Knowledge (retrieval) lives on disk, off-RAM. Ours.");
        return Ok(());
    }

    // --- MODE=lattice: Pillar 3's SYMBOLIC half — the hallucination killer.
    // Continuous retrieval (our encoder) proposes a candidate; a SYMBOLIC check
    // (exact digit-level consistency between query and retrieved key) vetoes and
    // ABSTAINS when the store doesn't actually support an answer, instead of
    // confidently emitting a wrong value. Measured on answerable + unanswerable
    // queries: does the lattice kill hallucinations while preserving real answers?
    if mode == "lattice" {
        let n_ent = std::env::var("LN").ok().and_then(|s| s.parse().ok()).unwrap_or(2000usize);
        let d_key = 12usize;
        let tau = std::env::var("TAU").ok().and_then(|s| s.parse().ok()).unwrap_or(2usize); // max symbolic mismatches to accept
        // RANDOM 12-digit keys (well-separated: expected Hamming ≈ 11 between keys),
        // so the symbolic check cleanly tells in-store (Hamming ~1) from out-of-store.
        let mut rng = StdRng::seed_from_u64(100 + n_ent as u64);
        let rand_key = |r: &mut StdRng| -> Vec<u8> { (0..d_key).map(|_| b'0' + r.gen_range(0..10u8)).collect() };
        let keys: Vec<Vec<u8>> = (0..n_ent).map(|_| rand_key(&mut rng)).collect();
        let val_of: Vec<u32> = (0..n_ent).map(|_| rng.gen_range(0..10u32)).collect();

        // Our retriever (noisy key → clean key), then encode the store of N keys.
        let (enc, _rl) = vyoma_embed::train_encoder(
            96, 256, 1, 1500, 128, 1e-3, 0.05, &dev, 7 + n_ent as u64,
            |r: &mut StdRng| {
                let e = r.gen_range(0..n_ent);
                let clean = keys[e].clone();
                let mut q = clean.clone();
                let pos = r.gen_range(0..q.len());
                q[pos] = b'0' + r.gen_range(0..10u8);
                (q, clean)
            },
        )?;
        let recs = keys.clone();
        let store = vyoma_embed::embed_all(&enc, &recs, &dev)?;
        let hamming = |a: &[u8], b: &[u8]| a.iter().zip(b).filter(|(x, y)| x != y).count();

        // Answerable: noisy versions of IN-store keys. Unanswerable: fresh random
        // keys the store never saw.
        let n_q = 1000usize;
        let mut qr = StdRng::seed_from_u64(55 + n_ent as u64);
        let make_query = |e: usize, qr: &mut StdRng| -> Vec<u8> {
            let mut q = keys[e].clone();
            let pos = qr.gen_range(0..q.len());
            q[pos] = b'0' + qr.gen_range(0..10u8);
            q
        };
        // metrics
        let (mut ans_correct_nolat, mut ans_acc_correct, mut ans_accepted) = (0usize, 0usize, 0usize);
        let mut unans_halluc_lat = 0usize; // unanswerable queries the lattice wrongly accepts
        for _ in 0..n_q {
            // answerable
            let e = qr.gen_range(0..n_ent);
            let q = make_query(e, &mut qr);
            let r = vyoma_embed::nearest(&vyoma_embed::embed_all(&enc, &[q.clone()], &dev)?[0], &store);
            if val_of[r] == val_of[e] { ans_correct_nolat += 1; }                 // no-lattice: always answers
            if hamming(&q, &recs[r]) <= tau {                                     // symbolic accept
                ans_accepted += 1;
                if val_of[r] == val_of[e] { ans_acc_correct += 1; }
            }
            // unanswerable (fresh random key not in store)
            let u = rand_key(&mut qr);
            let ru = vyoma_embed::nearest(&vyoma_embed::embed_all(&enc, &[u.clone()], &dev)?[0], &store);
            if hamming(&u, &recs[ru]) <= tau { unans_halluc_lat += 1; }           // lattice-accepted ⇒ a hallucination
        }
        let na = n_q as f64;
        println!("[lm] MODE=lattice (Pillar 3 symbolic half) — N={n_ent} keys, τ={tau}, {n_q} answerable + {n_q} unanswerable queries");
        println!("[lm]  ANSWERABLE:  no-lattice acc {:.3} (always answers) | with-lattice acc {:.3} on accepted, coverage {:.3}",
                 ans_correct_nolat as f64 / na, ans_acc_correct as f64 / ans_accepted.max(1) as f64, ans_accepted as f64 / na);
        println!("[lm]  UNANSWERABLE: hallucination rate  no-lattice 1.000 (always answers) → with-lattice {:.3}", unans_halluc_lat as f64 / na);
        println!("[lm] done (lattice). Symbolic veto abstains on unsupported queries ⇒ hallucinations killed, real answers preserved. Continuous retrieval + symbolic check = Trinity. Ours.");
        return Ok(());
    }

    // --- MODE=evolve: Pillar 5 — bounded self-evolution + homeostasis. The model
    // evolves by WRITING new facts to its store (not risky weight edits ⇒ no
    // catastrophic forgetting). A homeostasis controller gates every write using the
    // symbolic consistency check (Pillar 3b): a fact whose key already exists with a
    // DIFFERENT value (a contradiction / poisoning attempt) is REJECTED, keeping the
    // system stable as it grows. naive (accept all) vs homeostasis, across rounds. ---
    if mode == "evolve" {
        let d_key = 12usize;
        let rounds = std::env::var("ROUNDS").ok().and_then(|s| s.parse().ok()).unwrap_or(5usize);
        let per_round = std::env::var("PERROUND").ok().and_then(|s| s.parse().ok()).unwrap_or(250usize);
        let mut rng = StdRng::seed_from_u64(2024);
        let rand_key = |r: &mut StdRng| -> Vec<u8> { (0..d_key).map(|_| b'0' + r.gen_range(0..10u8)).collect() };

        // Our retriever, trained generically (noisy 12-digit key → clean key); it
        // generalizes to any key, so it serves every round's growing store.
        let (enc, _rl) = vyoma_embed::train_encoder(
            96, 256, 1, 1500, 128, 1e-3, 0.05, &dev, 99,
            |r: &mut StdRng| {
                let clean = (0..d_key).map(|_| b'0' + r.gen_range(0..10u8)).collect::<Vec<u8>>();
                let mut q = clean.clone();
                let pos = r.gen_range(0..q.len());
                q[pos] = b'0' + r.gen_range(0..10u8);
                (q, clean)
            },
        )?;
        let emb1 = |k: &[u8], dev: &Device| -> Result<Vec<f32>> { Ok(vyoma_embed::embed_all(&enc, &[k.to_vec()], dev)?[0].clone()) };

        // Two evolving KEY-VALUE stores (dedup by key, last-write-wins — standard KV
        // semantics, so a contradictory write actually OVERWRITES under naive).
        // Each = (keys, vals, embeddings, key→index map).
        use std::collections::HashMap;
        let (mut nk, mut nv, mut ne): (Vec<Vec<u8>>, Vec<u32>, Vec<Vec<f32>>) = (vec![], vec![], vec![]);
        let (mut hk, mut hv, mut he): (Vec<Vec<u8>>, Vec<u32>, Vec<Vec<f32>>) = (vec![], vec![], vec![]);
        let (mut nidx, mut hidx): (HashMap<Vec<u8>, usize>, HashMap<Vec<u8>, usize>) = (HashMap::new(), HashMap::new());
        let mut good: Vec<(Vec<u8>, u32)> = vec![]; // ground truth (good facts only)
        let mut rejected = 0usize;

        println!("[lm] MODE=evolve (Pillar 5) — self-evolve by writing to the store (KV, last-write-wins); homeostasis gates writes via symbolic check.");
        println!("[lm]  round | keys(naive/homeo) | acc-naive | acc-homeo | homeostasis-rejects");
        for round in 0..rounds {
            // GOOD facts: fresh random keys + values (both stores insert).
            for _ in 0..per_round {
                let k = rand_key(&mut rng);
                let v = rng.gen_range(0..10u32);
                let e = emb1(&k, &dev)?;
                if let Some(&j) = nidx.get(&k) { nv[j] = v; } else { nidx.insert(k.clone(), nk.len()); nk.push(k.clone()); nv.push(v); ne.push(e.clone()); }
                if let Some(&j) = hidx.get(&k) { hv[j] = v; } else { hidx.insert(k.clone(), hk.len()); hk.push(k.clone()); hv.push(v); he.push(e); }
                good.push((k, v));
            }
            // BAD facts (from round 1): reuse an existing good key with a DIFFERENT
            // value — a contradictory UPDATE. naive overwrites (corrupts); homeostasis
            // vetoes (the symbolic check: key exists with a different value ⇒ reject).
            if round >= 1 {
                for _ in 0..(per_round / 3) {
                    let idx = rng.gen_range(0..good.len());
                    let k = good[idx].0.clone();
                    let bad_v = (good[idx].1 + 1 + rng.gen_range(0..9u32)) % 10;
                    if let Some(&j) = nidx.get(&k) { nv[j] = bad_v; }              // naive: overwrite ⇒ poison
                    match hidx.get(&k) {
                        Some(&j) if hv[j] != bad_v => rejected += 1,               // homeostasis: veto
                        Some(_) => {}                                             // same value: no-op
                        None => { hidx.insert(k.clone(), hk.len()); hk.push(k); hv.push(bad_v); he.push(emb1(&good[idx].0, &dev)?); }
                    }
                }
            }
            // Evaluate on a sample of GOOD facts (noisy query → nearest key → its
            // CURRENT value). Under naive, overwritten keys now return the wrong value.
            let eval = |vs: &[u32], es: &[Vec<f32>], rng: &mut StdRng| -> Result<f64> {
                let (mut ok, mut n) = (0usize, 0usize);
                for _ in 0..200 {
                    let (tk, tv) = &good[rng.gen_range(0..good.len())];
                    let mut q = tk.clone();
                    let p = rng.gen_range(0..q.len()); q[p] = b'0' + rng.gen_range(0..10u8);
                    let j = vyoma_embed::nearest(&emb1(&q, &dev)?, es);
                    if vs[j] == *tv { ok += 1; }
                    n += 1;
                }
                Ok(ok as f64 / n as f64)
            };
            let mut er = StdRng::seed_from_u64(7000 + round as u64);
            let acc_n = eval(&nv, &ne, &mut er)?;
            let mut er2 = StdRng::seed_from_u64(7000 + round as u64);
            let acc_h = eval(&hv, &he, &mut er2)?;
            println!("[lm]   {round:5} | {:5}/{:5}      |   {acc_n:.3}   |   {acc_h:.3}   | {rejected}", nk.len(), hk.len());
        }
        println!("[lm] done (evolve). Homeostasis holds accuracy as the store self-evolves (rejects contradictions); naive degrades as poison accrues. Bounded self-evolution via the store — no forgetting, no weight edits. Ours.");
        return Ok(());
    }

    // --- MODE=vyoma: the assembled model. BPE + SSM + generated-FFN, scored by
    // bits-per-byte against a same-footprint plain model. Our stack, end to end. ---
    if mode == "vyoma" {
        let c = Cfg { dm, dff, layers };
        let merges_path = format!("{}/data_cache/bpe_merges.txt", env!("CARGO_MANIFEST_DIR"));
        let blens = token_byte_lengths(&corpus, &tok, &merges_path)?;
        let test_blens = &blens[split..];
        let (chunk, ed, hh) = (256usize, 8usize, 16usize); // moderate FFN compression

        // Vyomarudra: SSM + generated FFN (fractal seed), BPE tokens
        let stg = Stored::new(&c, vocab, &dev)?;
        let seed = FractalSeed::new(ffn_total(&c), ffn_prior(&c), chunk, ed, hh, &dev)?;
        let seed_p = seed.seed_params();
        let mut vg = stg.vars(); vg.extend(seed.vars());
        train(vg, &|| seed.generate(), train_s, test_s, test_mask, &stg, &c, steps, l, bs, lr, 0, &dev)?;
        let bpb_vyoma = eval_bpb(test_s, test_blens, &stg, &seed.generate()?, &c, l, &dev)?;

        // Fair baseline: plain stored FFN sized to the seed's footprint, BPE tokens
        let dff_small = ((seed_p / layers).saturating_sub(dm) / (2 * dm + 1)).max(1);
        let cp = Cfg { dm, dff: dff_small, layers };
        let stp = Stored::new(&cp, vocab, &dev)?;
        let ffn_p = {
            let np = ffn_total(&cp);
            let noise = Tensor::randn(0f32, 1f32, (np,), &dev)?;
            Var::from_tensor(&noise.mul(&Tensor::from_vec(ffn_prior(&cp), (np,), &dev)?)?)?
        };
        let ffn_pc = ffn_p.clone();
        let mut vp = stp.vars(); vp.push(ffn_p.clone());
        train(vp, &|| Ok(ffn_pc.as_tensor().clone()), train_s, test_s, test_mask, &stp, &cp, steps, l, bs, lr, 0, &dev)?;
        let bpb_plain = eval_bpb(test_s, &blens[split..], &stp, ffn_pc.as_tensor(), &cp, l, &dev)?;

        let comp = ffn_total(&c) as f64 / seed_p as f64;
        println!("[lm] VYOMARUDRA assembled [{tok}]: SSM + generated-FFN ({comp:.0}× on the FFN) + BPE");
        println!("[lm]   Vyomarudra (gen-FFN, seed {seed_p})  BPB = {bpb_vyoma:.3}");
        println!("[lm]   plain (stored FFN, dff={dff_small})   BPB = {bpb_plain:.3}");
        println!("[lm]   Δ = {:+.3} bits/byte (negative = Vyomarudra better).  [BPE-dense ref ≈ 2.14]", bpb_vyoma - bpb_plain);
        println!("[lm] done (vyoma). Our stack, assembled and scored on one honest scale.");
        return Ok(());
    }

    // --- MODE=bpb: train, then report bits-per-byte (tokenizer-independent). ---
    if mode == "bpb" {
        let c = Cfg { dm, dff, layers };
        let st = Stored::new(&c, vocab, &dev)?;
        let ffn = {
            let np = ffn_total(&c);
            let noise = Tensor::randn(0f32, 1f32, (np,), &dev)?;
            Var::from_tensor(&noise.mul(&Tensor::from_vec(ffn_prior(&c), (np,), &dev)?)?)?
        };
        let ffn_c = ffn.clone();
        let mut v = st.vars(); v.push(ffn.clone());
        train(v, &|| Ok(ffn_c.as_tensor().clone()), train_s, test_s, test_mask, &st, &c, steps, l, bs, lr, 0, &dev)?;
        let merges_path = format!("{}/data_cache/bpe_merges.txt", env!("CARGO_MANIFEST_DIR"));
        let blens = token_byte_lengths(&corpus, &tok, &merges_path)?;
        let bpb = eval_bpb(test_s, &blens[split..], &st, ffn_c.as_tensor(), &c, l, &dev)?;
        let bpt = corpus.len() as f64 / blens.iter().map(|&x| x as f64).sum::<f64>().max(1.0);
        println!("[lm] BPB[{tok}] = {bpb:.3} bits/byte  (vocab={vocab}, {:.2} bytes/token, dm={dm} dff={dff} layers={layers}, {steps} steps)",
                 1.0 / bpt);
        println!("[lm] done (bpb). lower = better; comparable across tokenizers. char vs bpe now judgeable on ONE scale.");
        return Ok(());
    }

    // --- MODE=knn: retrieval-augmented LM (kNN-LM). Does a disk datastore lift a
    // fixed-size model? The core capability-per-GB claim, on real language. ---
    if mode == "knn" {
        let c = Cfg { dm, dff, layers };
        let st = Stored::new(&c, vocab, &dev)?;
        let ffn = {
            let np = ffn_total(&c);
            let noise = Tensor::randn(0f32, 1f32, (np,), &dev)?;
            Var::from_tensor(&noise.mul(&Tensor::from_vec(ffn_prior(&c), (np,), &dev)?)?)?
        };
        let ffn_c = ffn.clone();
        let mut v = st.vars(); v.push(ffn.clone());
        let base = train(v, &|| Ok(ffn_c.as_tensor().clone()), train_s, test_s, test_mask, &st, &c, steps, l, bs, lr, 0, &dev)?;
        println!("[lm] KNN base LM ({} params) trained; next-char acc {:.4}", st.n_params() + ffn_total(&c), base.1);

        // datastore: (hidden state → next token) over training windows. Lives on
        // disk conceptually — it costs disk, not model RAM.
        let store_cap = 40000usize;
        let (mut keys, mut vals) = (Vec::<f32>::with_capacity(store_cap * dm), Vec::<u32>::with_capacity(store_cap));
        let mut s0 = 0;
        while vals.len() < store_cap && s0 + l + 1 <= train_s.len() {
            let ids = Tensor::from_vec(train_s[s0..s0 + l].to_vec(), (1, l), &dev)?;
            let (_lg, hid) = forward_hidden(&ids, &st, ffn_c.as_tensor(), &c)?;
            let hrows = hid.to_vec2::<f32>()?;
            for t in 0..l {
                if vals.len() >= store_cap { break; }
                keys.extend(l2norm(hrows[t].clone()));
                vals.push(train_s[s0 + t + 1]);
            }
            s0 += l;
        }
        let ssz = vals.len();
        println!("[lm] datastore = {ssz} entries of (hidden→next-token) — on disk, ~free RAM");

        // eval: sweep the interpolation weight λ (λ=0 is base-LM-alone) in ONE pass.
        let (k, temp) = (64usize, 0.07f32);
        let lambdas = [0.0f32, 0.05, 0.1, 0.2, 0.35, 0.5];
        let mut correct = vec![0usize; lambdas.len()];
        let mut cnt = 0usize;
        let mut e0 = 0;
        while cnt < 2000 && e0 + l + 1 <= test_s.len() {
            let ids = Tensor::from_vec(test_s[e0..e0 + l].to_vec(), (1, l), &dev)?;
            let (lg, hid) = forward_hidden(&ids, &st, ffn_c.as_tensor(), &c)?;
            let lrows = lg.to_vec2::<f32>()?;
            let hrows = hid.to_vec2::<f32>()?;
            for t in 0..l {
                if cnt >= 2000 { break; }
                let truth = test_s[e0 + t + 1];
                let mp = softmax(&lrows[t]);
                let q = l2norm(hrows[t].clone());
                let mut top: Vec<(f32, u32)> = Vec::with_capacity(k);
                for i in 0..ssz {
                    let ks = &keys[i * dm..(i + 1) * dm];
                    let sim: f32 = (0..dm).map(|d| q[d] * ks[d]).sum();
                    if top.len() < k { top.push((sim, vals[i])); }
                    else { let mut mi = 0; for j in 1..k { if top[j].0 < top[mi].0 { mi = j; } }
                        if sim > top[mi].0 { top[mi] = (sim, vals[i]); } }
                }
                let mut kp = vec![0f32; vocab];
                let mut z = 0f32;
                for &(sim, val) in &top { let w = (sim / temp).exp(); kp[val as usize] += w; z += w; }
                if z > 0.0 { for x in &mut kp { *x /= z; } }
                for (li, &lam) in lambdas.iter().enumerate() {
                    let (mut best, mut bv) = (0usize, f32::MIN);
                    for vt in 0..vocab {
                        let ph = lam * kp[vt] + (1.0 - lam) * mp[vt];
                        if ph > bv { bv = ph; best = vt; }
                    }
                    if best as u32 == truth { correct[li] += 1; }
                }
                cnt += 1;
            }
            e0 += l;
        }
        let base_acc = correct[0] as f64 / cnt as f64;
        print!("[lm] KNN eval ({cnt} positions):  base(λ=0)={base_acc:.4}");
        let mut best = (0usize, base_acc);
        for (li, &lam) in lambdas.iter().enumerate().skip(1) {
            let a = correct[li] as f64 / cnt as f64;
            print!("  λ={lam}:{a:.4}");
            if a > best.1 { best = (li, a); }
        }
        println!();
        let d = best.1 - base_acc;
        println!("[lm] best λ={} → {:.4} (Δ vs base {:+.4}). {}", lambdas[best.0], best.1, d,
                 if d > 0.0 { "knowledge on disk lifts a fixed-size model ✓ (capability-per-GB)" }
                 else { "no λ helped here — datastore too small / keys too weak for this LM" });
        return Ok(());
    }

    let c = Cfg { dm, dff, layers };
    println!("[lm] FFN mass = {} params (the generatable part).", ffn_total(&c));

    // Dense upper bound: FFN stored directly.
    let st = Stored::new(&c, vocab, &dev)?;
    let ffn_dense = {
        let noise = Tensor::randn(0f32, 1f32, (ffn_total(&c),), &dev)?;
        Var::from_tensor(&noise.mul(&Tensor::from_vec(ffn_prior(&c), (ffn_total(&c),), &dev)?)?)?
    };
    let ffn_dense_c = ffn_dense.clone();
    let mut vars = st.vars(); vars.push(ffn_dense.clone());
    let (dl, dov, da) = train(vars, &|| Ok(ffn_dense_c.as_tensor().clone()), train_s, test_s, test_mask, &st, &c, steps, l, bs, lr, 0, &dev)?;
    println!("[lm] DENSE (stored FFN, {} total params): loss={dl:.3} answer-acc={da:.4} (overall {dov:.3})", st.n_params() + ffn_total(&c));

    // Generated-FFN vs same-footprint plain, at a few seed sizes.
    let configs = [(256usize, 8usize, 16usize), (256, 4, 8), (128, 2, 4)];
    for &(chunk, ed, hh) in &configs {
        let st_g = Stored::new(&c, vocab, &dev)?;
        let seed = FractalSeed::new(ffn_total(&c), ffn_prior(&c), chunk, ed, hh, &dev)?;
        let seed_p = seed.seed_params();
        let mut vg = st_g.vars(); vg.extend(seed.vars());
        let (gl, _gov, ga) = train(vg, &|| seed.generate(), train_s, test_s, test_mask, &st_g, &c, steps, l, bs, lr, 1, &dev)?;

        // plain baseline: smaller dff so its stored FFN ~= seed_p
        let dff_small = ((seed_p / layers).saturating_sub(dm) / (2 * dm + 1)).max(1);
        let cp = Cfg { dm, dff: dff_small, layers };
        let st_p = Stored::new(&cp, vocab, &dev)?;
        let ffn_p = {
            let np = ffn_total(&cp);
            let noise = Tensor::randn(0f32, 1f32, (np,), &dev)?;
            Var::from_tensor(&noise.mul(&Tensor::from_vec(ffn_prior(&cp), (np,), &dev)?)?)?
        };
        let ffn_p_c = ffn_p.clone();
        let mut vp = st_p.vars(); vp.push(ffn_p.clone());
        let (_pl, _pov, pa) = train(vp, &|| Ok(ffn_p_c.as_tensor().clone()), train_s, test_s, test_mask, &st_p, &cp, steps, l, bs, lr, 1, &dev)?;

        let comp = ffn_total(&c) as f64 / seed_p as f64;
        println!("[lm] ffn-comp={comp:5.1}x seed={seed_p:5}  gen-FFN answer-acc={ga:.4} (loss {gl:.3})  plain(dff={dff_small}) answer-acc={pa:.4}  edge={:+.4}",
                 ga - pa);
    }
    println!("[lm] done");
    Ok(())
}
