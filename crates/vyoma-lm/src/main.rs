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

fn main() -> Result<()> {
    let steps: usize = std::env::var("STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(2000);
    let dm: usize = std::env::var("DM").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
    let dff: usize = std::env::var("DFF").ok().and_then(|s| s.parse().ok()).unwrap_or(512);
    let l: usize = std::env::var("SEQ").ok().and_then(|s| s.parse().ok()).unwrap_or(24);
    let layers: usize = std::env::var("LAYERS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let (bs, lr) = (64usize, 2e-3);
    // CPU by default: the SSM's sequential scan is many tiny ops, so Metal's
    // per-kernel launch overhead makes it ~2× SLOWER here (measured). DEVICE=metal
    // to opt in anyway (only worth it for much wider/FFN-heavier configs).
    let dev = if std::env::var("DEVICE").as_deref() == Ok("metal") {
        Device::new_metal(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    };
    println!("[lm] device={}", if dev.is_metal() { "metal(GPU)" } else { "cpu" });
    let dataset = std::env::var("DATASET").unwrap_or_else(|_| "arith".into());
    let mode = std::env::var("MODE").unwrap_or_else(|_| "sweep".into());

    // --- assemble corpus (+ vocab, + optional answer mask) ---
    let tok = std::env::var("TOKENIZER").unwrap_or_else(|_| "char".into());
    let (corpus, vocab, amask): (Vec<u32>, usize, Vec<bool>) = if dataset == "text" || dataset == "distilled" {
        let file = if dataset == "distilled" { "distilled.txt" } else { "tinyshakespeare.txt" };
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
