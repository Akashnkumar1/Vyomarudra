//! vyoma-embed — our learned neural retriever + persistent Ontological Store,
//! exposed as a reusable library (Pillar 4). Other crates (e.g. `vyoma-lm`)
//! depend on this to retrieve from an on-disk store with OUR encoder — no teacher,
//! no external model, ever.
//!
//! The encoder: byte embedding → our diagonal-SSM backbone (Pillar 2, reused as the
//! retriever) → mean-pool → projection → L2-normalize. Trained contrastively
//! (in-batch InfoNCE) from (query, document) byte-string pairs.

use anyhow::Result;
use candle_core::{DType, Device, Tensor, Var, D};
use candle_nn::{optim::{AdamW, ParamsAdamW}, Optimizer};
use rand::rngs::StdRng;

pub const VOCAB: usize = 256; // byte-level: any input encodes, no UNK

// ---------------------------------------------------------------------------
// Vector utilities
// ---------------------------------------------------------------------------
pub fn l2(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v {
            *x /= norm;
        }
    }
}
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn var_randn(shape: (usize, usize), std: f64, dev: &Device) -> Result<Var> {
    Ok(Var::from_tensor(&Tensor::randn(0f32, std as f32, shape, dev)?)?)
}
fn var_randn1(n: usize, std: f64, dev: &Device) -> Result<Var> {
    Ok(Var::from_tensor(&Tensor::randn(0f32, std as f32, (n,), dev)?)?)
}

// ---------------------------------------------------------------------------
// One diagonal-SSM block + the encoder
// ---------------------------------------------------------------------------
pub struct Layer {
    w_in: Var,   // (d, d)
    a_raw: Var,  // (d,)  A = sigmoid(a_raw) in (0,1) => Lyapunov-stable
    b_coef: Var, // (d,)
    c_coef: Var, // (d,)
}
impl Layer {
    fn new(d: usize, dev: &Device) -> Result<Self> {
        Ok(Self {
            w_in: var_randn((d, d), (1.0 / d as f64).sqrt(), dev)?,
            a_raw: var_randn1(d, 1.0, dev)?,
            b_coef: var_randn1(d, 0.5, dev)?,
            c_coef: var_randn1(d, 0.5, dev)?,
        })
    }
    fn vars(&self) -> Vec<Var> {
        vec![self.w_in.clone(), self.a_raw.clone(), self.b_coef.clone(), self.c_coef.clone()]
    }
    /// Input projection, BATCHED over all timesteps (one matmul instead of T).
    fn project(&self, x: &Tensor) -> Result<Tensor> {
        let (bsz, t_len, d) = x.dims3()?;
        let u = x.reshape((bsz * t_len, d))?.matmul(self.w_in.as_tensor())?.gelu()?;
        Ok(u.reshape((bsz, t_len, d))?)
    }
    /// General (multi-layer) path: x:(B,T,d) -> readout sequence + residual.
    fn scan(&self, x: &Tensor) -> Result<Tensor> {
        let (bsz, t_len, d) = x.dims3()?;
        let u = self.project(x)?;
        let a = candle_nn::ops::sigmoid(self.a_raw.as_tensor())?;
        let mut h = Tensor::zeros((bsz, d), DType::F32, x.device())?;
        let mut ys = Vec::with_capacity(t_len);
        for t in 0..t_len {
            let ut = u.narrow(1, t, 1)?.reshape((bsz, d))?;
            h = h.broadcast_mul(&a)?.add(&ut.broadcast_mul(self.b_coef.as_tensor())?)?;
            ys.push(h.broadcast_mul(self.c_coef.as_tensor())?.reshape((bsz, 1, d))?);
        }
        Ok((Tensor::cat(&ys, 1)? + x)?)
    }
    /// Fast fused mean-pool: mean_t(c⊙h_t) = c ⊙ mean_t(h_t). No cat, no residual.
    fn scan_mean(&self, x: &Tensor) -> Result<Tensor> {
        let (bsz, t_len, d) = x.dims3()?;
        let ub = self.project(x)?.broadcast_mul(self.b_coef.as_tensor())?;
        let a = candle_nn::ops::sigmoid(self.a_raw.as_tensor())?;
        let mut h = Tensor::zeros((bsz, d), DType::F32, x.device())?;
        let mut acc = Tensor::zeros((bsz, d), DType::F32, x.device())?;
        for t in 0..t_len {
            let ub_t = ub.narrow(1, t, 1)?.reshape((bsz, d))?;
            h = h.broadcast_mul(&a)?.add(&ub_t)?;
            acc = (acc + &h)?;
        }
        let mean_h = (acc / t_len as f64)?;
        Ok(mean_h.broadcast_mul(self.c_coef.as_tensor())?)
    }
}

/// Symmetric encoder: byte-embed → N SSM blocks → mean-pool → project → L2-norm.
pub struct Encoder {
    emb: Var,
    layers: Vec<Layer>,
    w_proj: Var,
    b_proj: Var,
    d: usize,
    pub dk: usize,
}
impl Encoder {
    pub fn new(d: usize, dk: usize, n_layers: usize, dev: &Device) -> Result<Self> {
        let layers = (0..n_layers).map(|_| Layer::new(d, dev)).collect::<Result<Vec<_>>>()?;
        Ok(Self {
            emb: var_randn((VOCAB, d), (1.0 / d as f64).sqrt(), dev)?,
            layers,
            w_proj: var_randn((d, dk), (1.0 / d as f64).sqrt(), dev)?,
            b_proj: var_randn1(dk, 1e-6, dev)?,
            d,
            dk,
        })
    }
    pub fn vars(&self) -> Vec<Var> {
        let mut v = vec![self.emb.clone(), self.w_proj.clone(), self.b_proj.clone()];
        for l in &self.layers {
            v.extend(l.vars());
        }
        v
    }
    pub fn n_params(&self) -> usize {
        self.vars().iter().map(|v| v.elem_count()).sum()
    }

    /// Persist the trained retriever (safetensors). Without this the encoder
    /// evaporates on exit and a VYST store becomes unusable — its int8 keys are
    /// only meaningful to the exact encoder that produced them. Store + encoder
    /// are a matched pair; save them together.
    pub fn save(&self, path: &str) -> Result<()> {
        let mut m: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();
        m.insert("emb".into(), self.emb.as_tensor().clone());
        m.insert("w_proj".into(), self.w_proj.as_tensor().clone());
        m.insert("b_proj".into(), self.b_proj.as_tensor().clone());
        for (i, l) in self.layers.iter().enumerate() {
            m.insert(format!("l.{i}.w_in"), l.w_in.as_tensor().clone());
            m.insert(format!("l.{i}.a_raw"), l.a_raw.as_tensor().clone());
            m.insert(format!("l.{i}.b_coef"), l.b_coef.as_tensor().clone());
            m.insert(format!("l.{i}.c_coef"), l.c_coef.as_tensor().clone());
        }
        candle_core::safetensors::save(&m, path)?;
        Ok(())
    }

    /// Load a saved retriever. Geometry (d, dk, n_layers) comes from tensor
    /// shapes, so weights and config can never drift apart.
    pub fn load(path: &str, dev: &Device) -> Result<Self> {
        let t = candle_core::safetensors::load(path, dev)?;
        let g = |k: &str| -> Result<Tensor> {
            t.get(k).cloned().ok_or_else(|| anyhow::anyhow!("retriever ckpt missing `{k}`"))
        };
        let v = |k: &str| -> Result<Var> { Ok(Var::from_tensor(&g(k)?)?) };
        let emb = g("emb")?;
        let (_vocab, d) = emb.dims2()?;
        let w_proj = g("w_proj")?;
        let (_d2, dk) = w_proj.dims2()?;
        let n = (0..).take_while(|i| t.contains_key(&format!("l.{i}.w_in"))).count().max(1);
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            layers.push(Layer {
                w_in: v(&format!("l.{i}.w_in"))?, a_raw: v(&format!("l.{i}.a_raw"))?,
                b_coef: v(&format!("l.{i}.b_coef"))?, c_coef: v(&format!("l.{i}.c_coef"))?,
            });
        }
        Ok(Self { emb: Var::from_tensor(&emb)?, layers, w_proj: Var::from_tensor(&w_proj)?, b_proj: v("b_proj")?, d, dk })
    }
    /// tokens: (B, T) u32 -> L2-normalized embeddings (B, dk)
    pub fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let (bsz, t_len) = tokens.dims2()?;
        let flat = tokens.reshape((bsz * t_len,))?;
        let x = self.emb.as_tensor().index_select(&flat, 0)?.reshape((bsz, t_len, self.d))?;
        let mean = if self.layers.len() == 1 {
            self.layers[0].scan_mean(&x)?
        } else {
            let mut h = x;
            for l in &self.layers[..self.layers.len() - 1] {
                h = l.scan(&h)?;
            }
            self.layers.last().unwrap().scan_mean(&h)?
        };
        let z = mean.matmul(self.w_proj.as_tensor())?.broadcast_add(self.b_proj.as_tensor())?;
        let norm = z.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        Ok(z.broadcast_div(&(norm + 1e-6)?)?)
    }
}

/// Pack ragged byte rows into a (B, T) u32 tensor, right-padded with 0 to the
/// longest row in the batch (so callers need not pre-equalize lengths).
pub fn tokens_from(rows: &[Vec<u8>], dev: &Device) -> Result<Tensor> {
    let t = rows.iter().map(|r| r.len()).max().unwrap_or(1).max(1);
    let n = rows.len();
    let mut flat = vec![0u32; n * t];
    for (i, r) in rows.iter().enumerate() {
        for (j, &b) in r.iter().enumerate() {
            flat[i * t + j] = b as u32;
        }
    }
    Ok(Tensor::from_vec(flat, (n, t), dev)?)
}

/// Embed many byte rows in batches → Vec of L2-normalized embeddings.
pub fn embed_all(enc: &Encoder, rows: &[Vec<u8>], dev: &Device) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(rows.len());
    let (bs, mut start) = (256usize, 0usize);
    while start < rows.len() {
        let len = bs.min(rows.len() - start);
        let z = enc.forward(&tokens_from(&rows[start..start + len], dev)?)?.to_vec2::<f32>()?;
        out.extend(z);
        start += len;
    }
    Ok(out)
}

/// Train the encoder contrastively from a (query, document) pair sampler.
/// `sample(rng)` returns one positive (query_bytes, doc_bytes) pair; the batch's
/// other documents are the in-batch negatives (InfoNCE).
#[allow(clippy::too_many_arguments)]
pub fn train_encoder<S>(
    d: usize, dk: usize, n_layers: usize, steps: usize, bsz: usize, lr: f64, temp: f64,
    dev: &Device, seed: u64, sample: S,
) -> Result<(Encoder, f32)>
where
    S: FnMut(&mut StdRng) -> (Vec<u8>, Vec<u8>),
{
    train_encoder_neg::<S, fn(&mut StdRng) -> Vec<u8>>(d, dk, n_layers, steps, bsz, lr, temp, dev, seed, sample, None, 0)
}

/// As `train_encoder`, plus **out-of-domain negatives**.
///
/// Plain in-batch InfoNCE only ever contrasts a query against other documents from
/// the SAME corpus, so the encoder learns "*which* passage?" and never "is this my
/// domain at all?" — measured consequence: an out-of-domain prompt scored *higher*
/// than a correct in-domain match, and no gate threshold could separate them (see
/// docs/PROGRESS.md, MODE=rag). Fix: append `n_neg` documents drawn from a
/// DIFFERENT corpus to every batch. They are negatives for every query, so the
/// model must push foreign text away from the whole in-domain query manifold —
/// which is exactly the signal a grounding gate needs.
///
/// Logits become (B, B+n_neg); targets stay the diagonal. The passage→query
/// direction uses only the in-domain block (foreign docs have no matching query).
#[allow(clippy::too_many_arguments)]
pub fn train_encoder_neg<S, N>(
    d: usize, dk: usize, n_layers: usize, steps: usize, bsz: usize, lr: f64, temp: f64,
    dev: &Device, seed: u64, mut sample: S, mut neg_sample: Option<N>, n_neg: usize,
) -> Result<(Encoder, f32)>
where
    S: FnMut(&mut StdRng) -> (Vec<u8>, Vec<u8>),
    N: FnMut(&mut StdRng) -> Vec<u8>,
{
    use rand::SeedableRng;
    let enc = Encoder::new(d, dk, n_layers, dev)?;
    let mut opt = AdamW::new(enc.vars(), ParamsAdamW { lr, ..Default::default() })?;
    let mut rng = StdRng::seed_from_u64(seed);
    let targets = Tensor::from_vec((0..bsz as u32).collect::<Vec<_>>(), (bsz,), dev)?;
    let mut last = 0f32;
    for _ in 0..steps {
        let mut q_rows = Vec::with_capacity(bsz);
        let mut p_rows = Vec::with_capacity(bsz);
        for _ in 0..bsz {
            let (q, p) = sample(&mut rng);
            q_rows.push(q);
            p_rows.push(p);
        }
        let q = enc.forward(&tokens_from(&q_rows, dev)?)?;
        let p = enc.forward(&tokens_from(&p_rows, dev)?)?;
        // query→doc: contrast against in-domain docs AND foreign negatives
        let logits = match (&mut neg_sample, n_neg) {
            (Some(ns), k) if k > 0 => {
                let neg_rows: Vec<Vec<u8>> = (0..k).map(|_| ns(&mut rng)).collect();
                let neg = enc.forward(&tokens_from(&neg_rows, dev)?)?;      // (K, dk)
                let all = Tensor::cat(&[&p, &neg], 0)?;                      // (B+K, dk)
                (q.matmul(&all.t()?)? / temp)?                               // (B, B+K)
            }
            _ => (q.matmul(&p.t()?)? / temp)?,
        };
        let loss_q = candle_nn::loss::cross_entropy(&logits, &targets)?;
        // doc→query: in-domain block only (foreign docs have no matching query)
        let logits_p = (p.matmul(&q.t()?)? / temp)?;
        let loss_p = candle_nn::loss::cross_entropy(&logits_p, &targets)?;
        let loss = ((loss_q + loss_p)? / 2.0)?;
        opt.backward_step(&loss)?;
        last = loss.to_scalar::<f32>()?;
    }
    Ok((enc, last))
}

// ---------------------------------------------------------------------------
// The persistent Ontological Store (VYST v2, variable-length records).
// Layout (little-endian): magic "VYST" | u32 ver=2 | u32 N | u32 dk |
//   N × (f32 scale, dk×i8 codes) | N × u32 len | concatenated record bytes.
// int8 keys on disk; dequantized + L2-normalized on load for cosine retrieval.
// ---------------------------------------------------------------------------
pub fn quant_i8(v: &[f32]) -> (Vec<i8>, f32) {
    let maxabs = v.iter().fold(0f32, |m, &x| m.max(x.abs()));
    let scale = if maxabs == 0.0 { 1.0 } else { maxabs / 127.0 };
    let codes = v.iter().map(|&x| (x / scale).round().clamp(-127.0, 127.0) as i8).collect();
    (codes, scale)
}

pub fn write_store(path: &str, records: &[Vec<u8>], embs: &[Vec<f32>]) -> Result<u64> {
    anyhow::ensure!(records.len() == embs.len(), "records/embs length mismatch");
    let (n, dk) = (records.len(), embs[0].len());
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"VYST");
    buf.extend_from_slice(&2u32.to_le_bytes());
    buf.extend_from_slice(&(n as u32).to_le_bytes());
    buf.extend_from_slice(&(dk as u32).to_le_bytes());
    for e in embs {
        let (codes, scale) = quant_i8(e);
        buf.extend_from_slice(&scale.to_le_bytes());
        buf.extend(codes.iter().map(|&c| c as u8));
    }
    for r in records {
        buf.extend_from_slice(&(r.len() as u32).to_le_bytes());
    }
    for r in records {
        buf.extend_from_slice(r);
    }
    std::fs::write(path, &buf)?;
    Ok(buf.len() as u64)
}

/// Load the store: (dequantized + L2-normalized embeddings, record bytes).
pub fn load_store(path: &str) -> Result<(Vec<Vec<f32>>, Vec<Vec<u8>>)> {
    let b = std::fs::read(path)?;
    anyhow::ensure!(b.len() >= 16 && &b[0..4] == b"VYST", "bad store magic");
    let rd = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as usize;
    let (n, dk) = (rd(8), rd(12));
    let mut o = 16usize;
    let mut embs = Vec::with_capacity(n);
    for _ in 0..n {
        let scale = f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        o += 4;
        let mut v: Vec<f32> = (0..dk).map(|k| (b[o + k] as i8) as f32 * scale).collect();
        o += dk;
        l2(&mut v);
        embs.push(v);
    }
    let mut lens = Vec::with_capacity(n);
    for _ in 0..n {
        lens.push(rd(o));
        o += 4;
    }
    let mut records = Vec::with_capacity(n);
    for len in lens {
        records.push(b[o..o + len].to_vec());
        o += len;
    }
    Ok((embs, records))
}

/// Retrieve the index of the nearest stored embedding to `query` (cosine).
pub fn nearest(query: &[f32], store: &[Vec<f32>]) -> usize {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (j, e) in store.iter().enumerate() {
        let s = dot(query, e);
        if s > best.1 {
            best = (j, s);
        }
    }
    best.0
}
