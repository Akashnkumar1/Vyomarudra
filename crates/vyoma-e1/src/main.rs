//! E1 — Generative Weights (Project Vyomarudra / Vyomarudra), Phase-0 falsification.
//!
//! Load-bearing bet (Risk #1, the project-killer): store the RULES that generate
//! a network's weights, not the weights. The honest test is not "does the seed
//! have fewer params" — it is "does generating a target's weights from a small
//! seed beat just training a plain network of that same size?"
//!
//! This build pushes toward Gate G0 (≥100× compression). Two seed designs:
//!   HyperSeed   — chunked hypernetwork with STORED per-chunk embeddings. Its
//!                 embedding table costs O(n_chunks), which caps compression.
//!   FractalSeed — the Vyomarudra addition: chunk addresses are GENERATED from a fixed
//!                 sinusoidal index encoding through a tiny address net, so seed
//!                 storage is O(1) in target size. This is what breaks the wall.
//!
//! Backend: candle (pure-Rust tensors + autograd), CPU. Real MNIST if present in
//! data_cache/, else an offline synthetic teacher task.

use anyhow::Result;
use candle_core::{DType, Device, Tensor, Var, D};
use candle_nn::{optim::{AdamW, ParamsAdamW}, Optimizer};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ---------------------------------------------------------------------------
// Target network geometry
// ---------------------------------------------------------------------------
#[derive(Clone, Copy)]
struct TargetSpec {
    in_dim: usize,
    hidden: usize,
    out_dim: usize,
}

impl TargetSpec {
    fn n_params(&self) -> usize {
        self.hidden * self.in_dim + self.hidden
            + self.out_dim * self.hidden + self.out_dim
    }
}

/// Run the target MLP from a flat weight vector (length = spec.n_params).
fn target_forward(x: &Tensor, flat: &Tensor, spec: &TargetSpec) -> Result<Tensor> {
    let (i, h, o) = (spec.in_dim, spec.hidden, spec.out_dim);
    let mut off = 0usize;
    let w1 = flat.narrow(0, off, h * i)?.reshape((h, i))?; off += h * i;
    let b1 = flat.narrow(0, off, h)?; off += h;
    let w2 = flat.narrow(0, off, o * h)?.reshape((o, h))?; off += o * h;
    let b2 = flat.narrow(0, off, o)?;
    let hidden = x.matmul(&w1.t()?)?.broadcast_add(&b1)?.relu()?;
    Ok(hidden.matmul(&w2.t()?)?.broadcast_add(&b2)?)
}

// ---------------------------------------------------------------------------
// Var helpers
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

/// Calibrate a scalar global scale so that a freshly-generated weight vector has
/// std ≈ target_std (not near-zero). Returns the multiplier to apply.
fn calibration(flat: &Tensor, target_std: f32) -> Result<f64> {
    let std = flat.broadcast_sub(&flat.mean_all()?)?.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
    Ok((target_std / (std + 1e-8)) as f64)
}

// ---------------------------------------------------------------------------
// HyperSeed: chunked hypernetwork with STORED per-chunk embeddings.
// Seed storage = params(g) + n_chunks*embed_dim + n_chunks(scale). The embedding
// table grows with the target, so this design cannot reach very high compression.
// ---------------------------------------------------------------------------
struct HyperSeed {
    embed: Var, g_w1: Var, g_b1: Var, g_w2: Var, g_b2: Var, out_scale: Var,
    n_params_target: usize,
}

impl HyperSeed {
    fn new(spec: &TargetSpec, chunk: usize, embed_dim: usize, hyper_hidden: usize, dev: &Device) -> Result<Self> {
        let p = spec.n_params();
        let n_chunks = (p + chunk - 1) / chunk;
        let s = Self {
            embed: var_randn((n_chunks, embed_dim), 0.1, dev)?,
            g_w1: var_randn((embed_dim, hyper_hidden), (1.0 / embed_dim as f64).sqrt(), dev)?,
            g_b1: var_randn1(hyper_hidden, 1e-6, dev)?,
            g_w2: var_randn((hyper_hidden, chunk), (1.0 / hyper_hidden as f64).sqrt(), dev)?,
            g_b2: var_randn1(chunk, 1e-6, dev)?,
            out_scale: var_full1(n_chunks, 1.0, dev)?,
            n_params_target: p,
        };
        let cal = calibration(&s.generate()?, 0.05)?;
        s.out_scale.set(&(Tensor::ones((n_chunks,), DType::F32, dev)? * cal)?)?;
        Ok(s)
    }
    fn vars(&self) -> Vec<Var> {
        vec![self.embed.clone(), self.g_w1.clone(), self.g_b1.clone(),
             self.g_w2.clone(), self.g_b2.clone(), self.out_scale.clone()]
    }
    fn seed_params(&self) -> usize {
        self.vars().iter().map(|v| v.elem_count()).sum()
    }
    fn generate(&self) -> Result<Tensor> {
        let h = self.embed.as_tensor().matmul(self.g_w1.as_tensor())?
            .broadcast_add(self.g_b1.as_tensor())?.gelu()?;
        let raw = h.matmul(self.g_w2.as_tensor())?.broadcast_add(self.g_b2.as_tensor())?;
        let scaled = raw.broadcast_mul(&self.out_scale.as_tensor().reshape((raw.dim(0)?, 1))?)?;
        Ok(scaled.flatten_all()?.narrow(0, 0, self.n_params_target)?)
    }
}

// ---------------------------------------------------------------------------
// FractalSeed: chunk addresses are GENERATED from a fixed sinusoidal encoding of
// the chunk index, so there is NO stored embedding table. Seed storage is O(1)
// in the number of chunks => compression scales with target size.
//   index -> sinusoidal PE (fixed) -> address net -> embedding
//   embedding -> generator net    -> chunk weights
//   embedding -> scale head       -> per-chunk gain
// ---------------------------------------------------------------------------
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
    pe: Tensor, // fixed, NOT a parameter
    a_w1: Var, a_b1: Var, a_w2: Var, a_b2: Var,   // address net: PE -> embedding
    g_w1: Var, g_b1: Var, g_w2: Var, g_b2: Var,   // generator: embedding -> chunk
    s_w: Var, s_b: Var,                           // scale head: embedding -> gain
    gscale: Var,                                  // global calibrated scale
    n_params_target: usize,
}

impl FractalSeed {
    fn new(spec: &TargetSpec, chunk: usize, embed_dim: usize, hyper_hidden: usize, dev: &Device) -> Result<Self> {
        let p = spec.n_params();
        let n_chunks = (p + chunk - 1) / chunk;
        let s = Self {
            pe: positional_encoding(n_chunks, PE_DIM, dev)?,
            a_w1: var_randn((PE_DIM, ADDR_HIDDEN), (1.0 / PE_DIM as f64).sqrt(), dev)?,
            a_b1: var_randn1(ADDR_HIDDEN, 1e-6, dev)?,
            a_w2: var_randn((ADDR_HIDDEN, embed_dim), (1.0 / ADDR_HIDDEN as f64).sqrt(), dev)?,
            a_b2: var_randn1(embed_dim, 1e-6, dev)?,
            g_w1: var_randn((embed_dim, hyper_hidden), (1.0 / embed_dim as f64).sqrt(), dev)?,
            g_b1: var_randn1(hyper_hidden, 1e-6, dev)?,
            g_w2: var_randn((hyper_hidden, chunk), (1.0 / hyper_hidden as f64).sqrt(), dev)?,
            g_b2: var_randn1(chunk, 1e-6, dev)?,
            s_w: var_randn((embed_dim, 1), 0.01, dev)?,
            s_b: var_full1(1, 1.0, dev)?, // start with unit gain
            gscale: var_full1(1, 1.0, dev)?,
            n_params_target: p,
        };
        let cal = calibration(&s.generate()?, 0.05)?;
        s.gscale.set(&(Tensor::ones((1,), DType::F32, dev)? * cal)?)?;
        Ok(s)
    }
    fn vars(&self) -> Vec<Var> {
        vec![self.a_w1.clone(), self.a_b1.clone(), self.a_w2.clone(), self.a_b2.clone(),
             self.g_w1.clone(), self.g_b1.clone(), self.g_w2.clone(), self.g_b2.clone(),
             self.s_w.clone(), self.s_b.clone(), self.gscale.clone()]
    }
    fn seed_params(&self) -> usize {
        self.vars().iter().map(|v| v.elem_count()).sum() // pe excluded — it is fixed
    }
    fn generate(&self) -> Result<Tensor> {
        let emb = self.pe.matmul(self.a_w1.as_tensor())?.broadcast_add(self.a_b1.as_tensor())?
            .gelu()?.matmul(self.a_w2.as_tensor())?.broadcast_add(self.a_b2.as_tensor())?; // (n, embed)
        let h = emb.matmul(self.g_w1.as_tensor())?.broadcast_add(self.g_b1.as_tensor())?.gelu()?;
        let raw = h.matmul(self.g_w2.as_tensor())?.broadcast_add(self.g_b2.as_tensor())?; // (n, chunk)
        let scale = emb.matmul(self.s_w.as_tensor())?.broadcast_add(self.s_b.as_tensor())?; // (n,1)
        let chunks = raw.broadcast_mul(&scale)?.broadcast_mul(self.gscale.as_tensor())?;
        Ok(chunks.flatten_all()?.narrow(0, 0, self.n_params_target)?)
    }
}

// ---------------------------------------------------------------------------
// PlainMlp: the fair baseline — a directly-trained MLP sized to ~S params.
// ---------------------------------------------------------------------------
struct PlainMlp { w1: Var, b1: Var, w2: Var, b2: Var, hidden: usize }

impl PlainMlp {
    fn new(target_params: usize, spec: &TargetSpec, dev: &Device) -> Result<Self> {
        let hidden = (((target_params as i64 - spec.out_dim as i64)
            / (spec.in_dim as i64 + 1 + spec.out_dim as i64)).max(1)) as usize;
        Ok(Self {
            w1: var_randn((hidden, spec.in_dim), (1.0 / spec.in_dim as f64).sqrt(), dev)?,
            b1: var_randn1(hidden, 1e-6, dev)?,
            w2: var_randn((spec.out_dim, hidden), (1.0 / hidden as f64).sqrt(), dev)?,
            b2: var_randn1(spec.out_dim, 1e-6, dev)?,
            hidden,
        })
    }
    fn vars(&self) -> Vec<Var> { vec![self.w1.clone(), self.b1.clone(), self.w2.clone(), self.b2.clone()] }
    fn n_params(&self, spec: &TargetSpec) -> usize {
        self.hidden * spec.in_dim + self.hidden + spec.out_dim * self.hidden + spec.out_dim
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = x.matmul(&self.w1.as_tensor().t()?)?.broadcast_add(self.b1.as_tensor())?.relu()?;
        Ok(h.matmul(&self.w2.as_tensor().t()?)?.broadcast_add(self.b2.as_tensor())?)
    }
}

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------
struct Dataset { xtr: Tensor, ytr: Tensor, xte: Tensor, yte: Tensor, name: &'static str }

fn load_idx(dir: &str, name: &'static str) -> Result<Option<Dataset>> {
    let need = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte",
                "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"];
    if !need.iter().all(|f| std::path::Path::new(&format!("{dir}/{f}")).exists()) {
        return Ok(None);
    }
    let rd = |name: &str| -> Result<Vec<u8>> { Ok(std::fs::read(format!("{dir}/{name}"))?) };
    let imgs = |b: &[u8]| -> (usize, Vec<f32>) {
        let n = u32::from_be_bytes([b[4], b[5], b[6], b[7]]) as usize;
        let px = (u32::from_be_bytes([b[8], b[9], b[10], b[11]])
                * u32::from_be_bytes([b[12], b[13], b[14], b[15]])) as usize;
        (n, (0..n * px).map(|i| b[16 + i] as f32 / 255.0).collect())
    };
    let labs = |b: &[u8]| -> Vec<u32> {
        let n = u32::from_be_bytes([b[4], b[5], b[6], b[7]]) as usize;
        (0..n).map(|i| b[8 + i] as u32).collect()
    };
    let dev = Device::Cpu;
    let (ntr, xtr) = imgs(&rd("train-images-idx3-ubyte")?);
    let (nte, xte) = imgs(&rd("t10k-images-idx3-ubyte")?);
    Ok(Some(Dataset {
        xtr: Tensor::from_vec(xtr, (ntr, 784), &dev)?,
        ytr: Tensor::from_vec(labs(&rd("train-labels-idx1-ubyte")?), (ntr,), &dev)?,
        xte: Tensor::from_vec(xte, (nte, 784), &dev)?,
        yte: Tensor::from_vec(labs(&rd("t10k-labels-idx1-ubyte")?), (nte,), &dev)?,
        name,
    }))
}

fn randn_vec(rng: &mut StdRng, n: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(n);
    while v.len() < n {
        let u1: f32 = rng.gen::<f32>().max(1e-7);
        let u2: f32 = rng.gen::<f32>();
        let r = (-2.0 * u1.ln()).sqrt();
        v.push(r * (2.0 * std::f32::consts::PI * u2).cos());
        if v.len() < n { v.push(r * (2.0 * std::f32::consts::PI * u2).sin()); }
    }
    v
}

fn make_synthetic(spec: &TargetSpec, n_train: usize, n_test: usize, dev: &Device) -> Result<Dataset> {
    let mut rng = StdRng::seed_from_u64(1234);
    let (i, th, o) = (spec.in_dim, 64usize, spec.out_dim);
    let s1 = 1.0f32 / (i as f32).sqrt();
    let s2 = 1.0f32 / (th as f32).sqrt();
    let tw1: Vec<f32> = randn_vec(&mut rng, th * i).into_iter().map(|v| v * s1).collect();
    let tw2: Vec<f32> = randn_vec(&mut rng, o * th).into_iter().map(|v| v * s2).collect();
    let gen = |rng: &mut StdRng, n: usize| -> (Vec<f32>, Vec<u32>) {
        let x = randn_vec(rng, n * i);
        let mut y = Vec::with_capacity(n);
        for r in 0..n {
            let mut hbuf = vec![0f32; th];
            for hh in 0..th {
                let mut acc = 0f32;
                for c in 0..i { acc += tw1[hh * i + c] * x[r * i + c]; }
                hbuf[hh] = acc.tanh();
            }
            let (mut best, mut bv) = (0u32, f32::NEG_INFINITY);
            for oo in 0..o {
                let mut acc = 0f32;
                for hh in 0..th { acc += tw2[oo * th + hh] * hbuf[hh]; }
                if acc > bv { bv = acc; best = oo as u32; }
            }
            y.push(best);
        }
        (x, y)
    };
    let (xtr, ytr) = gen(&mut rng, n_train);
    let (xte, yte) = gen(&mut rng, n_test);
    Ok(Dataset {
        xtr: Tensor::from_vec(xtr, (n_train, i), dev)?,
        ytr: Tensor::from_vec(ytr, (n_train,), dev)?,
        xte: Tensor::from_vec(xte, (n_test, i), dev)?,
        yte: Tensor::from_vec(yte, (n_test,), dev)?,
        name: "synthetic-teacher",
    })
}

// ---------------------------------------------------------------------------
// Train / eval
// ---------------------------------------------------------------------------
fn accuracy(logits: &Tensor, y: &Tensor) -> Result<f64> {
    let pred = logits.argmax(D::Minus1)?.to_dtype(DType::U32)?;
    let correct = pred.eq(y)?.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
    Ok(correct as f64 / y.elem_count() as f64)
}

fn train<F>(vars: Vec<Var>, data: &Dataset, epochs: usize, lr: f64, bs: usize,
            seed: u64, dev: &Device, forward: F) -> Result<f64>
where F: Fn(&Tensor) -> Result<Tensor> {
    let mut opt = AdamW::new(vars, ParamsAdamW { lr, ..Default::default() })?;
    let n = data.xtr.dim(0)?;
    let mut rng = StdRng::seed_from_u64(seed);
    for _ in 0..epochs {
        let mut idx: Vec<u32> = (0..n as u32).collect();
        for k in (1..idx.len()).rev() { let j = rng.gen_range(0..=k); idx.swap(k, j); }
        let idx_t = Tensor::from_vec(idx, (n,), dev)?;
        let xs = data.xtr.index_select(&idx_t, 0)?;
        let ys = data.ytr.index_select(&idx_t, 0)?;
        let mut start = 0;
        while start < n {
            let len = bs.min(n - start);
            let logits = forward(&xs.narrow(0, start, len)?)?;
            let loss = candle_nn::loss::cross_entropy(&logits, &ys.narrow(0, start, len)?)?;
            opt.backward_step(&loss)?;
            start += len;
        }
    }
    accuracy(&forward(&data.xte)?, &data.yte)
}

/// Symmetric per-tensor quantization to `bits`, returned dequantized (fake-quant).
/// This is what the SEED would cost in RAM at that precision.
fn quantize_vec(data: &[f32], bits: u32) -> Vec<f32> {
    let qmax = ((1i64 << (bits - 1)) - 1) as f32; // 127 @ 8-bit, 7 @ 4-bit
    let max_abs = data.iter().fold(0f32, |m, &x| m.max(x.abs())).max(1e-8);
    let scale = max_abs / qmax;
    data.iter().map(|&x| (x / scale).round().clamp(-qmax, qmax) * scale).collect()
}

fn mean_std(v: &[f64]) -> (f64, f64) {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64;
    (m, var.sqrt())
}

struct Row {
    chunk: usize, embed: usize, hh: usize,
    seed_fractal: usize, comp_fractal: f64, acc_fractal: f64, std_fractal: f64,
    plain_params: usize, acc_plain: f64, std_plain: f64,
    seed_chunked: usize, comp_chunked: f64,
}

fn main() -> Result<()> {
    let epochs: usize = std::env::var("EPOCHS").ok().and_then(|s| s.parse().ok()).unwrap_or(15);
    let seeds: usize = std::env::var("SEEDS").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    let hidden: usize = std::env::var("HIDDEN").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let (lr, bs) = (1e-3, 128usize);
    let dev = Device::Cpu;

    let spec = TargetSpec { in_dim: 784, hidden, out_dim: 10 };
    let p = spec.n_params();
    println!("[e1] device=cpu  target=784->{hidden}->10  P={p}  epochs={epochs} seeds={seeds}");

    let kind = std::env::var("DATASET").unwrap_or_else(|_| "mnist".into());
    let (sub, nm): (&str, &'static str) = match kind.as_str() {
        "fashion" => ("data_cache/fashion", "Fashion-MNIST"),
        _ => ("data_cache", "MNIST"),
    };
    let cache = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), sub);
    let data = match load_idx(&cache, nm)? { Some(d) => d, None => make_synthetic(&spec, 6000, 2000, &dev)? };
    println!("[e1] dataset={}  train={}  test={}", data.name, data.xtr.dim(0)?, data.xte.dim(0)?);

    // Upper bound: dense target trained directly.
    let mut full = Vec::new();
    for s in 0..seeds {
        let flat = var_randn1(p, 0.02, &dev)?;
        let fc = flat.clone();
        full.push(train(vec![flat], &data, epochs, lr, bs, s as u64, &dev,
            |xb| target_forward(xb, fc.as_tensor(), &spec))?);
    }
    let (full_m, full_s) = mean_std(&full);
    println!("[e1] UPPER BOUND dense ({p} params): {full_m:.4} ± {full_s:.4}");

    // QUANT: compose the two proven multipliers — generation × quantization.
    // Train a high-compression fractal seed, then evaluate the generated network
    // with the SEED quantized to fp32 / 8-bit / 4-bit (that is its real RAM cost).
    if std::env::var("QUANT").is_ok() {
        let (chunk, ed, hh) = (256usize, 2usize, 4usize);
        let frac = FractalSeed::new(&spec, chunk, ed, hh, &dev)?;
        let sp = frac.seed_params();
        let param_comp = p as f64 / sp as f64;
        let trained = train(frac.vars(), &data, epochs, lr, bs, 0, &dev,
            |xb| { let f = frac.generate()?; target_forward(xb, &f, &spec) })?;
        println!("[e1] QUANT — seed={sp} params ({param_comp:.0}× param-compression), trained acc {trained:.4}");
        let vars = frac.vars();
        // snapshot as raw data (+shape) so restores/quant use FRESH storage (Var::set
        // rejects a tensor sharing the variable's own storage).
        let orig: Vec<(Vec<f32>, Vec<usize>)> = vars.iter().map(|v| {
            let t = v.as_tensor();
            Ok((t.flatten_all()?.to_vec1::<f32>()?, t.dims().to_vec()))
        }).collect::<Result<_>>()?;
        for &bits in &[32u32, 8, 4] {
            for (i, v) in vars.iter().enumerate() {
                let (d, shape) = &orig[i];
                let vals = if bits == 32 { d.clone() } else { quantize_vec(d, bits) };
                v.set(&Tensor::from_vec(vals, shape.clone(), &dev)?)?;
            }
            let acc = accuracy(&target_forward(&data.xte, &frac.generate()?, &spec)?, &data.yte)?;
            let byte_comp = param_comp * (32.0 / bits as f64); // vs dense fp32
            let kb = sp as f64 * bits as f64 / 8.0 / 1024.0;
            println!("[e1]  seed@{bits:2}-bit: acc={acc:.4} ({:.1}% of dense)  effective compression {byte_comp:5.0}× vs dense-fp32  (seed {kb:.1} KB)",
                     acc / full_m * 100.0);
        }
        for (i, v) in vars.iter().enumerate() {
            let (d, shape) = &orig[i];
            v.set(&Tensor::from_vec(d.clone(), shape.clone(), &dev)?)?;
        }
        println!("[e1] done (quant). generation × quantization compose if 4-bit holds accuracy.");
        return Ok(());
    }

    // (chunk, embed, hyper_hidden) — spread to push fractal compression high.
    let configs = [
        (256usize, 16usize, 32usize),
        (512, 8, 16),
        (512, 4, 8),
        (256, 4, 8),
        (256, 2, 4),
        (128, 2, 4),
    ];

    let mut rows = Vec::new();
    for &(chunk, ed, hh) in &configs {
        let (mut fa, mut pa) = (Vec::new(), Vec::new());
        let (mut seed_f, mut plain_p, mut seed_c) = (0, 0, 0);
        for s in 0..seeds {
            let frac = FractalSeed::new(&spec, chunk, ed, hh, &dev)?;
            seed_f = frac.seed_params();
            fa.push(train(frac.vars(), &data, epochs, lr, bs, s as u64, &dev,
                |xb| { let flat = frac.generate()?; target_forward(xb, &flat, &spec) })?);

            let plain = PlainMlp::new(seed_f, &spec, &dev)?;
            plain_p = plain.n_params(&spec);
            pa.push(train(plain.vars(), &data, epochs, lr, bs, s as u64, &dev, |xb| plain.forward(xb))?);

            // measure the stored-embedding seed size at this config (no training)
            seed_c = HyperSeed::new(&spec, chunk, ed, hh, &dev)?.seed_params();
        }
        let (fm, fs) = mean_std(&fa);
        let (pm, ps) = mean_std(&pa);
        let comp_f = p as f64 / seed_f as f64;
        let comp_c = p as f64 / seed_c as f64;
        println!("[e1] fractal comp={comp_f:6.1}x seed={seed_f:5}  fractal={fm:.4}  plain(={plain_p})={pm:.4}  edge={:+.4}  | chunked-seed={seed_c} ({comp_c:.1}x)",
                 fm - pm);
        rows.push(Row { chunk, embed: ed, hh, seed_fractal: seed_f, comp_fractal: comp_f,
            acc_fractal: fm, std_fractal: fs, plain_params: plain_p, acc_plain: pm,
            std_plain: ps, seed_chunked: seed_c, comp_chunked: comp_c });
    }

    rows.sort_by(|a, b| a.comp_fractal.partial_cmp(&b.comp_fractal).unwrap());
    write_outputs(&spec, data.name, full_m, full_s, epochs, seeds, &rows)?;
    println!("[e1] wrote results/e1_results.md and results/e1_results.json");
    Ok(())
}

fn write_outputs(spec: &TargetSpec, dataset: &str, full_m: f64, full_s: f64, epochs: usize, seeds: usize, rows: &[Row]) -> Result<()> {
    let dir = format!("{}/results", env!("CARGO_MANIFEST_DIR"));
    std::fs::create_dir_all(&dir)?;
    let mut md = String::new();
    md.push_str("# E1 Results — Generative Weights, Fractal Seed (Rust / candle)\n\n");
    md.push_str(&format!("- Task: **{dataset}** | target 784->{}->10 = **{} params** | epochs: {epochs} | seeds: {seeds}\n", spec.hidden, spec.n_params()));
    md.push_str(&format!("- Dense upper bound: **{:.4} ± {:.4}**\n\n", full_m, full_s));
    md.push_str("| Fractal comp | Seed params | Fractal acc | Plain MLP (same budget) | Fractal − Plain | % of dense | Stored-embed seed would be |\n");
    md.push_str("|---|---|---|---|---|---|---|\n");
    for r in rows {
        let edge = r.acc_fractal - r.acc_plain;
        let mark = if edge > 0.005 { " ✅" } else if edge < -0.005 { " ❌" } else { " ➖" };
        md.push_str(&format!(
            "| {:.1}× | {} | {:.4} | {:.4} ({}) | {:+.4}{} | {:.1}% | {} ({:.1}×) |\n",
            r.comp_fractal, r.seed_fractal, r.acc_fractal, r.acc_plain, r.plain_params,
            edge, mark, r.acc_fractal / full_m * 100.0, r.seed_chunked, r.comp_chunked));
    }
    md.push_str("\n**Fractal seed** generates chunk addresses from a fixed sinusoidal index encoding, so its size is ~constant in target size — that is why it reaches high compression where the stored-embedding seed (last column) cannot.\n");
    std::fs::write(format!("{dir}/e1_results.md"), md)?;

    let mut js = String::from("{\n");
    js.push_str(&format!("  \"target_params\": {},\n  \"full_acc\": {:.4}, \"full_std\": {:.4},\n  \"epochs\": {epochs}, \"seeds\": {seeds},\n  \"configs\": [\n", spec.n_params(), full_m, full_s));
    for (k, r) in rows.iter().enumerate() {
        js.push_str(&format!(
            "    {{\"chunk\": {}, \"embed\": {}, \"hyper_hidden\": {}, \"comp_fractal\": {:.2}, \"seed_fractal\": {}, \"acc_fractal\": {:.4}, \"std_fractal\": {:.4}, \"plain_params\": {}, \"acc_plain\": {:.4}, \"std_plain\": {:.4}, \"edge\": {:.4}, \"seed_chunked\": {}, \"comp_chunked\": {:.2}}}{}\n",
            r.chunk, r.embed, r.hh, r.comp_fractal, r.seed_fractal, r.acc_fractal, r.std_fractal,
            r.plain_params, r.acc_plain, r.std_plain, r.acc_fractal - r.acc_plain, r.seed_chunked, r.comp_chunked,
            if k + 1 < rows.len() { "," } else { "" }));
    }
    js.push_str("  ]\n}\n");
    std::fs::write(format!("{dir}/e1_results.json"), js)?;
    Ok(())
}
