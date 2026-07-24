//! E1→E2 bridge — can the fractal "Eternal Seed" generate the weights of an
//! actual SEQUENCE model (an SSM), not just an MLP?
//!
//! Pillar 1 (generative weights) meets Pillar 2 (SSM backbone). The target is a
//! minimal diagonal state-space model — a stable per-channel linear recurrence
//! h_t = A⊙h_{t-1} + B⊙u_t with A = σ(a)∈(0,1) (Lyapunov-stable by construction),
//! read out y = C⊙h_T + D⊙u_T, mixed, then classified. We run it on row-wise
//! MNIST (28 timesteps of 28-dim rows) — real sequence modeling.
//!
//! Same honest test as E1: fractal-generated SSM vs. a same-size plain SSM vs.
//! the dense SSM upper bound.

use anyhow::Result;
use candle_core::{DType, Device, Tensor, Var, D};
use candle_nn::{optim::{AdamW, ParamsAdamW}, Optimizer};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ---------------------------------------------------------------------------
// SSM target geometry
// ---------------------------------------------------------------------------
#[derive(Clone, Copy)]
struct SsmSpec { d_in: usize, d_model: usize, n_classes: usize }

impl SsmSpec {
    fn n_params(&self) -> usize {
        let (di, dm, nc) = (self.d_in, self.d_model, self.n_classes);
        dm * di + dm            // w_enc, b_enc
            + 4 * dm            // a_raw, B, C, D
            + dm * dm + dm      // w_mix, b_mix
            + nc * dm + nc      // w_cls, b_cls
    }
    /// pick d_model so an SSM has ~target params (solve the quadratic in d_model)
    fn width_for(&self, target: usize) -> usize {
        let (di, nc) = (self.d_in as f64, self.n_classes as f64);
        let b = di + 6.0 + nc;            // linear coefficient
        let c = nc - target as f64;       // constant (negative)
        let disc = (b * b - 4.0 * c).max(0.0);
        (((-b + disc.sqrt()) / 2.0).floor() as usize).max(1)
    }
}

/// Per-parameter init-scale prior (Kaiming-style), in ssm_forward slice order.
/// Derived purely from the architecture, so it costs O(1) to store (a formula),
/// not O(P). It gives each parameter group the magnitude it needs — critical for
/// the SSM, whose encoder / recurrence / readout want very different scales.
fn group_std_prior(spec: &SsmSpec) -> Vec<f32> {
    let (di, dm, nc) = (spec.d_in, spec.d_model, spec.n_classes);
    let mut v = Vec::with_capacity(spec.n_params());
    let mut push = |std: f32, count: usize, v: &mut Vec<f32>| v.extend(std::iter::repeat(std).take(count));
    push((1.0 / di as f32).sqrt(), dm * di, &mut v); // w_enc
    push(0.01, dm, &mut v);                          // b_enc
    push(1.0, dm, &mut v);                           // a_raw  (spread A=σ(a) over timescales)
    push(0.5, dm, &mut v);                           // B
    push(0.5, dm, &mut v);                           // C
    push(0.5, dm, &mut v);                           // D
    push((1.0 / dm as f32).sqrt(), dm * dm, &mut v); // w_mix
    push(0.01, dm, &mut v);                          // b_mix
    push((1.0 / dm as f32).sqrt(), nc * dm, &mut v); // w_cls
    push(0.01, nc, &mut v);                          // b_cls
    v
}

/// Prior for the GENERATED feedforward matrices only, order: w_enc, w_mix, w_cls.
fn gen_prior(spec: &SsmSpec) -> Vec<f32> {
    let (di, dm, nc) = (spec.d_in, spec.d_model, spec.n_classes);
    let mut v = Vec::new();
    v.extend(std::iter::repeat((1.0 / di as f32).sqrt()).take(dm * di)); // w_enc
    v.extend(std::iter::repeat((1.0 / dm as f32).sqrt()).take(dm * dm)); // w_mix
    v.extend(std::iter::repeat((1.0 / dm as f32).sqrt()).take(nc * dm)); // w_cls
    v
}
fn gen_params(spec: &SsmSpec) -> usize {
    spec.d_model * spec.d_in + spec.d_model * spec.d_model + spec.n_classes * spec.d_model
}
/// Prior for the STORED params, order: b_enc, a_raw, B, C, D, b_mix, b_cls.
fn stored_prior(spec: &SsmSpec) -> Vec<f32> {
    let (dm, nc) = (spec.d_model, spec.n_classes);
    let mut v = Vec::new();
    let mut push = |std: f32, n: usize, v: &mut Vec<f32>| v.extend(std::iter::repeat(std).take(n));
    push(0.01, dm, &mut v); // b_enc
    push(1.0, dm, &mut v);  // a_raw
    push(0.5, dm, &mut v);  // B
    push(0.5, dm, &mut v);  // C
    push(0.5, dm, &mut v);  // D
    push(0.01, dm, &mut v); // b_mix
    push(0.01, nc, &mut v); // b_cls
    v
}

fn ssm_forward(x: &Tensor, flat: &Tensor, spec: &SsmSpec) -> Result<Tensor> {
    let (di, dm, nc) = (spec.d_in, spec.d_model, spec.n_classes);
    let mut o = 0usize;
    let mut take = |rows: usize, cols: usize, flat: &Tensor| -> Result<Tensor> {
        let t = if cols == 1 { flat.narrow(0, o, rows)? }
                else { flat.narrow(0, o, rows * cols)?.reshape((rows, cols))? };
        o += rows * cols;
        Ok(t)
    };
    let w_enc = take(dm, di, flat)?;
    let b_enc = take(dm, 1, flat)?;
    let a_raw = take(dm, 1, flat)?;
    let b_coef = take(dm, 1, flat)?;
    let c_coef = take(dm, 1, flat)?;
    let d_coef = take(dm, 1, flat)?;
    let w_mix = take(dm, dm, flat)?;
    let b_mix = take(dm, 1, flat)?;
    let w_cls = take(nc, dm, flat)?;
    let b_cls = take(nc, 1, flat)?;

    let (bsz, t_len, _) = x.dims3()?;
    let a = candle_nn::ops::sigmoid(&a_raw)?;          // (dm,) in (0,1) => stable
    let mut h = Tensor::zeros((bsz, dm), DType::F32, x.device())?;
    let mut last_u: Option<Tensor> = None;
    for t in 0..t_len {
        let xt = x.narrow(1, t, 1)?.reshape((bsz, di))?;
        let u = xt.matmul(&w_enc.t()?)?.broadcast_add(&b_enc)?.gelu()?; // (b, dm)
        h = h.broadcast_mul(&a)?.add(&u.broadcast_mul(&b_coef)?)?;
        last_u = Some(u);
    }
    let u = last_u.unwrap();
    let y = h.broadcast_mul(&c_coef)?.add(&u.broadcast_mul(&d_coef)?)?;
    let mix = y.matmul(&w_mix.t()?)?.broadcast_add(&b_mix)?.gelu()?;
    Ok(mix.matmul(&w_cls.t()?)?.broadcast_add(&b_cls)?)
}

// ---------------------------------------------------------------------------
// Var helpers + fractal seed (generalized to a raw target param count)
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
    pe: Tensor,
    prior: Tensor, // fixed per-position init-scale prior (length = n_params_target)
    a_w1: Var, a_b1: Var, a_w2: Var, a_b2: Var,
    g_w1: Var, g_b1: Var, g_w2: Var, g_b2: Var,
    gscale: Var,
    n_params_target: usize,
}

impl FractalSeed {
    fn new(n_params_target: usize, prior_vals: Vec<f32>, chunk: usize, embed_dim: usize, hyper_hidden: usize, dev: &Device) -> Result<Self> {
        debug_assert_eq!(prior_vals.len(), n_params_target);
        let n_chunks = (n_params_target + chunk - 1) / chunk;
        Ok(Self {
            pe: positional_encoding(n_chunks, PE_DIM, dev)?,
            prior: Tensor::from_vec(prior_vals, (n_params_target,), dev)?,
            a_w1: var_randn((PE_DIM, ADDR_HIDDEN), (1.0 / PE_DIM as f64).sqrt(), dev)?,
            a_b1: var_randn1(ADDR_HIDDEN, 1e-6, dev)?,
            a_w2: var_randn((ADDR_HIDDEN, embed_dim), (1.0 / ADDR_HIDDEN as f64).sqrt(), dev)?,
            a_b2: var_randn1(embed_dim, 1e-6, dev)?,
            g_w1: var_randn((embed_dim, hyper_hidden), (1.0 / embed_dim as f64).sqrt(), dev)?,
            g_b1: var_randn1(hyper_hidden, 1e-6, dev)?,
            g_w2: var_randn((hyper_hidden, chunk), (1.0 / hyper_hidden as f64).sqrt(), dev)?,
            g_b2: var_randn1(chunk, 1e-6, dev)?,
            gscale: var_full1(1, 1.0, dev)?,
            n_params_target,
        })
    }
    fn vars(&self) -> Vec<Var> {
        vec![self.a_w1.clone(), self.a_b1.clone(), self.a_w2.clone(), self.a_b2.clone(),
             self.g_w1.clone(), self.g_b1.clone(), self.g_w2.clone(), self.g_b2.clone(),
             self.gscale.clone()]
    }
    fn seed_params(&self) -> usize { self.vars().iter().map(|v| v.elem_count()).sum() }
    fn generate(&self) -> Result<Tensor> {
        let emb = self.pe.matmul(self.a_w1.as_tensor())?.broadcast_add(self.a_b1.as_tensor())?
            .gelu()?.matmul(self.a_w2.as_tensor())?.broadcast_add(self.a_b2.as_tensor())?;
        let h = emb.matmul(self.g_w1.as_tensor())?.broadcast_add(self.g_b1.as_tensor())?.gelu()?;
        let raw = h.matmul(self.g_w2.as_tensor())?.broadcast_add(self.g_b2.as_tensor())?;
        let flat = raw.flatten_all()?.narrow(0, 0, self.n_params_target)?;
        // normalize to unit variance, then impose per-group scale prior + learned gain
        let std = flat.sqr()?.mean_all()?.sqrt()?;
        let unit = flat.broadcast_div(&(std + 1e-6)?)?;
        Ok(unit.broadcast_mul(&self.prior)?.broadcast_mul(self.gscale.as_tensor())?)
    }
}

// A plain trainable flat weight vector for an SSM, initialized with the same
// per-group scale prior (fair init for dense and small baselines alike).
fn plain_ssm_flat(spec: &SsmSpec, dev: &Device) -> Result<Var> {
    let n = spec.n_params();
    let noise = Tensor::randn(0f32, 1f32, (n,), dev)?;
    let prior = Tensor::from_vec(group_std_prior(spec), (n,), dev)?;
    Ok(Var::from_tensor(&noise.mul(&prior)?)?)
}

// Stored (directly-trained) recurrence + bias params for the hybrid.
fn stored_init(spec: &SsmSpec, dev: &Device) -> Result<Var> {
    let prior = stored_prior(spec);
    let n = prior.len();
    let noise = Tensor::randn(0f32, 1f32, (n,), dev)?;
    Ok(Var::from_tensor(&noise.mul(&Tensor::from_vec(prior, (n,), dev)?)?)?)
}

/// Hybrid forward: generated feedforward matrices (w_enc, w_mix, w_cls) + stored
/// recurrence/bias params, reassembled into ssm_forward's slice order.
fn ssm_forward_hybrid(x: &Tensor, gen: &Tensor, stored: &Tensor, spec: &SsmSpec) -> Result<Tensor> {
    let (di, dm, nc) = (spec.d_in, spec.d_model, spec.n_classes);
    // gen: [w_enc | w_mix | w_cls]
    let w_enc = gen.narrow(0, 0, dm * di)?;
    let w_mix = gen.narrow(0, dm * di, dm * dm)?;
    let w_cls = gen.narrow(0, dm * di + dm * dm, nc * dm)?;
    // stored: [b_enc | a_raw | B | C | D | b_mix | b_cls]
    let mut o = 0usize;
    let mut s = |n: usize, st: &Tensor| -> Result<Tensor> { let t = st.narrow(0, o, n)?; o += n; Ok(t) };
    let b_enc = s(dm, stored)?;
    let a_raw = s(dm, stored)?;
    let b_co = s(dm, stored)?;
    let c_co = s(dm, stored)?;
    let d_co = s(dm, stored)?;
    let b_mix = s(dm, stored)?;
    let b_cls = s(nc, stored)?;
    // reassemble in ssm_forward order
    let full = Tensor::cat(&[&w_enc, &b_enc, &a_raw, &b_co, &c_co, &d_co, &w_mix, &b_mix, &w_cls, &b_cls], 0)?;
    ssm_forward(x, &full, spec)
}

// ---------------------------------------------------------------------------
// Data — row-wise sequence MNIST/Fashion (28 steps × 28 dims)
// ---------------------------------------------------------------------------
struct Dataset { xtr: Tensor, ytr: Tensor, xte: Tensor, yte: Tensor, name: &'static str }

fn load_idx(dir: &str, name: &'static str) -> Result<Option<Dataset>> {
    let need = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte",
                "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"];
    if !need.iter().all(|f| std::path::Path::new(&format!("{dir}/{f}")).exists()) { return Ok(None); }
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
        xtr: Tensor::from_vec(xtr, (ntr, 28, 28), &dev)?,
        ytr: Tensor::from_vec(labs(&rd("train-labels-idx1-ubyte")?), (ntr,), &dev)?,
        xte: Tensor::from_vec(xte, (nte, 28, 28), &dev)?,
        yte: Tensor::from_vec(labs(&rd("t10k-labels-idx1-ubyte")?), (nte,), &dev)?,
        name,
    }))
}

// ---------------------------------------------------------------------------
// Train / eval
// ---------------------------------------------------------------------------
fn accuracy(logits: &Tensor, y: &Tensor) -> Result<f64> {
    let pred = logits.argmax(D::Minus1)?.to_dtype(DType::U32)?;
    let correct = pred.eq(y)?.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
    Ok(correct as f64 / y.elem_count() as f64)
}

fn eval_batched<F>(forward: &F, x: &Tensor, y: &Tensor, bs: usize) -> Result<f64>
where F: Fn(&Tensor) -> Result<Tensor> {
    let n = x.dim(0)?;
    let (mut correct, mut start) = (0f64, 0usize);
    while start < n {
        let len = bs.min(n - start);
        let logits = forward(&x.narrow(0, start, len)?)?;
        let yb = y.narrow(0, start, len)?;
        correct += accuracy(&logits, &yb)? * len as f64;
        start += len;
    }
    Ok(correct / n as f64)
}

fn train<F>(vars: Vec<Var>, data: &Dataset, ntrain: usize, epochs: usize, lr: f64, bs: usize,
            seed: u64, dev: &Device, forward: F) -> Result<f64>
where F: Fn(&Tensor) -> Result<Tensor> {
    let mut opt = AdamW::new(vars, ParamsAdamW { lr, ..Default::default() })?;
    let n = ntrain.min(data.xtr.dim(0)?);
    let mut rng = StdRng::seed_from_u64(seed);
    for _ in 0..epochs {
        let mut idx: Vec<u32> = (0..n as u32).collect();
        for k in (1..idx.len()).rev() { let j = rng.gen_range(0..=k); idx.swap(k, j); }
        let idx_t = Tensor::from_vec(idx, (n,), dev)?;
        let xs = data.xtr.narrow(0, 0, n)?.index_select(&idx_t, 0)?;
        let ys = data.ytr.narrow(0, 0, n)?.index_select(&idx_t, 0)?;
        let mut start = 0;
        while start < n {
            let len = bs.min(n - start);
            let logits = forward(&xs.narrow(0, start, len)?)?;
            let loss = candle_nn::loss::cross_entropy(&logits, &ys.narrow(0, start, len)?)?;
            opt.backward_step(&loss)?;
            start += len;
        }
    }
    eval_batched(&forward, &data.xte, &data.yte, 512)
}

fn mean_std(v: &[f64]) -> (f64, f64) {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64;
    (m, var.sqrt())
}

fn main() -> Result<()> {
    let epochs: usize = std::env::var("EPOCHS").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let seeds: usize = std::env::var("SEEDS").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    let dm: usize = std::env::var("DMODEL").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let ntrain: usize = std::env::var("TRAIN_SUBSET").ok().and_then(|s| s.parse().ok()).unwrap_or(20000);
    let (lr, bs) = (2e-3, 128usize);
    let dev = Device::Cpu;

    let spec = SsmSpec { d_in: 28, d_model: dm, n_classes: 10 };
    let p = spec.n_params();
    println!("[ssm] diagonal SSM target d_model={dm}  P={p}  epochs={epochs} seeds={seeds} train_subset={ntrain}");

    let kind = std::env::var("DATASET").unwrap_or_else(|_| "mnist".into());
    let (sub, nm): (&str, &'static str) = match kind.as_str() {
        "fashion" => ("data_cache/fashion", "Fashion-MNIST"),
        _ => ("data_cache", "MNIST"),
    };
    // datasets are shared with the vyoma-e1 crate's cache
    let cache = format!("{}/../vyoma-e1/{}", env!("CARGO_MANIFEST_DIR"), sub);
    let data = match load_idx(&cache, nm)? {
        Some(d) => d,
        None => { println!("[ssm] no dataset at {cache}; place MNIST IDX files there (see vyoma-e1)."); return Ok(()); }
    };
    println!("[ssm] dataset={} (row-wise, 28x28)  train={}(subset {})  test={}",
             data.name, data.xtr.dim(0)?, ntrain, data.xte.dim(0)?);

    // Upper bound: dense SSM trained directly.
    let mut full = Vec::new();
    for s in 0..seeds {
        let flat = plain_ssm_flat(&spec, &dev)?;
        let fc = flat.clone();
        full.push(train(vec![flat], &data, ntrain, epochs, lr, bs, s as u64, &dev,
            |xb| ssm_forward(xb, fc.as_tensor(), &spec))?);
    }
    let (full_m, full_s) = mean_std(&full);
    println!("[ssm] UPPER BOUND dense SSM ({p} params): {full_m:.4} ± {full_s:.4}");

    let mode = std::env::var("MODE").unwrap_or_else(|_| "pure".into());
    if mode == "hybrid" {
        let gp = gen_params(&spec);
        let stored_len = stored_prior(&spec).len();
        println!("[ssm] HYBRID: generate FFN mass ({gp} params), store recurrence+biases ({stored_len} params)");
        let configs = [(512usize, 8usize, 16usize), (256, 4, 8), (256, 2, 4)];
        for &(chunk, ed, hh) in &configs {
            let (mut ha, mut pa) = (Vec::new(), Vec::new());
            let (mut seed_f, mut footprint, mut plain_p) = (0usize, 0usize, 0usize);
            for s in 0..seeds {
                let frac = FractalSeed::new(gp, gen_prior(&spec), chunk, ed, hh, &dev)?;
                seed_f = frac.seed_params();
                let stored = stored_init(&spec, &dev)?;
                let stored_c = stored.clone();
                let mut vars = frac.vars();
                vars.push(stored.clone());
                footprint = seed_f + stored_len;
                ha.push(train(vars, &data, ntrain, epochs, lr, bs, s as u64, &dev,
                    |xb| { let g = frac.generate()?; ssm_forward_hybrid(xb, &g, stored_c.as_tensor(), &spec) })?);

                let small = SsmSpec { d_model: spec.width_for(footprint), ..spec };
                plain_p = small.n_params();
                let flat = plain_ssm_flat(&small, &dev)?;
                let fc = flat.clone();
                pa.push(train(vec![flat], &data, ntrain, epochs, lr, bs, s as u64, &dev,
                    |xb| ssm_forward(xb, fc.as_tensor(), &small))?);
            }
            let (hm, _) = mean_std(&ha);
            let (pm, _) = mean_std(&pa);
            let gen_comp = gp as f64 / seed_f as f64;
            let whole_comp = p as f64 / footprint as f64;
            println!("[ssm] gen-mass-comp={gen_comp:5.1}x  whole-comp={whole_comp:4.1}x  footprint={footprint:5}  hybrid={hm:.4}  plain-SSM(={plain_p})={pm:.4}  edge={:+.4}",
                     hm - pm);
        }
        println!("[ssm] done (hybrid)");
        return Ok(());
    }

    let configs = [(512usize, 8usize, 16usize), (256, 4, 8), (256, 2, 4)];
    for &(chunk, ed, hh) in &configs {
        let (mut fa, mut pa) = (Vec::new(), Vec::new());
        let (mut seed_f, mut plain_p) = (0usize, 0usize);
        for s in 0..seeds {
            let frac = FractalSeed::new(spec.n_params(), group_std_prior(&spec), chunk, ed, hh, &dev)?;
            seed_f = frac.seed_params();
            fa.push(train(frac.vars(), &data, ntrain, epochs, lr, bs, s as u64, &dev,
                |xb| { let flat = frac.generate()?; ssm_forward(xb, &flat, &spec) })?);

            let small = SsmSpec { d_model: spec.width_for(seed_f), ..spec };
            plain_p = small.n_params();
            let flat = plain_ssm_flat(&small, &dev)?;
            let fc = flat.clone();
            pa.push(train(vec![flat], &data, ntrain, epochs, lr, bs, s as u64, &dev,
                |xb| ssm_forward(xb, fc.as_tensor(), &small))?);
        }
        let (fm, _fs) = mean_std(&fa);
        let (pm, _ps) = mean_std(&pa);
        let comp = p as f64 / seed_f as f64;
        println!("[ssm] comp={comp:6.1}x seed={seed_f:5}  fractal-SSM={fm:.4}  plain-SSM(={plain_p})={pm:.4}  edge={:+.4}",
                 fm - pm);
    }
    println!("[ssm] done");
    Ok(())
}
