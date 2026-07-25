//! vyoma-embed (bin) — experiment harness for our learned retriever + store.
//!
//! The reusable retriever/store lives in `lib.rs` (used by `vyoma-lm` too). This
//! binary is the evaluation driver: it trains the encoder on tiny-shakespeare
//! passages and measures it head-to-head against a classical trigram baseline
//! (dimension efficiency, noise robustness), reports quantized store cost, and can
//! build the persistent on-disk Ontological Store (`MODE=store`).
//!
//! Two claims vs the trigram baseline: (1) DIMENSION EFFICIENCY — reach the trigram
//! plateau at a far smaller dk; (2) ROBUSTNESS under query corruption. Train/test
//! passages are DISJOINT (generalization, not memorization).

use anyhow::Result;
use candle_core::Device;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use vyoma_embed::{
    dot, embed_all, load_store, nearest, train_encoder, write_store, Encoder,
};

const PASSAGE_LEN: usize = 128; // each "fact" is a 128-byte passage
const QUERY_LEN: usize = 48; // a query is a 48-byte fragment of a passage

// ---------------------------------------------------------------------------
// Classical trigram-bag baseline (kept here so the comparison is head-to-head).
// ---------------------------------------------------------------------------
fn hash_trigram(a: u8, b: u8, c: u8, dk: usize) -> usize {
    let h = (a as u64).wrapping_mul(2654435761)
        ^ (b as u64).wrapping_mul(40503).rotate_left(7)
        ^ (c as u64).wrapping_mul(2246822519).rotate_left(13);
    (h % dk as u64) as usize
}
fn trigram_embed(text: &[u8], dk: usize) -> Vec<f32> {
    let mut v = vec![0f32; dk];
    if text.len() >= 3 {
        for w in text.windows(3) {
            v[hash_trigram(w[0], w[1], w[2], dk)] += 1.0;
        }
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// Per-vector symmetric fake-quant + re-L2-norm (in-memory store-cost eval).
fn quant_vec(v: &[f32], bits: u32) -> Vec<f32> {
    let maxabs = v.iter().fold(0f32, |m, &x| m.max(x.abs()));
    if maxabs == 0.0 {
        return v.to_vec();
    }
    let levels = ((1u32 << (bits - 1)) - 1) as f32;
    let scale = maxabs / levels;
    let mut q: Vec<f32> = v
        .iter()
        .map(|&x| (x / scale).round().clamp(-levels, levels) * scale)
        .collect();
    let norm = q.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut q {
            *x /= norm;
        }
    }
    q
}

/// Retrieval accuracy: for a sampled subset of queries, is the nearest passage the
/// source? (every 7th query, matching E2b's quick unbiased estimate).
fn retrieval_acc(query_embs: &[Vec<f32>], passage_embs: &[Vec<f32>]) -> f64 {
    let (mut correct, mut count, mut i) = (0usize, 0usize, 0usize);
    while i < query_embs.len() {
        if nearest(&query_embs[i], passage_embs) == i {
            correct += 1;
        }
        count += 1;
        i += 7;
    }
    correct as f64 / count as f64
}

fn center_fragment(p: &[u8]) -> Vec<u8> {
    p[(PASSAGE_LEN - QUERY_LEN) / 2..][..QUERY_LEN].to_vec()
}
fn corrupt(frag: &[u8], frac: f64, rng: &mut StdRng) -> Vec<u8> {
    frag.iter()
        .map(|&b| if rng.gen::<f64>() < frac { rng.gen::<u8>() } else { b })
        .collect()
}
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn main() -> Result<()> {
    let dev = Device::Cpu;
    let steps = env_usize("STEPS", 1200);
    let bsz = env_usize("BATCH", 96);
    let d = env_usize("DMODEL", 48);
    let n_layers = env_usize("LAYERS", 1);
    let lr = 1e-3;
    let temp = 0.05f64;

    let path = format!("{}/../vyoma-lm/data_cache/tinyshakespeare.txt", env!("CARGO_MANIFEST_DIR"));
    let bytes = std::fs::read(&path)
        .map_err(|e| anyhow::anyhow!("need tinyshakespeare at {path}: {e}"))?;
    let all: Vec<Vec<u8>> = bytes
        .chunks(PASSAGE_LEN)
        .filter(|c| c.len() == PASSAGE_LEN)
        .map(|c| c.to_vec())
        .collect();
    let n_train = all.len() * 8 / 10;
    let (train, test) = all.split_at(n_train);
    let test: Vec<Vec<u8>> = test.to_vec();
    println!(
        "[embed] tiny-shakespeare: {} passages ({PASSAGE_LEN}B). train={} test={} (disjoint).",
        all.len(), train.len(), test.len()
    );
    println!("[embed] encoder = byte-embed -> {n_layers}x diagonal-SSM block(s) -> mean-pool -> proj -> L2. InfoNCE temp={temp}.");
    println!("[embed] steps={steps} batch={bsz} d_model={d} layers={n_layers} lr={lr}\n");

    // Pair sampler for training: (random 48B window, full passage). Inlined fresh
    // at each call site (a fn returning a new closure that borrows `train`).
    let sampler = || {
        |rng: &mut StdRng| {
            let p = &train[rng.gen_range(0..train.len())];
            let off = rng.gen_range(0..=(PASSAGE_LEN - QUERY_LEN));
            (p[off..off + QUERY_LEN].to_vec(), p.clone())
        }
    };

    // Fixed test queries: clean + corruption levels, shared across all encoders.
    let test_frags_clean: Vec<Vec<u8>> = test.iter().map(|p| center_fragment(p)).collect();
    let noise_levels = [0.0f64, 0.15, 0.30];
    let mut noise_rng = StdRng::seed_from_u64(777);
    let test_frags: Vec<Vec<Vec<u8>>> = noise_levels
        .iter()
        .map(|&f| {
            test_frags_clean
                .iter()
                .map(|frag| if f == 0.0 { frag.clone() } else { corrupt(frag, f, &mut noise_rng) })
                .collect()
        })
        .collect();

    // -------- MODE=gate: does the retriever know what it does NOT know? --------
    // Directly measures the grounding signal the RAG gate depends on: for probes
    // from several domains, how far above the store's own similarity distribution
    // does the top hit sit (z-score)? In-domain should be clearly higher than
    // out-of-domain; if they overlap, no threshold can gate (our earlier negative).
    if std::env::var("MODE").as_deref() == Ok("gate") {
        let encp = format!("{}/data_cache/retriever.safetensors", env!("CARGO_MANIFEST_DIR"));
        let storep = format!("{}/data_cache/ontological_store.vyst", env!("CARGO_MANIFEST_DIR"));
        let enc = vyoma_embed::Encoder::load(&encp, &dev)?;
        let (embs, _recs) = load_store(&storep)?;
        println!("== MODE=gate: grounding discrimination (store={} facts, dk={}) ==", embs.len(), enc.dk);
        let probes: Vec<(&str, &str)> = vec![
            ("IN  shakespeare", "Who will believe thee, Isabel? My unsoil'd name,"),
            ("IN  shakespeare", "But soft, what light through yonder window breaks?"),
            ("OOD distilled  ", "Photosynthesis is the process by which plants convert sunlight"),
            ("OOD distilled  ", "The teacher explained that water expands when it freezes"),
            ("OOD code       ", "def quicksort(arr): return sorted(arr) # python function"),
            ("OOD code       ", "fn main() { let x: Vec<u32> = vec![1,2,3]; println!(\"{:?}\", x); }"),
            ("OOD random     ", "zzzq xkcd 99182 ~~~ !!! qqqq zzzz 00000 ????"),
        ];
        println!("   domain          |  top-cos |  z-score | probe");
        for (tag, text) in &probes {
            let q = embed_all(&enc, &[text.as_bytes().to_vec()], &dev)?;
            let sims: Vec<f32> = embs.iter().map(|e| dot(&q[0], e)).collect();
            let top = sims.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let n = sims.len() as f32;
            let mean = sims.iter().sum::<f32>() / n;
            let sd = (sims.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / n).sqrt().max(1e-6);
            println!("   {tag} |  {top:+.3}  |  {:6.2}σ | {:?}", (top - mean) / sd, &text.chars().take(46).collect::<String>());
        }

        // ---- proper CALIBRATION: distributions, not hand-picked probes ----
        // A handful of probes gave a misleadingly clean gap; a real threshold needs
        // real distributions. Sample many in-domain queries (held-out passages) and
        // many out-of-domain ones (the NEG corpus), then report the best achievable
        // separation and its error rates. This is what a gate should be tuned on.
        let ncal = env_usize("NCAL", 200);
        let top_cos = |text: &[u8], enc: &vyoma_embed::Encoder| -> Result<f32> {
            let q = embed_all(enc, &[text.to_vec()], &dev)?;
            Ok(embs.iter().map(|e| dot(&q[0], e)).fold(f32::NEG_INFINITY, f32::max))
        };
        let mut in_s: Vec<f32> = Vec::new();
        for p in test.iter().take(ncal) { in_s.push(top_cos(&center_fragment(p), &enc)?); }
        let mut ood_s: Vec<f32> = Vec::new();
        if let Ok(f) = std::env::var("NEG") {
            let path = format!("{}/../vyoma-lm/data_cache/{f}", env!("CARGO_MANIFEST_DIR"));
            if let Ok(b) = std::fs::read(&path) {
                for c in b.chunks(QUERY_LEN).filter(|c| c.len() == QUERY_LEN).take(ncal) {
                    ood_s.push(top_cos(c, &enc)?);
                }
            }
        }
        if !in_s.is_empty() && !ood_s.is_empty() {
            let pct = |v: &mut Vec<f32>, q: f64| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[((v.len() - 1) as f64 * q) as usize] };
            let (mut a, mut b) = (in_s.clone(), ood_s.clone());
            // pick the threshold that maximizes balanced accuracy over the samples
            let mut cands: Vec<f32> = a.iter().chain(b.iter()).cloned().collect();
            cands.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let (mut best_t, mut best_acc) = (0f32, 0f32);
            for &t in &cands {
                let tpr = a.iter().filter(|&&s| s >= t).count() as f32 / a.len() as f32;
                let tnr = b.iter().filter(|&&s| s < t).count() as f32 / b.len() as f32;
                let acc = (tpr + tnr) / 2.0;
                if acc > best_acc { best_acc = acc; best_t = t; }
            }
            let tpr = a.iter().filter(|&&s| s >= best_t).count() as f32 / a.len() as f32;
            let tnr = b.iter().filter(|&&s| s < best_t).count() as f32 / b.len() as f32;
            println!("\n[gate] CALIBRATION over {} in-domain / {} out-of-domain samples:", a.len(), b.len());
            println!("[gate]   in-domain  cos: p05={:.3} median={:.3} p95={:.3}", pct(&mut a.clone(), 0.05), pct(&mut a, 0.5), pct(&mut a.clone(), 0.95));
            println!("[gate]   out-domain cos: p05={:.3} median={:.3} p95={:.3}", pct(&mut b.clone(), 0.05), pct(&mut b, 0.5), pct(&mut b.clone(), 0.95));
            println!("[gate]   best threshold GATE={best_t:.3} → balanced acc {:.1}% (accepts {:.1}% of in-domain, rejects {:.1}% of out-of-domain)",
                     best_acc * 100.0, tpr * 100.0, tnr * 100.0);
            println!("[gate]   ⇒ set GATE={best_t:.3} in MODE=rag. Overlap here is the honest false-accept/false-veto rate.");
        }
        return Ok(());
    }

    // -------- MODE=store: build the real persistent on-disk Ontological Store --------
    if std::env::var("MODE").as_deref() == Ok("store") {
        let dk = env_usize("DK", 128);
        println!("== MODE=store: our encoder -> int8 -> persistent on-disk store (Pillar 4) ==");
        // OUT-OF-DOMAIN NEGATIVES (NEG=<file in vyoma-lm/data_cache>, NNEG=count).
        // Without these the encoder only learns "which passage?", never "is this my
        // domain?", and the grounding gate has no signal (see docs/PROGRESS.md).
        let neg_file = std::env::var("NEG").ok();
        let n_neg = env_usize("NNEG", if neg_file.is_some() { 32 } else { 0 });
        let neg_chunks: Vec<Vec<u8>> = match &neg_file {
            Some(f) => {
                let p = format!("{}/../vyoma-lm/data_cache/{f}", env!("CARGO_MANIFEST_DIR"));
                let b = std::fs::read(&p).map_err(|e| anyhow::anyhow!("NEG corpus {p}: {e}"))?;
                let v: Vec<Vec<u8>> = b.chunks(PASSAGE_LEN).filter(|c| c.len() == PASSAGE_LEN).map(|c| c.to_vec()).collect();
                println!("[store] out-of-domain negatives: {f} ({} passages, {n_neg}/batch)", v.len());
                v
            }
            None => { println!("[store] no NEG corpus given — gate will have no out-of-domain signal"); vec![] }
        };
        let (enc, loss) = if neg_chunks.is_empty() {
            train_encoder(d, dk, n_layers, steps, bsz, lr, temp, &dev, 1234 + dk as u64, sampler())?
        } else {
            vyoma_embed::train_encoder_neg(
                d, dk, n_layers, steps, bsz, lr, temp, &dev, 1234 + dk as u64, sampler(),
                Some(|r: &mut StdRng| neg_chunks[r.gen_range(0..neg_chunks.len())].clone()),
                n_neg,
            )?
        };
        println!("[store] encoder trained (dk={dk}, final InfoNCE loss {loss:.3}). Encoding {} facts...", test.len());
        let embs = embed_all(&enc, &test, &dev)?;
        let store_path = format!("{}/data_cache/ontological_store.vyst", env!("CARGO_MANIFEST_DIR"));
        std::fs::create_dir_all(format!("{}/data_cache", env!("CARGO_MANIFEST_DIR")))?;
        let size = write_store(&store_path, &test, &embs)?;
        let (disk_embs, disk_recs) = load_store(&store_path)?;
        // Persist the encoder next to the store — the int8 keys are only meaningful
        // to the encoder that produced them, so they are a matched pair.
        let enc_path = format!("{}/data_cache/retriever.safetensors", env!("CARGO_MANIFEST_DIR"));
        enc.save(&enc_path)?;
        println!("[store] wrote {} facts to {store_path}", disk_recs.len());
        println!("[store] wrote retriever -> {enc_path} (store+encoder are a matched pair)");
        println!("[store] file {:.1} KB = {:.1} bytes/fact (int8 key dk={dk} + text)", size as f64 / 1024.0, size as f64 / test.len() as f64);
        let q_clean = embed_all(&enc, &test_frags[0], &dev)?;
        let q_noise = embed_all(&enc, &test_frags[1], &dev)?;
        println!("[store] retrieval FROM DISK: clean {:.3}  noise15 {:.3}", retrieval_acc(&q_clean, &disk_embs), retrieval_acc(&q_noise, &disk_embs));
        let probe = 3usize.min(test.len() - 1);
        let hit = nearest(&q_clean[probe], &disk_embs);
        println!("[store] probe: {:?}", String::from_utf8_lossy(&test_frags[0][probe]));
        println!("[store]  -> fetched ({}): {:?}", if hit == probe { "✓" } else { "✗" }, String::from_utf8_lossy(&disk_recs[hit][..64.min(disk_recs[hit].len())]));
        println!("[store] Knowledge on disk (ours), encoder (skills) in RAM. No teacher anywhere.");
        return Ok(());
    }

    // -------- Trigram baseline (no training) --------
    println!("== trigram baseline (classical, no training) — retrieval acc on {} test passages ==", test.len());
    println!("   dk  |  clean  noise15  noise30");
    for &dk in &[64usize, 128, 256, 1024] {
        let pe: Vec<Vec<f32>> = test.iter().map(|p| trigram_embed(p, dk)).collect();
        let mut line = format!("  {dk:5} |");
        for lvl in 0..noise_levels.len() {
            let qe: Vec<Vec<f32>> = test_frags[lvl].iter().map(|q| trigram_embed(q, dk)).collect();
            line += &format!("  {:.3} ", retrieval_acc(&qe, &pe));
        }
        println!("{line}");
    }

    // -------- Learned encoder (ours), across output dim dk --------
    let dk_list: Vec<usize> = match std::env::var("DK").ok().and_then(|s| s.parse().ok()) {
        Some(dk) => vec![dk],
        None => vec![32, 64, 128],
    };
    println!("\n== learned SSM encoder (ours, trained, {n_layers} layer(s)) — retrieval acc on {} test passages ==", test.len());
    println!("   dk  | params |  clean  noise15  noise30");
    for &dk in &dk_list {
        let (enc, loss): (Encoder, f32) =
            train_encoder(d, dk, n_layers, steps, bsz, lr, temp, &dev, 1234 + dk as u64, sampler())?;
        print!("  {dk:5} | {:6} ", enc.n_params());
        let pe = embed_all(&enc, &test, &dev)?;
        for lvl in 0..noise_levels.len() {
            let qe = embed_all(&enc, &test_frags[lvl], &dev)?;
            print!(" {:.3} ", retrieval_acc(&qe, &pe));
        }
        println!("   (final InfoNCE loss {loss:.3})");

        if std::env::var("QUANT").is_ok() {
            let pe32 = embed_all(&enc, &test, &dev)?;
            let qe32 = embed_all(&enc, &test_frags[0], &dev)?;
            for &bits in &[32u32, 8, 4] {
                let (pe, qe): (Vec<Vec<f32>>, Vec<Vec<f32>>) = if bits == 32 {
                    (pe32.clone(), qe32.clone())
                } else {
                    (pe32.iter().map(|v| quant_vec(v, bits)).collect(),
                     qe32.iter().map(|v| quant_vec(v, bits)).collect())
                };
                let _ = dot; // (nearest uses dot internally)
                println!("  [quant] dk={dk:4} {bits:2}-bit: clean acc {:.3}  bytes/fact={}", retrieval_acc(&qe, &pe), (dk * bits as usize + 7) / 8);
            }
        }
    }

    println!("\n[embed] learned reaching the trigram plateau at a SMALLER dk = cheaper store per fact;");
    println!("[embed] the retriever + store are the RETRO-quality engine of Pillar 4, built entirely by us.");
    Ok(())
}
