//! E2b — Retrieval quality vs. library size (Ontological Store, Pillar 4).
//!
//! E2 showed that IF you can fetch the right fact, a tiny model wins (oracle
//! retrieval). This asks the question the oracle skipped: **does retrieval stay
//! accurate as the library grows?** — the real determinant of "small brain + big
//! library." No gradient training: passages are embedded with a hashed
//! char-trigram bag (a real, classic text representation), optionally random-
//! projected to `dk` dims, and retrieved by cosine similarity. A query is a short
//! fragment of a source passage; retrieval is correct if the nearest passage is
//! the source. We sweep library size N and embedding dim dk.
//!
//! This is a conservative baseline — learned/neural embeddings (RETRO) do better —
//! so it lower-bounds how big a store a given embedding size can reliably serve.

use anyhow::Result;

fn hash_trigram(a: u8, b: u8, c: u8, dk: usize) -> usize {
    // simple deterministic hash of a byte trigram into [0, dk)
    let h = (a as u64).wrapping_mul(2654435761)
        ^ (b as u64).wrapping_mul(40503).rotate_left(7)
        ^ (c as u64).wrapping_mul(2246822519).rotate_left(13);
    (h % dk as u64) as usize
}

/// Embed a byte slice as an L2-normalized hashed char-trigram histogram in R^dk.
fn embed(text: &[u8], dk: usize) -> Vec<f32> {
    let mut v = vec![0f32; dk];
    if text.len() >= 3 {
        for w in text.windows(3) {
            v[hash_trigram(w[0], w[1], w[2], dk)] += 1.0;
        }
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 { for x in &mut v { *x /= norm; } }
    v
}

fn dot(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() }

fn main() -> Result<()> {
    let path = format!("{}/../vyoma-lm/data_cache/tinyshakespeare.txt", env!("CARGO_MANIFEST_DIR"));
    let bytes = std::fs::read(&path).map_err(|e| anyhow::anyhow!("need tinyshakespeare at {path}: {e}"))?;

    let passage_len = 128usize;         // each "fact" = a 128-byte passage
    let query_len = 48usize;            // query = a 48-byte fragment of the passage
    let passages: Vec<&[u8]> = bytes.chunks(passage_len).filter(|c| c.len() == passage_len).collect();
    let total = passages.len();
    println!("[ret] tiny-shakespeare: {} passages of {passage_len}B  (query fragment {query_len}B)", total);
    println!("[ret] retrieval = nearest passage by cosine over hashed char-trigram embeddings. No training.");

    let query_frag = |p: &[u8]| -> Vec<u8> { p[(passage_len - query_len) / 2..][..query_len].to_vec() };

    for &dk in &[128usize, 256, 1024] {
        // precompute passage embeddings once per dk
        let emb: Vec<Vec<f32>> = passages.iter().map(|p| embed(p, dk)).collect();
        let mut line = format!("[ret] dk={dk:5}");
        for &n in &[100usize, 1000, 5000, total] {
            let n = n.min(total);
            // query every 7th passage in [0,n) for a quick unbiased sample
            let (mut correct, mut count) = (0usize, 0usize);
            let mut i = 0;
            while i < n {
                let q = embed(&query_frag(passages[i]), dk);
                let mut best = (0usize, f32::NEG_INFINITY);
                for j in 0..n {
                    let s = dot(&q, &emb[j]);
                    if s > best.1 { best = (j, s); }
                }
                if best.0 == i { correct += 1; }
                count += 1;
                i += 7;
            }
            line += &format!("  N={n}:{:.3}", correct as f64 / count as f64);
        }
        println!("{line}");
    }
    println!("[ret] done. Flat accuracy as N grows ⇒ the store scales; a drop ⇒ collisions ⇒ need bigger dk / better embeddings.");
    Ok(())
}
