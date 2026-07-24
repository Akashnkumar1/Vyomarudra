//! Byte-level BPE tokenizer (Project Vyomarudra / Vyomarudra) — built from scratch, ours.
//!
//! Base vocabulary = the 256 bytes (so ANY input encodes, no UNK). Training
//! iteratively merges the most frequent adjacent token pair into a new token, up
//! to a target vocab size. This lifts our models off the char-level ceiling: fewer
//! tokens per sequence, meaningful subword units, better modeling — with zero
//! external dependency (no teacher, no library).
//!
//! Run:  DATASET=text VOCAB=512 ./target/release/vyoma-tokenizer
//! Env:  DATASET (text|distilled), VOCAB (target vocab size, ≥257).

use anyhow::{Context, Result};
use std::collections::HashMap;

/// Split bytes into "words": each whitespace byte starts a new word (kept as its
/// first byte), so concatenating all words reproduces the input exactly (reversible),
/// and merges never cross word boundaries.
fn pretokenize(bytes: &[u8]) -> Vec<Vec<u32>> {
    let mut words: Vec<Vec<u32>> = Vec::new();
    let mut cur: Vec<u32> = Vec::new();
    for &b in bytes {
        if (b == b' ' || b == b'\n' || b == b'\t') && !cur.is_empty() {
            words.push(std::mem::take(&mut cur));
        }
        cur.push(b as u32);
    }
    if !cur.is_empty() { words.push(cur); }
    words
}

fn merge_seq(w: &[u32], a: u32, b: u32, new: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(w.len());
    let mut i = 0;
    while i < w.len() {
        if i + 1 < w.len() && w[i] == a && w[i + 1] == b { out.push(new); i += 2; }
        else { out.push(w[i]); i += 1; }
    }
    out
}

struct Bpe {
    merges: Vec<(u32, u32)>,               // ordered
    rank: HashMap<(u32, u32), usize>,      // pair -> merge index
    id_of: HashMap<(u32, u32), u32>,       // pair -> new token id
    bytes_of: Vec<Vec<u8>>,                // token id -> its bytes (for decode)
}

impl Bpe {
    fn train(words_freq: &[(Vec<u32>, usize)], num_merges: usize) -> Self {
        let mut words: Vec<(Vec<u32>, usize)> = words_freq.to_vec();
        let mut merges = Vec::new();
        let (mut rank, mut id_of) = (HashMap::new(), HashMap::new());
        let mut bytes_of: Vec<Vec<u8>> = (0..256u32).map(|b| vec![b as u8]).collect();
        for m in 0..num_merges {
            let mut pc: HashMap<(u32, u32), i64> = HashMap::new();
            for (w, f) in &words {
                for p in w.windows(2) { *pc.entry((p[0], p[1])).or_insert(0) += *f as i64; }
            }
            let best = pc.iter().filter(|(_, c)| **c > 0)
                .max_by_key(|(k, c)| (**c, std::cmp::Reverse(**k)));
            let (a, b) = match best { Some((k, _)) => *k, None => break };
            let new = 256 + m as u32;
            merges.push((a, b));
            rank.insert((a, b), m);
            id_of.insert((a, b), new);
            let mut nb = bytes_of[a as usize].clone();
            nb.extend_from_slice(&bytes_of[b as usize]);
            bytes_of.push(nb);
            for (w, _) in words.iter_mut() { *w = merge_seq(w, a, b, new); }
        }
        Bpe { merges, rank, id_of, bytes_of }
    }

    fn encode(&self, bytes: &[u8]) -> Vec<u32> {
        let mut out = Vec::new();
        for word in pretokenize(bytes) {
            let mut seq = word;
            loop {
                let mut best: Option<(usize, usize)> = None; // (rank, pos)
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

    fn decode(&self, ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids { out.extend_from_slice(&self.bytes_of[id as usize]); }
        out
    }
    fn vocab_size(&self) -> usize { 256 + self.merges.len() }
}

fn main() -> Result<()> {
    let dataset = std::env::var("DATASET").unwrap_or_else(|_| "text".into());
    let vocab: usize = std::env::var("VOCAB").ok().and_then(|s| s.parse().ok()).unwrap_or(512);
    let file = if dataset == "distilled" { "distilled.txt" } else { "tinyshakespeare.txt" };
    let path = format!("{}/../vyoma-lm/data_cache/{file}", env!("CARGO_MANIFEST_DIR"));
    let bytes = std::fs::read(&path).with_context(|| format!("need corpus at {path}"))?;
    let num_merges = vocab.saturating_sub(256);
    println!("[bpe] corpus={} ({} bytes)  target vocab={vocab} ({num_merges} merges)", file, bytes.len());

    // dedup words → (word, freq)
    let mut wf: HashMap<Vec<u32>, usize> = HashMap::new();
    for w in pretokenize(&bytes) { *wf.entry(w).or_insert(0) += 1; }
    let words_freq: Vec<(Vec<u32>, usize)> = wf.into_iter().collect();
    println!("[bpe] {} unique words", words_freq.len());

    let bpe = Bpe::train(&words_freq, num_merges);

    // compression: bytes per token on the corpus
    let ids = bpe.encode(&bytes);
    let bpt = bytes.len() as f64 / ids.len() as f64;
    // round-trip check
    let ok = bpe.decode(&ids) == bytes;
    println!("[bpe] vocab={}  tokens={}  compression={:.2} bytes/token (char-level = 1.00)  round-trip={}",
        bpe.vocab_size(), ids.len(), bpt, if ok { "OK ✓" } else { "MISMATCH ✗" });

    // show a sample of learned subwords (merges resolved to text)
    println!("[bpe] sample learned tokens:");
    let show: Vec<usize> = [256, 260, 280, 320, 400, 500].into_iter()
        .filter(|&i| i < bpe.bytes_of.len()).collect();
    for i in show {
        let s = String::from_utf8_lossy(&bpe.bytes_of[i]);
        println!("[bpe]   token {i:4} = {:?}", s);
    }
    // longest merged tokens = the most "word-like" units it discovered
    let mut by_len: Vec<(usize, &Vec<u8>)> = bpe.bytes_of.iter().enumerate().skip(256).collect();
    by_len.sort_by_key(|(_, b)| std::cmp::Reverse(b.len()));
    let longest: Vec<String> = by_len.iter().take(12)
        .map(|(_, b)| format!("{:?}", String::from_utf8_lossy(b))).collect();
    println!("[bpe] longest units learned: {}", longest.join(" "));

    // save merges (ours, reusable by vyoma-lm later)
    let out = format!("{}/../vyoma-lm/data_cache/bpe_merges.txt", env!("CARGO_MANIFEST_DIR"));
    let dump: String = bpe.merges.iter().map(|(a, b)| format!("{a} {b}\n")).collect();
    std::fs::write(&out, dump)?;
    println!("[bpe] wrote {} merges to {out}", bpe.merges.len());
    println!("[bpe] done. {:.2} bytes/token ⇒ ~{:.0}% fewer tokens/sequence than char-level.", bpt, (1.0 - 1.0 / bpt) * 100.0);
    Ok(())
}
