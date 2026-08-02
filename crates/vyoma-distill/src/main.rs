//! Distillation data pipeline (Project Vyomarudra / Vyomarudra).
//!
//! A pretrained model is a TEACHER we learn from — never a component we ship.
//! The teacher (phi4-mini via the local Ollama API) generates a clean, varied
//! corpus written to disk; OUR model (`vyoma-lm DATASET=distilled`) trains on it.
//! The teacher is a tutor; the corpus and the model are entirely ours. This is the
//! roadmap's Phase-1 mechanism ("distill from an open teacher").
//!
//! Scales via COUNT (number of generations). Writes incrementally, so a long
//! background run is never lost. Env: COUNT (default 200), APPEND=1 to grow an
//! existing corpus.

use anyhow::{Context, Result};
use std::io::Write;
use std::time::Duration;

fn topics() -> Vec<&'static str> {
    vec![
        "how rivers shape valleys","why the sky is blue","how bread rises","the life of a honeybee",
        "how a compass works","why leaves change color","how memory works in the brain","the water cycle",
        "how bridges stay up","why the moon has phases","how sound travels","the basics of trade",
        "how seeds become plants","why ice floats","how a clock keeps time","the idea of gravity",
        "how languages change over time","why we sleep","how volcanoes form","what makes rainbows",
        "how birds navigate","why metals rust","how muscles move","the phases of a star's life",
        "how vaccines work","why the ocean is salty","how a battery stores energy","what causes tides",
        "how ants build colonies","why paper turns yellow","how echoes work","the idea of probability",
        "how a seed knows which way is up","why deserts are cold at night","how glass is made",
        "what makes music pleasant","how the heart pumps blood","why the wind blows","how maps are made",
        "the idea of zero","how spiders spin webs","why honey never spoils","how a rainbow trout swims upstream",
        "how fossils form","why cats purr","how a lever multiplies force","what makes a good story",
        "how the postal system routes mail","why bread goes stale","how a telescope sees far",
        "the idea of a cycle in nature","how salt preserves food","why the seasons change",
        "how a plant drinks water","what makes a bell ring","how a key opens a lock","why smoke rises",
    ]
}
fn forms() -> Vec<&'static str> {
    vec![
        "Explain {} simply, in a short clear paragraph.",
        "Write a short story that teaches {}.",
        "Give three plain facts about {}.",
        "Write a brief dialogue between a curious child and a teacher about {}.",
        "Describe {} step by step, briefly.",
        "Answer for a beginner: {} — what is the key idea?",
        "In a few sentences, describe {} to someone who has never heard of it.",
    ]
}

fn ask_once(agent: &ureq::Agent, prompt: &str) -> Result<String> {
    let resp = agent.post("http://localhost:11434/api/generate")
        .send_json(serde_json::json!({
            "model": "phi4-mini", "prompt": prompt, "stream": false,
            "options": {"temperature": 0.8, "num_predict": 320}
        })).context("Ollama call failed (need `ollama serve` + phi4-mini)")?;
    let v: serde_json::Value = resp.into_json()?;
    Ok(v["response"].as_str().unwrap_or("").trim().to_string())
}

/// Retry a few times on transient errors (e.g. a stuck generation timing out) so a
/// single bad call never kills a long batch.
fn ask(agent: &ureq::Agent, prompt: &str, tries: usize) -> Result<String> {
    let mut last = anyhow::anyhow!("no attempt");
    for _ in 0..tries {
        match ask_once(agent, prompt) {
            Ok(t) if !t.is_empty() => return Ok(t),
            Ok(_) => last = anyhow::anyhow!("empty response"),
            Err(e) => last = e,
        }
    }
    Err(last)
}

/// Instruction-tuning corpus. Our LM only ever learned to CONTINUE text, so "hi"
/// produced "hirest;" rather than a reply — it has no notion of a turn. Here the
/// teacher writes (instruction, response) PAIRS, which we wrap in explicit turn
/// markers so OUR model learns the shape of a conversation. The teacher supplies
/// training data only; it is never part of Vyomarudra.
const SFT_USER: &str = "<|user|>";
const SFT_ASSISTANT: &str = "<|assistant|>";
const SFT_END: &str = "<|end|>";

fn instructions() -> Vec<&'static str> {
    vec![
        "Explain {} in two sentences.", "What is {}?", "Give a simple example of {}.",
        "Why does {} matter?", "How would you describe {} to a child?",
        "List two facts about {}.", "What is one common misconception about {}?",
        "Summarise {} briefly.",
    ]
}

fn main() -> Result<()> {
    let count: usize = std::env::var("COUNT").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let append = std::env::var("APPEND").is_ok();
    let sft = std::env::var("SFT").is_ok();
    let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs(180)).build();
    let out = if sft {
        format!("{}/../vyoma-lm/data_cache/sft.txt", env!("CARGO_MANIFEST_DIR"))
    } else {
        format!("{}/../vyoma-lm/data_cache/distilled.txt", env!("CARGO_MANIFEST_DIR"))
    };
    std::fs::create_dir_all(std::path::Path::new(&out).parent().unwrap())?;

    if sft {
        // (instruction, response) pairs wrapped in turn markers
        let (topics, instrs) = (topics(), instructions());
        println!("[distill] SFT mode: teacher writes instruction/response pairs → {out}");
        let mut f = std::fs::OpenOptions::new().create(true).write(true).append(append).truncate(!append).open(&out)?;
        let mut total = 0usize;
        for i in 0..count {
            let topic = topics[i % topics.len()];
            let instr = instrs[(i / topics.len()) % instrs.len()].replace("{}", topic);
            let ask_prompt = format!("{instr}\nAnswer directly and concisely, no preamble.");
            let reply = match ask(&agent, &ask_prompt, 3) {
                Ok(t) => t,
                Err(e) => { eprintln!("[distill] skip {i}: {e}"); continue; }
            };
            let rec = format!("{SFT_USER}\n{instr}\n{SFT_ASSISTANT}\n{}\n{SFT_END}\n\n", reply.trim());
            write!(f, "{rec}")?;
            f.flush()?;
            total += rec.len();
            if i % 10 == 0 || i + 1 == count {
                println!("[distill] {:3}/{count}  (+{} chars, {} total)", i + 1, rec.len(), total);
            }
        }
        println!("\n[distill] done: {total} chars of instruction data → {out}");
        println!("[distill] next: DATASET=sft TOKENIZER=bpe MODE=sft ./target/release/vyoma-lm");
        return Ok(());
    }

    let (topics, forms) = (topics(), forms());
    println!("[distill] teacher=phi4-mini → distilled corpus for OUR model ({count} generations, {} topics × {} forms)", topics.len(), forms.len());
    println!("[distill] the teacher is a tutor; it is NOT part of Vyomarudra. Writing incrementally to {out}\n");

    let mut f = std::fs::OpenOptions::new().create(true).write(true).append(append).truncate(!append).open(&out)?;
    let mut total = 0usize;
    for i in 0..count {
        // combinatorial coverage: cycle topics fastest, forms slower → diverse pairs
        let topic = topics[i % topics.len()];
        let form = forms[(i / topics.len()) % forms.len()];
        let prompt = form.replace("{}", topic);
        let text = match ask(&agent, &prompt, 3) {
            Ok(t) => t,
            Err(e) => { eprintln!("[distill] skip {i}: {e}"); continue; } // never abort the batch
        };
        writeln!(f, "{text}\n")?;
        f.flush()?;
        total += text.len();
        if i % 10 == 0 || i + 1 == count {
            println!("[distill] {:3}/{count}  (+{} chars, {} total)", i + 1, text.len(), total);
        }
    }
    println!("\n[distill] done: {total} chars across {count} generations → {out}");
    println!("[distill] next: DATASET=distilled TOKENIZER=bpe MODE=bpb ./target/release/vyoma-lm  (our model learns from it)");
    Ok(())
}
