//! vyoma-data — turn a FineWeb (or any) parquet shard into training text for OUR
//! model. Reads the `text` column, writes plain UTF-8 to `fineweb.txt` (capped),
//! which `vyoma-lm DATASET=fineweb` then trains on. Parquet/arrow deps live ONLY
//! here, isolated from the model crates.
//!
//! Usage:
//!   vyoma-data <in.parquet> [out.txt] [max_mb]   extract text (default cap 50 MB)
//!   vyoma-data --make-test [path.parquet]        write a tiny synthetic parquet (self-test)
//!
//! FineWeb shards (no full 75 TB download — one shard at a time):
//!   curl -skL -o shard.parquet \
//!     https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/main/sample/10BT/000_00000.parquet

use anyhow::{anyhow, Result};
use arrow::array::{Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use std::fs::File;
use std::sync::Arc;

/// Write a small synthetic parquet with a `text` column, so the reader can be
/// round-trip tested without downloading FineWeb.
fn make_test(path: &str) -> Result<()> {
    let docs = vec![
        "Vyomarudra is a redefined-architecture language model built corner by corner.",
        "The Ontological Store keeps knowledge on disk and retrieves it at inference.",
        "A sparse mixture of experts gives big-model quality at small active cost.",
        "Bounded self-evolution writes new facts to the store without forgetting.",
    ];
    let arr = StringArray::from(docs);
    let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(arr)])?;
    let mut w = ArrowWriter::try_new(File::create(path)?, schema, None)?;
    w.write(&batch)?;
    w.close()?;
    Ok(())
}

/// Extract the `text` column of a parquet file into a UTF-8 text file, capped at
/// `max_bytes` so a shard never blows up local disk.
fn extract(inp: &str, outp: &str, max_bytes: usize) -> Result<usize> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(inp)?)?.build()?;
    let mut out = String::new();
    let mut docs = 0usize;
    'outer: for batch in reader {
        let batch: RecordBatch = batch?;
        let col = batch
            .column_by_name("text")
            .ok_or_else(|| anyhow!("no `text` column in {inp}"))?;
        let arr = col
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow!("`text` column is not UTF-8 string"))?;
        for i in 0..arr.len() {
            if arr.is_null(i) {
                continue;
            }
            out.push_str(arr.value(i));
            out.push('\n');
            docs += 1;
            if out.len() >= max_bytes {
                break 'outer;
            }
        }
    }
    std::fs::write(outp, out.as_bytes())?;
    Ok(docs)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--make-test") {
        let p = args.get(2).cloned().unwrap_or_else(|| "test.parquet".into());
        make_test(&p)?;
        println!("[data] wrote synthetic parquet -> {p}");
        return Ok(());
    }
    let inp = args
        .get(1)
        .ok_or_else(|| anyhow!("usage: vyoma-data <in.parquet> [out.txt] [max_mb]  |  --make-test [path]"))?;
    let default_out = format!("{}/../vyoma-lm/data_cache/fineweb.txt", env!("CARGO_MANIFEST_DIR"));
    let outp = args.get(2).cloned().unwrap_or(default_out);
    let max_mb: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(50);
    if let Some(dir) = std::path::Path::new(&outp).parent() {
        std::fs::create_dir_all(dir).ok();
    }
    let docs = extract(inp, &outp, max_mb * 1024 * 1024)?;
    let bytes = std::fs::metadata(&outp)?.len();
    println!("[data] {inp} -> {outp}: {docs} docs, {:.1} MB (cap {max_mb} MB)", bytes as f64 / 1048576.0);
    println!("[data] next: DATASET=fineweb TOKENIZER=bpe MODE=moe ./target/release/vyoma-lm  (OUR model trains on it)");
    Ok(())
}
