# Running Vyomarudra on Kaggle (free T4 GPU)

The full pipeline, copy-paste. Everything is our own Rust — no Python ML, no
external model. The teacher never appears here at all; FineWeb is raw web text.

**Session setup:** Notebook → Settings → **Accelerator: GPU T4 x2**, **Internet: On**.

> Kaggle sessions are ephemeral. `/kaggle/working` survives cell-to-cell but a
> *session restart* wipes the Rust toolchain and often the repo. Re-run Cells 1–2.
> **Download anything you want to keep** (Cell 9) before the session ends.

---

## 1. Toolchain
```python
import os
os.environ['PATH'] = os.path.expanduser('~/.cargo/bin') + ':' + os.environ['PATH']
!cargo --version || (curl -sSf https://sh.rustup.rs | sh -s -- -y -q)
```

## 2. Repo
```python
%cd /kaggle/working
!git clone --depth 1 https://github.com/Akashnkumar1/Vyomarudra.git || (cd Vyomarudra && git pull)
%cd /kaggle/working/Vyomarudra
```

## 3. Build (CUDA)
The CUDA linker needs the driver lib on its search path — this is the one Kaggle
quirk. Set `LIBRARY_PATH` **before** building, in the same session:
```python
import os
os.environ['LIBRARY_PATH'] = '/usr/local/nvidia/lib64:' + os.environ.get('LIBRARY_PATH', '')
!cargo build --release -p vyoma-data -p vyoma-tokenizer -p vyoma-embed
!cargo build --release -p vyoma-lm --features cuda
```
CPU-only fallback if CUDA misbehaves: drop `--features cuda` (slower, always works).

## 4. Data — one FineWeb shard (not the 75 TB)
```python
!curl -skL -o shard.parquet https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/main/sample/10BT/014_00000.parquet
!./target/release/vyoma-data shard.parquet crates/vyoma-lm/data_cache/fineweb.txt 200
```
`014` is the small shard (~549 MB). Others are ~2.1 GB each (`000`…`013`).
The last arg caps extracted text in MB.

## 5. Tokenizer — our BPE on real web text
```python
!DATASET=fineweb VOCAB=4096 SAMPLE_MB=30 ./target/release/vyoma-tokenizer
```
Merges train on a capped sample, then apply to the whole corpus. Expect
~3.4 bytes/token and `round-trip=OK`.

## 6. Train the LM — and KEEP it
```python
!SAVE=/kaggle/working/vyoma_fineweb.safetensors \
 DATASET=fineweb TOKENIZER=bpe MODE=moe ONLY=moe \
 STEPS=30000 DM=384 DFF=512 MOE_E=4 SEQ=64 \
 ./target/release/vyoma-lm
```
`ONLY=moe` skips the two dense baselines so all GPU time goes into the model you
keep. Drop `ONLY=moe` to reproduce the three-way comparison instead.
Two GPUs, in parallel (separate processes — we have no multi-GPU tensor code):
```python
%%bash
CUDA_VISIBLE_DEVICES=0 ONLY=small,moe SAVE=/kaggle/working/vyoma_fineweb.safetensors \
  DATASET=fineweb TOKENIZER=bpe MODE=moe STEPS=30000 DM=384 DFF=512 ./target/release/vyoma-lm > g0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 ONLY=big \
  DATASET=fineweb TOKENIZER=bpe MODE=moe STEPS=30000 DM=384 DFF=512 ./target/release/vyoma-lm > g1.log 2>&1 &
wait; cat g0.log g1.log
```

## 7. Read what it writes
```python
!MODE=generate LOAD=/kaggle/working/vyoma_fineweb.safetensors \
 PROMPT="The future of artificial intelligence is" NEW=300 TEMP=0.8 TOPK=40 \
 ./target/release/vyoma-lm
```

## 8. Store + retriever + grounding head (Pillar 4 / 3b) on real text
```python
!DATASET=fineweb MAX_MB=20 MODE=store STEPS=1500 BATCH=128 DMODEL=96 DK=256 \
 NEG=tinyshakespeare.txt NNEG=32 ./target/release/vyoma-embed
!DATASET=fineweb MAX_MB=20 NEG=tinyshakespeare.txt NCAL=400 MODE=head ./target/release/vyoma-embed
!DATASET=fineweb MAX_MB=20 NEG=tinyshakespeare.txt NCAL=200 MODE=gate ./target/release/vyoma-embed
```
Then the whole system in one command:
```python
!MODE=rag LOAD=/kaggle/working/vyoma_fineweb.safetensors \
 RET=crates/vyoma-embed/data_cache/retriever.safetensors \
 STORE=crates/vyoma-embed/data_cache/ontological_store.vyst \
 HEAD=crates/vyoma-embed/data_cache/grounding_head.safetensors \
 PROMPT="..." NEW=200 ./target/release/vyoma-lm
```

## 9. Keep the artifacts
```python
!cp crates/vyoma-embed/data_cache/{retriever,grounding_head}.safetensors /kaggle/working/
!cp crates/vyoma-embed/data_cache/ontological_store.vyst /kaggle/working/
!cp crates/vyoma-lm/data_cache/bpe_merges.txt /kaggle/working/fineweb_merges_4096.txt
!ls -lh /kaggle/working/*.safetensors /kaggle/working/*.vyst /kaggle/working/*.txt
```
Download from the Kaggle file browser (right panel → `/kaggle/working`). The LM
checkpoint + merges + store + retriever + head is a complete, portable Vyomarudra.

---

## Honest expectations

**Timing.** CUDA build ~10–20 min first time (kernel compilation). 30 K steps at
dm=384 is roughly 1–3 h on a T4. Want a quick artifact instead? `STEPS=8000`.

**Text quality.** At ~5 M params over a partial pass of 210 M tokens, expect
fluent-*looking* English with weak coherence. That is scale, not architecture —
compare against the logged BPB numbers, not against a chat model.

**The grounding caveat that matters.** Our 86.6% grounding-head result used a
*Shakespeare* store with *distilled text* as negatives — a clean **domain**
boundary. A **FineWeb store has no such boundary**: FineWeb is open web text, so
"out of domain" is nearly undefined and using Shakespeare as the negative only
teaches "not archaic verse". The honest FineWeb question is **fact-level**: *is
this specific passage in my store?* — for which the right negatives are FineWeb
passages **held out of the store**, not a foreign corpus. That is a harder and more
realistic problem, and the Shakespeare number should **not** be expected to
transfer. Treat step 8 on FineWeb as a new experiment, not a confirmation.

**Reproducing the logged FineWeb LM result** (`docs/PROGRESS.md`, 2026-07-26):
char tokenizer, `DM=384 DFF=512 MOE_E=4 STEPS=6000`, three-way comparison →
dense-small 2.797 / MoE 2.677 / dense-big 2.801.
