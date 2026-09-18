# MEMORY.md — Long-Term Project Memory

Durable knowledge about this project: goal, structure, conventions, and
major decisions. Update this file when something becomes true "for the
life of the project," not for in-flight task state (that belongs in
`SHORT_MEMORY.md`).

## Project Goal

_Pending — not yet defined. Waiting on the user to describe what this
project is for._

## Environment

- **Model:** `/data/models/huggingface/qwen3-0.6b` — Qwen3ForCausalLM,
  0.6B (hidden_size 1024, 28 layers, 16 attn heads / 8 KV heads, vocab
  151936, bf16, tied embeddings). Saved with `transformers==4.51.0`
  per its `config.json`.
- **Python env:** `/home/a84460786/EdgeAlign/.venv` (Python 3.12.3,
  reused from the separate EdgeAlign project — not local to
  testfolder). Key packages: `torch==2.14.0`, `transformers==5.17.0`,
  `accelerate`, `datasets`, `liger_kernel`, CUDA 13 wheels.
  Note: this is EdgeAlign's *current* venv, whose `transformers` version
  (5.17.0) EdgeAlign's own SHORT_MEMORY.md flags as a suspect in an
  active bf16 bug investigation against the 32B checkpoint — worth
  keeping in mind if similar symptoms show up here with the 0.6B model.
- **Cluster:** 3x NVIDIA A100-PCIE-40GB (per `nvidia-smi`); shared with
  other users' running jobs — check `nvidia-smi` and scope
  `CUDA_VISIBLE_DEVICES` before use, don't assume exclusive access.

## Repository Structure

- Git repo initialized 2026-09-18, default branch `main`.
- No commits yet.
- No source files yet.

## Conventions & Decisions

_None recorded yet._

## Key Terminology / Domain Notes

_None recorded yet._

## Training smoke test (`smoke_train.py`)

- Purpose: verify the forward/backward/optimizer loop works for the
  Qwen3-0.6B architecture on this cluster/env, and compare random-init
  vs. loading real pretrained weights, all else held equal.
- Fixed setup: seq_len=1024, batch_size=1, lr=5e-5 flat (AdamW, no
  schedule), 10 steps, fp32, seed=0, same fixed synthetic random-token
  batch (`torch.randint` over full vocab) reused every step in both
  runs — not real text.
- `--pretrained` flag switches `from_config` (random init) →
  `from_pretrained` (real checkpoint weights) with nothing else
  changed, for apples-to-apples comparison.
- **Result caveat:** the pretrained run's initial loss (13.60) is
  *higher* than the random-init run's (12.12). This is expected, not a
  bug: the data is pure random noise tokens, so a random-init model's
  near-uniform output distribution sits close to the uniform baseline
  (`ln(151936) ≈ 11.9`), while a pretrained model's confident language
  priors make it more "surprised" by pure noise. Any comparison of
  these two modes should switch to real text data to be meaningful.
- GPU choice: GPU 4 (initially requested) was OOM at the time (~37GB/40GB
  already used by another job) — ran on GPU 7 instead (idle). GPU
  availability on this shared cluster changes run to run; always check
  `nvidia-smi` first.

### Real data: FineWeb-Edu (`--data fineweb-edu`)

- Dataset: `HuggingFaceFW/fineweb-edu`, `sample-10BT` subset (its
  smallest sample split, ~28.5GB/14 parquet shards). We only read the
  first shard (`sample/10BT/000_00000.parquet`).
- **Important implementation note:** `datasets.load_dataset(...,
  streaming=True)` was tried first and tested to download the *entire*
  ~2GB parquet shard before yielding even one row (confirmed: a plain
  `curl` range request on the same file hit ~10MB/s, i.e. ~200s to pull
  the whole shard, matching the observed hang). Switched to opening the
  parquet file directly via `fsspec` + `pyarrow.parquet.ParquetFile` and
  reading only row group 0 (1000 docs) via HTTP range requests — same
  underlying data, ~1.2s instead of minutes. This is the loading method
  in `fineweb_edu_batches()` in `smoke_train.py`; don't switch back to
  naive `datasets` streaming for small-sample use cases like this one.
- Documents are tokenized with the checkpoint's own tokenizer,
  concatenated with an EOS separator, and sliced into fixed `seq_len`
  chunks. Unlike the synthetic mode, **a fresh chunk is used every
  step** (real dataloader, not a reused batch) — so this loss curve is
  a genuine (tiny) training curve, not a single-batch-overfit signal.
- First run (random-init, GPU 4): loss 12.07 → 10.82 over 10 steps,
  noisier than the synthetic runs (expected, since data changes every
  step) — e.g. a spike to 12.49 at step 8.

## Dual-embedding architectures (`qwen_dual_embedding.py`)

Two distinct architectures, both built on Qwen3-0.6B's tied
Embedding1/LM_head weight (`W1`), distinguished by where the trainable
`Embedding2` sits:

1. `build_frozen_qwen_with_trainable_embedding2` — Embedding2(trainable,
   **input**) → Transformer(frozen) → LM_head(frozen) → Embedding1(frozen,
   **output** projection via `W1`). Used by `embedding_roundtrip_forward.py`
   (the original standalone forward-pass smoke test).
2. `build_softprompt_generator` — Embedding1(frozen, **input**) →
   Transformer(frozen) → LM_head(frozen) → Embedding2(trainable,
   **output** projection). Used by the distillation encoder
   (`distill_softprompt.py`). Corrected 2026-09-18 from an earlier,
   wrong version that had Embedding2 at the input — user caught this:
   real tokens should go through the well-calibrated frozen embedding,
   only the *output* projection should be the thing being learned.

Both cases: the vocab→hidden output projection uses
**`softmax(logits) @ weight`**, never raw `logits @ weight` — see "soft
token collapse bug" below for why.

## Soft-token collapse bug (found & fixed 2026-09-18)

When the roundtrip's output projection was `logits @ W` (raw,
unnormalized logits, unbounded scale) fed back autoregressively as the
next step's `inputs_embeds`, the rollout blew up ~24x in magnitude after
one step, then **hit a fixed point**: steps 2-99 of a 100-step rollout
were numerically identical (cosine similarity 1.0, relative diff ~1e-7).
Only 2 of the intended 100 "soft tokens" carried any distinct
information — the rest were copies. Root cause: raw logits as weights
for combining `W`'s rows produces vectors far outside the scale the
frozen transformer's layers actually operate at, and the recurrence
converges to an attractor almost immediately.

**Fix:** apply `softmax(logits)` before the projection (`probs @ W`) —
makes it a genuine convex "soft lookup" over `W`'s rows, naturally
bounded to roughly normal embedding scale. Confirmed post-fix: soft
token norms sane (0.35-0.95, vs. raw version's ~2.4e6), and tokens are
actually distinct (cosine(token50, token99) ≈ 0.013, near-orthogonal).
Applied in both `qwen_dual_embedding.py`-based architectures.
**Any future addition of a similar vocab→hidden projection in this
project should default to `softmax(logits) @ W`, not raw logits.**

## Soft-prompt distillation setup (`distill_softprompt.py`)

- Teacher: plain frozen Qwen3-0.6B instance, sees the real prompt,
  produces target logits.
- Student encoder: `build_softprompt_generator` (see above), rolls out
  autoregressively for `NUM_SOFT_TOKENS=100` steps with KV-cache, each
  step's `softmax(logits) @ Embedding2` output feeding back as the next
  step's input. No discretization at any point. Output: `(100, 1024)`
  softprompt.
- Student receiver: a *separate* frozen Qwen3-0.6B instance. Input is
  conceptually `[100 MASK tokens] + [real prompt]`; implemented as
  `torch.cat([softprompt, receiver.embed_tokens(prompt_ids)])` rather
  than embedding placeholders then overwriting in-place (in-place slice
  assignment into a non-grad tensor doesn't reliably propagate
  gradients back to `Embedding2`; concatenation does).
- Loss: `KL(teacher_probs || student_probs)` over vocab, temperature
  1.0, at the receiver's output positions aligned to the real prompt
  (i.e. skipping the 100 soft-prefix positions).
- Only `Embedding2` (155.6M params) is trainable; both frozen Qwen
  instances plus the encoder's own frozen backbone (596M params each)
  get no gradients. Verified directly (not just inferred): after the
  fix, `embedding2.weight.grad` is dense (near 100% nonzero — expected,
  since it's now a matrix-multiply projection, not a sparse row lookup),
  finite, no NaN/Inf.
- Position IDs: no manual override needed — HF's default
  `position_ids = 0..seq_len-1` for a fresh `inputs_embeds` call already
  gives the real prompt tokens positions `100..100+N-1` in the receiver
  (continuing from the soft prefix), matching the user's explicit choice.
- Defaults not specified by the user, flagged: `PROMPT_LEN=128` (vs. the
  1024 used in plain-LM smoke tests — kept smaller here to keep the
  100-step rollout manageable), `NUM_STEPS=10`, `LR=5e-5` AdamW,
  `KD_TEMPERATURE=1.0`.
- Post-fix 10-step run: KD loss in a sane 0.22-0.74 range (vs. ~10-12
  pre-fix, when the receiver was getting a garbage-scale softprompt).

## Changelog (durable, high-level only)

- 2026-09-18: Repo initialized (`git init`, branch renamed to `main`).
  `AGENT.md`, `MEMORY.md`, `SHORT_MEMORY.md` scaffolded per user's
  standing instructions.
- 2026-09-18: `smoke_train.py` added and run (random-init and
  pretrained variants) on GPU 7. See "Training smoke test" above.
- 2026-09-18: Built and ran the dual-embedding roundtrip smoke test,
  froze it into `qwen_dual_embedding.py`, built the soft-prompt
  distillation setup (`distill_softprompt.py`), corrected the
  encoder's architecture (Embedding2 moved from input to output
  projection) per user feedback, found and fixed the soft-token
  collapse bug (raw logits → softmax(logits) in the projection). See
  sections above.
