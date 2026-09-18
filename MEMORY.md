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

## Changelog (durable, high-level only)

- 2026-09-18: Repo initialized (`git init`, branch renamed to `main`).
  `AGENT.md`, `MEMORY.md`, `SHORT_MEMORY.md` scaffolded per user's
  standing instructions.
- 2026-09-18: `smoke_train.py` added and run (random-init and
  pretrained variants) on GPU 7. See "Training smoke test" above.
