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

## Changelog (durable, high-level only)

- 2026-09-18: Repo initialized (`git init`, branch renamed to `main`).
  `AGENT.md`, `MEMORY.md`, `SHORT_MEMORY.md` scaffolded per user's
  standing instructions.
