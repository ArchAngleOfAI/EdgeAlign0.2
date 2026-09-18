# SHORT_MEMORY.md — Short-Term / In-Flight Memory

Running log of current tasks and conversation state. Read this first on
restart to pick up where things left off. Prune/archive into `MEMORY.md`
once something becomes durable project knowledge instead of active state.

## Current Status (2026-09-18)

- Just initialized this project: `git init`, branch renamed to `main`,
  created `AGENT.md`, `MEMORY.md`, `SHORT_MEMORY.md`.
- **Blocked / waiting on user:** what is this project actually for? Need
  the goal, scope, and any existing code/design to populate `MEMORY.md`
  and start real work.
- Nothing committed yet. First commit is pending until the user confirms
  what to include (and asks for a commit — per policy I don't commit
  unless asked).

## Open Questions

1. What is the project's goal/purpose?
2. What language/stack (if any is already decided)?
3. Any existing code elsewhere to import, or starting from scratch?

## Recent Actions Log

- 2026-09-18: Scaffolded repo + the three memory/personality files per
  user's standing instructions in `AGENT.md`.
- 2026-09-18: Built `smoke_train.py` — 10-step training smoke test on
  Qwen3-0.6B (config from `/data/models/huggingface/qwen3-0.6b`),
  seq_len=1024, batch_size=1, lr=5e-5 flat, fp32, fixed random synthetic
  batch (seed 0) reused across all steps. `--pretrained` flag toggles
  between random-init (`from_config`) and loading the real checkpoint
  weights (`from_pretrained`) while keeping everything else identical.
- 2026-09-18: Ran both variants on GPU 7 (idle at the time; GPU 4 OOM'd
  — only ~5MB free — so switched per user instruction to run on an idle
  GPU instead). Random-init: loss 12.12 → 9.38 over 10 steps.
  Pretrained: loss 13.60 → 10.33 over 10 steps. Both decrease
  monotonically-ish, confirming the train loop works in both cases.
  Pretrained starting *higher* than random-init is expected here (input
  is pure random-noise tokens, not real text — see MEMORY.md), not a
  bug. Committed as `58aa973` (scaffolding + first script version);
  `--pretrained` flag change not yet committed.

## Open Questions (updated)

- Confirm whether future runs should use real text data instead of
  synthetic random tokens — the pretrained-vs-random-init loss
  comparison is only really meaningful on real text.
