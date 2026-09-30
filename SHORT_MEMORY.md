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

- 2026-09-18: Added `--data fineweb-edu` to `smoke_train.py` — real
  text from `HuggingFaceFW/fineweb-edu` (`sample-10BT`, first shard
  only), read via direct `fsspec`+`pyarrow` row-group access (NOT
  `datasets` streaming — that downloaded the whole ~2GB shard first;
  see MEMORY.md for why). Fresh seq_len=1024 chunk per step. Ran
  random-init variant on GPU 4 (idle at check time): loss 12.07 → 10.82
  over 10 steps. Not yet committed. Haven't run `--pretrained` on real
  data yet — didn't assume the user wants that repeated, will ask/wait.

## DEAD job (started 2026-09-18 21:05, found dead 2026-09-22)

- `distill_softprompt.py --steps 100000 --batch-size 4 --checkpoint-every 10000`
  was running on GPU 6 (PID 2669321), log at `logs/distill_100k.log`
  (gitignored, not in repo). On 2026-09-22 found PID no longer exists;
  log frozen at step 17262/100000 since 2026-09-21 20:23 UTC (~22h before
  discovery) with no error visible in the tail -- cause of death not
  investigated. Only checkpoint on disk is `embedding2_step10000.pt`
  (steps 10000->17262 lost). That checkpoint uses the OLD save schema
  (`embedding2_state_dict` only) and is INCOMPATIBLE with the current
  `--resume` code (now also expects `lm_head_state_dict` and
  `last_layer_state_dict` -- see below), so it can't be resumed from
  as-is. Not yet decided whether to archive/delete it or investigate
  the crash; ask the user before touching it.

## KD alignment confirmed correct (2026-09-22)

- User asked whether the teacher-stream vs. student-stream position
  offset (student has 100 soft-prefix tokens before the real prompt) was
  handled correctly in the KD loss. Traced it through: causal-LM logits
  at position p always predict position p+1, so `receiver_forward`'s
  slice `logits_full[:, 100:, :]` lines up index-for-index with
  `teacher_logits` by construction -- no shift bug. Full derivation is
  in the conversation, not duplicated in code comments.

## Soft-prompt-length sweep (2026-09-22)

- Built `sweep_softprompt_length.py`: no training, pretrained weights
  only, measures initial KD loss for soft-prompt length 1..200
  (averaged over 5 FineWeb-Edu prompts). Rolls out the full length-200
  softprompt ONCE per prompt (causal, so prefixes are reusable) rather
  than 200 separate rollouts.
- **Counterintuitive finding**: KD loss does NOT decrease with longer
  soft-prompts pre-training. It rises sharply from L=1 (~0.073) to a
  peak around L=40-50 (~0.54), then plateaus/slowly declines to ~0.46
  by L=200 -- never recovering to anywhere near the L=1 level. Plot:
  `kd_loss_vs_softprompt_length.png`. Likely cause (not confirmed):
  embedding2 starts as a clone of the tied embedding table, so a
  1-token soft-prefix barely perturbs the receiver's input relative to
  the teacher, but the untrained autoregressive rollout compounds
  drift over more steps, plus growing RoPE-position offset for the real
  prompt tokens. Whether training fixes this is untested.

## Trainable set expanded (2026-09-22, per user request)

- `qwen_dual_embedding.build_softprompt_generator` now also trains the
  student encoder's LM_head and its transformer's LAST decoder layer,
  in addition to `embedding2`. LM_head was TIED to `embed_tokens`
  (Embedding1, tie_word_embeddings=True, confirmed same data_ptr) in
  this checkpoint -- untied it first (own cloned Parameter) per user's
  explicit choice, so Embedding1 (real-token input embedding) stays
  frozen exactly as the 2026-09-18 correction intended. Verified via a
  real backward pass: embed_tokens and layer 26 get no grad, lm_head +
  layer 27 + embedding2 all do. New trainable count: 326,895,872 (was
  155,582,464).
- `distill_softprompt.py`'s optimizer/checkpoint/resume all updated to
  cover the 3 trainable pieces (was just embedding2).

## Gradient accumulation added, NO gradient checkpointing (2026-09-22)

- New `--grad-accum-steps` flag; `--steps` counts OPTIMIZER steps, each
  running `grad_accum_steps` micro-batches (backward scaled by 1/N)
  before one `optimizer.step()`. Global batch size = batch-size *
  grad-accum-steps.
- Gradient checkpointing was explicitly NOT implemented after
  measuring it wouldn't help: `model.gradient_checkpointing_enable()`
  forces `use_cache=False` in HF, which would silently break
  `generate_softprompt`'s KV-cache-based rollout (confirmed
  empirically -- past_key_values comes back None, each step would lose
  all prior context). Checkpointing only the receiver (which has no
  cache dependency) was measured to save just ~0.5GB / 1.6% of peak
  memory (30.44GB -> 29.96GB at batch=4) since the encoder rollout
  dominates peak memory (~86%) and isn't checkpointable this way. User
  confirmed: skip GC, grad-accum-only is fine (receiver has zero
  trainable params anyway, so accumulation only ever touches the
  encoder side by construction).
- Added explicit `torch.OutOfMemoryError` handling per micro-batch:
  logs step/micro-batch index + allocated/reserved memory before
  re-raising, so an OOM is unambiguous in the log instead of a bare
  traceback.

## Active job: bs64/ga16 test (started 2026-09-22, in progress)

- `distill_softprompt.py --steps 63 --batch-size 4 --grad-accum-steps 16`
  on GPU 7 (only sufficiently idle GPU at launch time; GPU 4 got grabbed
  by another user's job mid-session), with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (mitigates the
  fragmentation-driven OOM warnings seen during benchmarking -- peaked
  at 33.07GB allocated on a near-idle GPU with only ~4.4GB used by
  others, i.e. margin was already thin). 63 steps x 16 micro-batches =
  1008 micro-batches, rounded up from the user's "1000 step test"
  (clarified to mean ~1000 micro-batches, not 1000 optimizer steps --
  1000 optimizer steps would have been ~3.4 days). Log:
  `logs/global_bs64_ga16_test.log` (gitignored). Benchmarked
  ~18.5s/micro-batch -> ETA ~5.2 hours from launch. Check
  `ps aux | grep distill_softprompt` and tail the log if picking this up
  fresh; still need to plot the kd_loss curve to PNG once it finishes
  and report final OOM status.

## Open Questions (updated)

- Whether to also run `--pretrained --data fineweb-edu` for a
  pretrained-vs-random-init comparison on real text (more meaningful
  than the earlier synthetic-noise comparison).
- Whether the counterintuitive soft-prompt-length sweep result also
  holds AFTER training (untested).
- What to do with the dead 100k job's orphaned, schema-incompatible
  checkpoint (`checkpoints/embedding2_step10000.pt`).

## Flat-KD-loss diagnostics (2026-09-29, done)

- User asked to test Hypothesis A (broken/vanishing grads) vs B (good init +
  too little capacity). 7 diagnostics in `diagnostics/`, raw logs in
  `diagnostics/results/`, write-up in `report.md`. A ruled out; B partially.
- Key facts: the flat 17k run was EMBEDDING2-ONLY (pre-2026-09-22 code);
  its step-10000 ckpt is 9% better than init on paired batches (hidden by
  ~0.2 per-batch noise). 3 trainable layers: held-out 0.708->0.438 in 57
  steps vs 1 layer 0.708->0.615. BPTT supplies 97% of embedding2 grad and
  amplifies early-token grads; LR 1e-3 is unstable.

## Stability vs soft-prompt length (2026-09-29, done)

- `diagnostics/stability_vs_length.py` (+ `run_stability.sh`): 30 steps on
  one fixed batch per run, L in {8,16} x LR {5e-5,1e-3} x clip {off, 1.0}.
  Write-up in `stability_report.md`, logs in `diagnostics/results/stability_*.log`.
- The 4 L=100 runs were CANCELLED by the user; test 4 run B (20 steps) is the
  only L=100 reference. Not yet run: L=100 with clipping.
- Findings: LR 1e-3 jump shrinks with length (+0.56 L=100, +0.18 L=16,
  +0.06 L=8); clip max_norm=1.0 did NOT remove the L=16 jump; clipping did
  not slow training-batch learning at 5e-5. At LR 1e-3 the mean held-out
  gain (~-0.2) hides 4/8 batches getting worse (held-out losses converge to
  a narrow ~0.16-0.32 band).
- GPU 0 threw a CUDA hardware error ("invalid access of peer GPU memory ...
  hardware error") on 2026-09-29 -- avoid it; GPU 5 was used.

## 3-layer long run (2026-09-29 -> stopped 2026-09-30, done)

- `train_3layer.py`: Embedding2 + LM_head + last 3 layers, L=16, AdamW peak
  LR 1e-4 (100-step warmup, then constant), global batch 16 (2 x accum 8),
  clip 1.0. Details in `RUN_NOTES_3layer.md`; plot with `replot_3layer.py`.
- STOPPED by the user at step 3029/10000 (SIGTERM, not a crash). Best
  held-out 0.0105 @ step 2600 (from 0.3930 @ 0); last eval 0.2066 @ 3000.
- Loss curve shows INSTABILITY after step ~2000: held-out spike at 2000
  (0.084), then a regression from ~2650 to ~0.21 with no recovery; clipping
  active on 52% of steps after 2600; median pre-clip grad norm rose to ~1.0.
  Earlier regression at step 157 recovered. Cause not isolated (LR constant).
- Checkpoints (~4.3 GB each) in /data/a84460786/testfolder_checkpoints/
  train_3layer/: best.pt (step 2600), step_01000/02000/03000.pt, latest.pt
  (= step 3000). `--resume` continues from step 3000.
- Environment: EdgeAlign (and its .venv) was deleted on 2026-09-29. New venv:
  /data/a84460786/venvs/testfolder (torch 2.14.0+cu130, transformers 5.17.0).
  The diagnostics/*.sh runners still default to the old venv; pass
  PY=/data/a84460786/venvs/testfolder/bin/python.
- GPUs: 0 is faulty (CUDA hardware error); GPU 2 worked (~20 s/step, shared
  with another user's vLLM worker); GPU 5 is often taken by other users.
- Held-out set for this run comes from FineWeb-Edu shard 001_00000
  (training reads only 000_00000). The user decided to keep FineWeb-Edu
  rather than /data/r50058044/.../wiki-18.jsonl (which is a tar-wrapped
  FlashRAG wiki dump, not plain JSONL).

## Gist health check (2026-09-30, done)

- `diagnostics_gist/` (t1_swap ... t8_steer, `_gist_common.py`, `run_all.sh`);
  results in `diagnostics_gist/results/`; write-up in `gist_health_report.md`.
  Checkpoints tested: step0 (fresh init, seed 0), step1000 (`step_01000.pt`),
  best (`best.pt` = step 2600, held-out 0.0105). Ran on GPU 5.
- Verdict: COLLAPSED BUT ALIVE. From step 1000 the generator is one-hot at
  every rollout step and emits the same sequence for every prompt
  (' it', 'gle', 'gle', ' it' x13): cross-prompt cos 0.996, swap effect
  ~4e-8. Not dead: naive prefixes are 14-16x worse (text16 0.165,
  rand_tok 0.148-0.196, pad16 0.394, zeros 3.29); the best gist is 2.5-77x
  more noise-sensitive than real text; directly steerable to log p("{") ~0;
  generator steering from best +9.6 nats vs +0.9 from step0.
- Only sign of invisibility: attention to gist positions 2-16 falls
  0.125 -> 0.022 -> 0.0035 (text16 0.026); the receiver uses the gist mostly
  via position 0 (attention sink).
- Gradient at an exactly-zero prefix is NaN (steering from zeros undefined).
- Next steps suggested: monitor swap / cross-prompt cos / rollout entropy
  during training (KD loss alone missed the collapse); test whether one
  learned constant prefix matches 0.0105; test whether gist positions 2-16
  matter at all.
