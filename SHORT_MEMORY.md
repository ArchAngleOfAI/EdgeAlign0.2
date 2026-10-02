# SHORT_MEMORY.md — Short-Term / In-Flight Memory

Running log of current tasks and conversation state. Read this first on
restart to pick up where things left off. Prune/archive into `MEMORY.md`
once something becomes durable project knowledge instead of active state.

## >>> ICAE + RL TASK (branch `icae-rl`, started 2026-10-02) -- read this first <<<

Work happens ONLY on branch `icae-rl` (never commit/push main). Spec: `icae_rl/TASK_PROMPT.md`
(verbatim). Running notes: `icae_rl/NOTES.md`. Big files: /data/a84460786/edgealign_icae_rl/.
Venv: /data/a84460786/venvs/icae_rl (torch 2.14, transformers 4.57.6, peft 0.21.2, bfcl_eval editable).
- Stage 0 DONE (2026-10-02): ICAE cloned, weights downloaded, base = Mistral-7B-Instruct-v0.2
  (weights identical to the March-2024 revision), sanity check passed (FT QA correct, pretrained
  reconstruction exact on prose, lossy on a tool trace). Committed + pushed on icae-rl.
- Stage 1 (2026-10-02): FULL on TRAIN (strict BFCL parser) = 1/120 (0.8%), per-turn pass 7.1%
  -> gate (a) (>=15%) FAILS. Only 14.5% of calls parse (30% escape `\_`, 19% add prose after the call).
  Saved: /data/a84460786/edgealign_icae_rl/rollouts/stage1/full_train.jsonl; icae_rl/stage1/.
- USER DECISION (2026-10-02 ~15:20 UTC): STOP the remaining Stage 1 runs (icae_ft/TRAIN was at 77/120,
  1 success, 7.8% turn pass, only in logs/stage1.out) and run a LENIENT-PARSER diagnostic instead:
  FULL on TRAIN with `--parser lenient` (bfcl_env.lenient_decode: un-escape `\_`, first parsable
  bracketed call list; NOT BFCL's parser). Offline re-parse of the strict responses: 14.5% -> 55.6%.
  Purpose: separate "bad at BFCL format" from "bad at the task".
- Lenient run: two OOM crashes (batch 6, batch 4; logs/stage1_lenient_oom_*.out). Fixed in
  icae_model.generate (DynamicCache init copied the padded KV -> per-layer handover; chunked prefill)
  + OOM fallback (one prompt at a time) + per-episode .partial.jsonl with resume. RUNNING since
  ~16:40 UTC on GPU 3, batch 4, log icae_rl/logs/stage1_lenient.out. Rerun the same command to resume:
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=3 python run_stage1.py
  --batch-size 4 --runs full/ft --splits train --parser lenient`
- NEXT: compare lenient vs strict FULL; report to user; then write icae_rl_feasibility_report.md
  (gate failed) and push. The user decides whether any more Stage 1 runs happen.

## >>> CURRENT STATUS (updated 2026-10-01 ~21:00 UTC) -- read this first <<<

**Nothing is running.** The warm-start full run (5,000 planned steps) was STOPPED by the user at
step 1353 (2026-10-01 20:37 UTC). No background processes (the gpu_hold scripts have exited too).
- It ran on GPUs 6 + 4 from 17:25 UTC (2-GPU torchrun, code 8b57a57, OMP_NUM_THREADS=14).
- RESULT: the gist COLLAPSED by step 90 (LR 9e-5, in warmup) to `' prompt'` x16 for every input
  (entropy 0, cosine 1.000, swap minus own 0.000) and never recovered. After that, the Embedding2
  row of `' prompt'` trained as one constant soft prompt repeated 16 times: held-out A recovery
  0.393 (ratio of means 0.498, gap KL 3.71 -> 1.86) and B 0.331 at step 1300, vs 0.136 / 0.102
  for the input-specific step-0 gist. Loss and recovery looked like progress; only swap minus
  own, cosine and entropy showed the collapse.
- Full write-up: `warmstart_collapse_report.md` (pipeline, collapse, causes, options, full
  metric/loss tables for steps 0-1353).
- Checkpoints: /data/a84460786/testfolder_checkpoints/warmstart/ (step_01000.pt, latest.pt =
  best.pt = step 1300), ~4.3 GB each. "best" is by recovery, i.e. the best constant prefix.
- Next steps are the user's call, none started (report section 6): measure/train against a
  placeholder or constant-prefix baseline instead of deletion; anti-saturation (--entropy-coef,
  frozen LM_head, logit cap, noise); per-example gradient clipping; alerts on swap minus own ~ 0
  and cosine ~ 1.

Everything is committed and pushed to `origin/main` (latest commit 39b00ee, "Warm-start run
stopped at step 1353: final logs, plots, report, memory"; working tree clean after it)
(https://github.com/ArchAngleOfAI/EdgeAlign0.2). Pushing works without a prompt: a GitHub
token is stored in `~/.git-credentials`.

**Latest state of the research (newest first):**
0. **Warm-start pretraining: full run stopped at step 1353; gist collapsed to `' prompt'` x16** (see `warmstart_collapse_report.md`).
1. **Gist health check, done:** `gist_health_report.md`, `diagnostics_gist/`.
   The best 3-layer generator (step 2600) is COLLAPSED BUT ALIVE. It emits the
   same 16-vector gist for every prompt (' it', 'gle', 'gle', ' it' x13), yet
   still beats naive prefixes 14-16x and is steerable.
2. **3-layer long run, stopped by the user at step 3029/10000:**
   `train_3layer.py`, `RUN_NOTES_3layer.md`. Best held-out 0.0105 @ step 2600;
   unstable after step ~2000 (regressed to ~0.21 by step 3000).
3. **Stability vs soft-prompt length, done:** `stability_report.md`.
4. **Flat-KD-loss diagnosis, done:** `report.md`.

**Next steps: the user's call** (see warmstart_collapse_report.md section 6). Older candidates the
reports raise, none started (warm-start evals already log swap test,
cross-prompt cos, rollout entropy and gist attention):
- monitor gist collapse during training (swap test, cross-prompt cos, rollout
  entropy)
- test a single learned constant prefix against 0.0105 -- effectively answered for the
  warm-start objective: the collapsed run trained one (`' prompt'` row of Embedding2) and it
  reached 0.39 recovery on held-out A (not tested for the 3-layer objective)
- test whether gist positions 2-16 matter
- a lower LR / LR decay to address the late instability

**Where things are:**
- Python venv: `/data/a84460786/venvs/testfolder/bin/python`. The EdgeAlign
  venv was deleted on 2026-09-29. `diagnostics/*.sh` still default to the old
  path; pass `PY=...`.
- 3-layer checkpoints (~4.3 GB each, outside git):
  `/data/a84460786/testfolder_checkpoints/train_3layer/`. `best.pt` = step
  2600; also `step_01000.pt`, `step_02000.pt`, `step_03000.pt`, and
  `latest.pt` (= step 3000).
- Old checkpoint `checkpoints/embedding2_step10000.pt` (1.8 GB, from the
  flat 17k run, Embedding2 only). report.md test 5 uses it. Kept; the user was
  told about it and has not asked to delete it.
- GPUs: 0 is faulty (CUDA hardware error). 5, 2, 6, 7 and 4 have worked; 6 + 4 worked well
  for the 2-GPU run. Ownership by other users changes within hours, so check `nvidia-smi` plus
  the process owner right before launching.
- Warm-start checkpoints (~4.3 GB each, outside git):
  /data/a84460786/testfolder_checkpoints/warmstart/ (step_01000.pt, latest.pt = best.pt = step
  1300; best = best recovery = best constant prefix). Smoke-run checkpoints:
  /data/a84460786/testfolder_checkpoints/warmstart_smoke/.
- logs/gpu_hold/ (gitignored): hug.py (reserves almost all free GPU memory and sleeps) and
  watch_and_launch.sh (checked GPU 4 every 10 min, launched the run when it freed). Used at the
  user's request on 2026-10-01; both processes have exited.
- Warm-start data (outside git): /data/a84460786/warmstart_data/ (train.jsonl,
  heldout_a.jsonl, heldout_b.jsonl, data_stats.json; raw/ = SNI clone @55a3656 + HF
  parquets). A 30-example sample + stats are committed in warmstart/.

**Resolved older items** (the sections below are historical log):
- The dead 17k-step job is superseded. Its checkpoint was used in
  report.md; no decision is pending.
- The bs64/ga16 test was interrupted at step 26/63. Its plot is committed as
  `kd_loss_bs64_ga16_test.png`.
- The project-goal questions from 2026-09-18 are answered by the work
  itself: soft-prompt ("gist") distillation research on Qwen3-0.6B.

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

## [HISTORICAL] DEAD job (started 2026-09-18 21:05, found dead 2026-09-22) -- superseded

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

## [HISTORICAL] bs64/ga16 test (started 2026-09-22) -- ended: interrupted at step 26/63, plot committed

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

## [HISTORICAL] Open Questions (2026-09-22; see CURRENT STATUS at top)

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

## Warm-start gist pretraining (2026-09-30 -> 2026-10-01, full run collapsed, stopped at step 1353)

- Spec (user-pasted): "fill in the removed part" -- the generator reads only the
  removed span of a user message and rolls out L=16 gist vectors; the receiver sees
  the message with the span replaced by the gist (no-gap examples: span kept, gist
  after it) + the response; loss KL(teacher || receiver), T=1, response tokens only.
  Sources SNI / RLVR-IFeval / SQuAD at 50/30/20, 5,000 steps, global batch 16,
  peak LR 1e-4 (100-step warmup, cosine to 1e-5), clip 1.0, last 3 layers +
  LM_head + Embedding2 trainable.
- User decisions (2026-09-30): keep 50/30/20 but shrink the total (RLVR too small)
  and train multi-epoch (fresh permutation per epoch); RLVR decontamination by
  13-gram on task text only. Also excluded: All Lowercase + All Uppercase instruction types --
  the agent's judgment call (only types saying "in English"), NOT explicitly decided by the user.
- Data (commit 1dee1c3): 20,754 train, held-out A 256, held-out B 128 (unseen SNI
  tasks), CAP 208 (p95 203.25). Generation ran as 3 shards on 2026-09-30; shard 2
  OOM'd on GPU 2 (another user's process) and was re-run as 4 sub-shards.
- Smoke test (2026-10-01, original serial code, GPU 7): 20 steps, exit 0. Last
  night's smoke had died silently in the step-0 eval (session ended, no error).
  Step-0 eval: held-out A recovery 0.1363 (KL_gist 3.19 vs no-gist 3.71), B 0.104,
  swap-own +0.69, cross-cos 0.195, entropy 2.98, gist attention 7%. Timing: 38.4
  s/step, 420 s per full eval -> ~62 h for the full run. Grad norms 1e2-1e5 at
  startup, clipping on at every step.
- Speed fix (2026-10-01, user asked): GPU was ~22% busy because rollout_gists ran
  generate_softprompt one example at a time. New `rollout_gists` in common.py is
  the same math on a LEFT-padded batch with attention mask + position ids; the
  old one is kept as `rollout_gists_serial`. New `--micro-batch N` (accum = 16/N;
  resume refuses a different micro-batch). Result at --micro-batch 4: 12.2
  s/step, 84 s per eval (3.2x / 5x faster). Step-0 eval matches the serial run.
  Memory: ~1.5 GB per example in the batched rollout (serial ~0.25) -- micro-batch
  8 untested, may not fit.
- Check (e) added (step2_checks.json): (e1) swapping the pad token gives
  bit-identical gists, entropy, top-1 and grads (under deterministic algorithms);
  (e2) first soft token vs an unpadded batch of copies within 4e-6. Why not
  "batched == serial": see MEMORY.md "Numerics of the gist rollout".
- Loss definition changed (user chose "option 3", 2026-10-01): KD is now the
  token-weighted mean over ALL response tokens of the step's 16 examples (was: token
  mean per micro-batch, averaged over micro-batches, which made the objective
  depend on the micro-batch size). Verified: step loss 1.2393 / 1.2396 / 1.2402 at
  micro 1 / 2 / 4. Speed unaffected. Logged train_kd is this step-wide mean, so it
  is NOT comparable to the 2026-10-01 serial smoke numbers.
- Smoke logs (gitignored): logs/warmstart_smoke.out (serial, 20 steps),
  logs/warmstart_smoke_mb4.out (batched, 6 steps, before the loss change).
- 2-GPU data parallel (2026-10-01, user asked): torchrun, each GPU takes 8 of the 16 examples,
  grads summed before clipping; rank 0 does checks/evals/logs/checkpoints. NCCL P2P hangs on this
  machine -> NCCL_P2P_DISABLE=1 (set in the trainer). Smoke on GPUs 6+4: 7.05 s/step, step-1 loss
  and grad norm identical to one GPU, weights identical across GPUs at the end.
- Spec review vs TASK_PROMPT.md (2026-10-01): undiscussed deviations were reported to the user --
  20.7k examples (~3.9 passes) instead of the ~34k estimated when they chose to shrink; language
  exclusion; batched rollout departs from "reuse generate_softprompt" / right padding; final config
  never had a 20-step smoke (user then said skip it); instruct checkpoint confirmed but was never
  reported before; responses generated in bf16.
- Full run (2026-10-01): launched 17:25 UTC on GPUs 6 + 4 (2-GPU torchrun, --micro-batch 4, code
  8b57a57), fully detached (`setsid nohup ... < /dev/null &`); GPU 4 was taken at first, so a
  watcher (logs/gpu_hold/) launched it when it freed. Logs now tracked in git:
  logs/warmstart_train.jsonl, logs/warmstart_eval.jsonl, logs/warmstart.out; plots
  warmstart_loss_curve.png, warmstart_recovery.png.
- Collapse: by step 90 (LR 9e-5, warmup) the gist was `' prompt'` x16 for every input (entropy 0,
  cosine 1.000, swap minus own 0.000); never recovered. Afterwards only the Embedding2 row of
  `' prompt'` trained (constant soft prompt): held-out A recovery 0.393 / B 0.331 at step 1300.
  Weights moved ~0.3-0.4% per tensor; the `' prompt'` row was the most-changed row in LM_head
  (4.3%) and Embedding2 (7.9%); next LM_head rows '>>\n', ' instruction', 'Prompt', ' instructions'.
- Causes (report section 5): main = recovery measured against DELETING the span (malformed
  prompt), so a generic placeholder earns large "recovery". Supporting: huge clipped gradients
  dominated by one example; softmax saturation makes one-hot absorbing (top-1/top-2 gap
  1.2 -> 14 -> 23, max logit 18 -> 73); LR rise as the timing trigger (untested).
- Stopped by the user at step 1353 (20:37 UTC). Report: warmstart_collapse_report.md. Final
  logs/plots/report/memory committed and pushed as 39b00ee.
