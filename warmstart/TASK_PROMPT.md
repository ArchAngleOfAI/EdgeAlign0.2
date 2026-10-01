# Warm-start pretraining: original task prompt

The task prompt the user gave on 2026-09-30 (20:14 UTC), copied verbatim below.
It is the spec for everything in `warmstart/`.

Changes made since then, at the user's direction (details in SHORT_MEMORY.md / MEMORY.md):
- Data size: the 90,000-example target was not reachable (RLVR-IFeval too small for 30%).
  The user chose to keep 50/30/20, shrink the total (20,754 train examples) and train
  multi-epoch (fresh permutation per epoch, no repeat within an epoch).
- RLVR decontamination: 13-gram check on the task text only (message minus constraint).
- Language exclusion: "All Lowercase" + "All Uppercase" constraint types.
- Batching (2026-10-01): batched gist rollout; run with `--micro-batch 4` (4 x accum 4,
  global batch still 16). Extra check (e) for the padded batched rollout.
- KD loss (2026-10-01): token-weighted mean over all response tokens of the optimizer
  step (all 16 examples), replacing "divide each micro-batch loss by 8".

---

Set up and run the WARM-START pretraining for the soft-prompt (gist) generator,
using a "fill in the removed part" objective on datasets whose inputs have two
parts by nature. Read AGENT.md, MEMORY.md, SHORT_MEMORY.md, report.md,
stability_report.md, RUN_NOTES_3layer.md, gist_health_report.md and
train_3layer.py first, and follow the repo conventions in AGENT.md (including
updating the memory files). Do NOT modify any existing file except the memory
files. Put all new code in a new folder warmstart/. (This replaces an earlier
WildChat-based plan; do not use WildChat.)

WHY
gist_health_report.md showed the previous objective (receiver must match the
teacher on the same prompt) has a trivial optimum: a prefix with no effect. The
generator collapsed to one constant, prompt-independent hard-token prefix
(softmax entropy ~0) acting as an attention sink at position 0. The new
objective removes real information from the receiver, so the gist must carry it.

OBJECTIVE (per example)
- Teacher (frozen): full user message + response.
- Generator: reads ONLY the removed part (raw token ids, no chat template) and
  produces L=16 gist vectors via the existing autoregressive soft rollout.
- Receiver (frozen): user message with the removed part replaced by the gist,
  + the same response.
- Loss: KL(teacher || receiver), temperature 1, ONLY on the response tokens
  (assistant content + its closing <|im_end|>), aligned by content (teacher and
  receiver positions differ by span_len - 16). Token-weighted mean over the
  batch.
- The response is fixed text generated offline (step 1). Teacher and receiver
  each do ONE teacher-forced forward pass over the whole sequence; only the
  16-step gist rollout is sequential.

MODEL
- INSTRUCT Qwen3-0.6B (not -Base). Confirm which checkpoint the existing code
  loads and report it; switch if needed.
- Non-thinking mode everywhere: apply_chat_template(..., enable_thinking=False).
  Keep whatever the template inserts for non-thinking mode identical in teacher
  and receiver.
- Generator = build_softprompt_generator with the last 3 decoder layers +
  LM_head + Embedding2 trainable, everything else frozen, full backprop through
  the rollout (no detaching). Set config.tie_word_embeddings = False; assert
  lm_head, embed_tokens and embedding2 have distinct data_ptrs.
- Reuse generate_softprompt and the KD-loss math from distill_softprompt.py.
  Write a NEW receiver function for mid-sequence insertion:
  inputs_embeds = cat(prefix_embeds, gist, suffix_embeds).

STEP 1 - DATA PREPARATION (warmstart/prepare_data.py, run once, offline)

Layout rule for ALL sources: build the user message yourself so the removed
part comes LAST in the user message. The gist then sits at the end of the user
turn, the same slot the later language/format directive will use.

Source A - Super-NaturalInstructions (target ~50% of training examples)
- Get it from GitHub allenai/natural-instructions (tasks/*.json and
  splits/default/train_tasks.txt, test_tasks.txt); verify the layout and field
  names, and report what you used.
- Train pool: tasks in the default TRAIN split with English input AND output.
  The default TEST split tasks (119 English tasks) must never be used for
  training; they are used only for held-out set B below.
- Definition only (zero-shot): do not include positive/negative examples.
- User message: "Input:\n{instance input}\n\nTask:\n{definition}".
  Removed part = the definition text (the "Task:" label stays visible).
- SPAN CAP: first measure the definition length in Qwen tokens over the train
  pool (report p50/p75/p90/p95/p99 and the number of tasks). Set
  CAP = the 95th percentile rounded up to a multiple of 16, but never above
  256. Drop tasks whose definition exceeds CAP and report how many.
- At most 150 instances per task (random, seed 0), so large tasks don't
  dominate.

Source B - allenai/RLVR-IFeval (target ~30%)
- Use the user message from `messages` and the `constraint` field.
- Exclude every example whose constraint_type concerns response language, so
  the later language experiment tests something new. Report which types were
  excluded and how many examples.
- Locate the constraint text inside the user message. Rebuild the message as
  (message with the constraint removed) + "\n\n" + constraint. Removed part =
  the constraint. If the constraint text cannot be located reliably, drop the
  example and count it.
- Decontaminate: drop any prompt sharing a 13-gram with google/IFEval prompts,
  and with IFBench test prompts if you can obtain them. Report counts.

Source C - SQuAD v1.1 (rajpurkar/squad) (target ~20%)
- Training examples from the TRAIN split only.
- User message: "Context:\n{passage}\n\nQuestion: {question}". Removed part =
  the question text (the "Question:" label stays visible).

All sources:
- Removed-part length must be 8..CAP tokens; the rest of the user message must
  be at least 8 tokens; total teacher sequence (template + message + response)
  at most 1024 tokens. Drop and count violations.
- No-gap examples: mark 15% of examples (per source) as no-gap. For these the
  receiver keeps the removed part AND gets the gist after it; the generator
  still reads the removed part. They teach the gist to add nothing when
  nothing is missing.
- Responses: generate with the frozen instruct Qwen3-0.6B from the FULL user
  message, non-thinking, temperature 0.7, top_p 0.8, top_k 20,
  max_new_tokens 256 (vLLM if available, else batched HF generate). Drop
  degenerate responses: empty, no end-of-turn token within the limit, or heavy
  repetition. Report the rule and counts.
- KL_nogist FILTER (gap examples only): compute KL(teacher || receiver with
  the removed part deleted and NOTHING inserted) on the response tokens. Within
  each source, drop the gap examples in the bottom 30% of KL_nogist (the removed
  part barely mattered). Report the KL_nogist distribution per source before
  and after filtering. No-gap examples are not filtered.
- Size: generate enough to end up with at least 90,000 training examples after
  all filtering, mixed ~50/30/20 across A/B/C (report the final counts).

Held-out sets (fixed seed, never used in training):
- Held-out A (256 examples, in distribution): same sources and proportions,
  ~15% no-gap. SNI: train-split tasks but instances not used in training.
  RLVR-IFeval: prompts not used in training. SQuAD: from the VALIDATION split.
- Held-out B (128 examples, unseen tasks): SNI default TEST split tasks
  (English), definition removed, all gap examples, same layout, cap and
  filters. This measures whether the gist works on task types never seen.

Save JSONL with: id, source, task (SNI task name where relevant), user_full,
user_kept, removed_part, gap (bool), response, token counts, KL_nogist. Keep
data files out of git (.gitignore); commit a 30-example sample (10 per source)
and a stats file with every count and distribution reported above.

STEP 2 - SEQUENCE CONSTRUCTION (get this exactly right)
Tokenize the pieces SEPARATELY and build both sequences from the SAME token ids:
- teacher  = prefix_ids + removed_ids + suffix_ids + response_ids
- receiver = prefix_ids + [16 gist vectors] + suffix_ids + response_ids
  (no-gap: prefix_ids + removed_ids + [gist] + suffix_ids + response_ids)
prefix_ids = chat template up to and including user_kept (with its label);
suffix_ids = the rest of the rendered template after the user content (end of
user turn, assistant header, non-thinking block). Right padding + attention
masks; micro-batch 2.
Before training, run these checks and STOP if any fails:
(a) response token ids identical in teacher and receiver for every example;
(b) ALIGNMENT CHECK: for examples whose removed part is exactly 16 tokens,
    replace the gist with the real embeddings of the removed part; the
    response-token KL must be ~0 (report it);
(c) with NO gist on gap examples, KL must be clearly > 0 (report the mean);
(d) after one backward pass, embedding2, lm_head and the 3 trainable layers
    all have nonzero gradients, and the last soft token's grad_fn is not None.

STEP 3 - TRAINING (warmstart/train_warmstart.py)
- L = 16. fp32. Seed 0.
- AdamW: peak LR 1e-4, betas (0.9, 0.95), eps 1e-6, same weight decay as
  train_3layer.py (report it).
- LR: linear warmup 0 -> 1e-4 over 100 steps, then cosine decay to 1e-5 at the
  final step.
- Global batch 16 = micro-batch 2 x accumulation 8; divide each micro-batch
  loss by 8; clip_grad_norm_(trainable, 1.0) once per optimizer step after
  accumulation; log the pre-clip norm.
- 5,000 optimizer steps, sampling without replacement (no example repeated).
- Log mean rollout softmax entropy every step. Support --entropy-coef (default
  0.0; keep 0 for this run). If mean held-out rollout entropy falls below 0.05
  nats at any eval, log a clear WARNING (do not stop).
- Stop and report on any NaN/inf.

STEP 4 - EVALUATION
Schedule: step 0, every 10 steps up to step 300, then every 100 steps.
Compute once at start: KL_nogist per gap example in both held-out sets.
On held-out A at every eval, overall AND per source AND per removed-length
bucket (<=64, 65-128, 129-CAP tokens):
1. KL_gist for gap and no-gap examples separately.
2. RECOVERY on gap examples: mean of (KL_nogist - KL_gist) / KL_nogist, and
   (mean KL_nogist - mean KL_gist) / mean KL_nogist.
3. Swap test: KL with another example's gist (fixed derangement, seed 0);
   report swapped minus own on gap examples.
4. Mean pairwise cosine between gists of different examples.
5. Mean rollout entropy and distinct top-1 tokens across the 16 steps.
Every 100 steps: recovery and swap test on held-out B (unseen tasks).
Every 500 steps: attention fraction on gist positions 1..16 and 2..16 from an
eager-attention copy of the receiver, on 32 held-out A gap examples, averaged
over heads, layers and response query positions.
Logs: logs/warmstart_train.jsonl (per step), logs/warmstart_eval.jsonl (per
eval, with per-example values saved).

PLOTS (regenerate and overwrite atomically at every eval)
- warmstart_loss_curve.png: per-step train loss (thin), 50-step moving average,
  held-out A KL_gist (gap), and a horizontal line for mean KL_nogist.
- warmstart_recovery.png: held-out A recovery (overall and per source),
  held-out B recovery (when available), swap difference and cross-gist cosine
  vs step; rollout entropy on a second axis.
Vertical dashed line at step 100 on both.

CHECKPOINTS (keep out of git)
latest (every eval), best by held-out A mean recovery on gap examples (NOT by
lowest loss), and permanent every 1,000 steps. Support --resume.

RUNNING
- Healthy GPU (GPU 0 had a hardware error before; GPU 5 worked).
- 20-step smoke run first; report seconds per step and estimated total time.
  Then launch the full run with nohup, logging to logs/warmstart.out.

GIT
- After data prep + the Step 2 checks: commit code, data sample and stats; push.
- After step 300: commit logs, both plots and RUN_NOTES_warmstart.md (exact
  settings, model checkpoint, CAP and how it was chosen, data counts per source,
  check results (a)-(d), timing, step-0 and step-300 recovery); push.
- When the run finishes: write warmstart_report.md with:
  * a plain-language summary;
  * final and best recovery (overall, per source, per length bucket, and on
    held-out B), the best step, and swap / cosine / entropy / attention numbers
    at the best step vs step 0;
  * whether these criteria are met: recovery clearly above 0; swap test shows
    gist-specific information; gists differ across examples (cosine well below
    1); rollout entropy not collapsed; no-gap KL_gist stays small; recovery on
    held-out B (unseen tasks) above 0;
  * anything suspicious. Do not overclaim; if evidence is mixed, say so.
  Commit everything except data and checkpoints, push to main on
  ArchAngleOfAI/EdgeAlign0.2, confirm the push succeeded, and update the memory
  files.
