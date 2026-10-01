# Warm-start gist pretraining: pipeline and gist collapse

Snapshot at **step 140 of 5,000** (2026-10-01, ~18:00 UTC); the appendix tables run to **step 210**. The full run is still running on
GPUs 6 + 4 (started 17:25 UTC, code at commit `8b57a57`); logs and plots up to step 131 are in
commit `aa08e71`. The spec is `warmstart/TASK_PROMPT.md`.

## 1. Summary

- The new "fill in the removed part" objective was implemented as specified, apart from the
  changes listed in section 2.9. Every pre-training check passed. The trainer ran correctly on one
  GPU and on two GPUs, with identical step-1 loss and gradient norm.
- **The gist collapsed by step 90 (learning rate 9e-5, still in warmup).** From then on the generator
  emits the token `" prompt"` at all 16 gist positions for every input. Rollout entropy is 0,
  gists of different examples are identical (cosine 1.000), and swapping in another
  example's gist changes nothing (swap minus own = 0.000). The gist carries no information
  about the removed part. It is the same failure mode as the 3-layer run
  (`gist_health_report.md`), even though the objective was changed specifically to rule it out.
- **The loss hides it.** Held-out KL keeps improving after the collapse, and by step 140 the
  constant `" prompt"` x16 prefix "recovers" more of the gap than the informative step-0 gist did
  (0.145 vs 0.136). Under this objective a content-free placeholder is a strong solution. That is
  the main reason I think the collapse happened (section 5).
- **Update at step 210:** the gap has widened. With the same constant prefix (swap minus own
  still exactly 0), held-out A recovery is 0.241 (ratio of means 0.275) and held-out B, the
  unseen tasks, is at 0.200. That is far above the input-specific step-0 gist (0.136 and 0.102).
  Every metric except swap minus own, cosine and entropy now looks like progress. Full tables are
  in the appendix.

## 2. The training pipeline as built

### 2.1 Objective and models
Per example (all models are Qwen3-0.6B instruct, `/data/models/huggingface/qwen3-0.6b`, fp32):
- **Generator (trainable):** `build_softprompt_generator`. It reads only the removed part (raw
  token ids, no template) and rolls out L = 16 soft tokens. Each step computes
  `softmax(LM_head(h)) @ Embedding2` and feeds the result back as the next input (KV cache).
  Trainable parts: Embedding2, the untied LM_head and the last 3 decoder layers (25-27), in total
  358,357,760 parameters. Full backprop runs through all 16 steps.
- **Teacher (frozen):** the full user message plus the response, as token ids.
- **Receiver (frozen):** the user message with the removed part replaced by the 16 gist vectors,
  plus the same response.
- **Loss:** KL(teacher || receiver), temperature 1, on the response tokens only (assistant
  content plus the closing `<|im_end|>`).

### 2.2 Data (`warmstart/prepare_data.py`, `warmstart/data_stats.json`)
| source | removed part | user message layout | train examples |
|---|---|---|---|
| Super-NaturalInstructions (train-split tasks) | task definition | `Input:\n{input}\n\nTask:\n{definition}` | 10,377 |
| RLVR-IFeval | constraint sentence | `{message minus constraint}\n\n{constraint}` | 6,226 |
| SQuAD v1.1 train | question | `Context:\n{passage}\n\nQuestion: {question}` | 4,151 |

- **Total:** 20,754 training examples, exactly 50/30/20 across the sources. 15.0% are no-gap
  (17,640 gap / 3,114 no-gap).
- **Removed-part cap:** CAP = 208 tokens, the 95th percentile of SNI definition lengths (203.25)
  rounded up to a multiple of 16.
- **Filters:**
  - removed part 8..CAP tokens; kept part at least 8 tokens; teacher sequence at most 1,024 tokens;
  - degenerate responses dropped;
  - KL_nogist filter: within each source, the bottom 30% of gap examples by KL_nogist are dropped.
- **Responses:** generated offline by the same model in non-thinking mode (temperature 0.7,
  top_p 0.8, top_k 20, at most 256 tokens), with HF generate in bf16. Median length is 16 tokens,
  and 10% are 2 tokens or fewer.
- **Held-out sets:**
  - A: 256 examples (217 gap, 39 no-gap), same sources and proportions.
  - B: 128 gap examples from 98 SNI test-split tasks never used in training.

### 2.3 Input formats (`warmstart/common.py`)
- **Pieces:** each is tokenized separately.
  - `prefix` = `<|im_start|>user\n` + the kept text with its label;
  - `removed` = the raw removed part;
  - `suffix` = `<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n` (9 tokens);
  - `response` ends in `<|im_end|>`.
  - There is no system prompt.
- **Layouts:**
  ```
  generator  [PAD..][ removed ]                                (left-padded, raw ids)
  teacher    [ prefix ][ removed ][ suffix ][ response ][PAD..]  (right-padded ids)
  receiver   [ prefix ][ g1..g16 ][ suffix ][ response ][PAD..]  (gap)
             [ prefix ][ removed ][ g1..g16 ][ suffix ][ response ][PAD..]  (no-gap)
  ```
- **Padding:** every pad is `<|endoftext|>` and is masked out. In the generator, position ids
  restart at 0 for each row's first real token.
- **Alignment:** response logits are cut out per example from (start of response − 1). The
  response ids are identical on both sides, so the two sides line up by content. The teacher and
  receiver differ in length by (removed length − 16).

### 2.4 Loss weighting
The KD loss for each optimizer step is the **token-weighted mean over all response tokens of the
step's 16 examples**. Each micro-batch contributes its summed per-token KL divided by the step's
total token count. This makes the objective independent of the micro-batch size. The original
prompt said "divide each micro-batch loss by 8". The user chose this version on 2026-10-01.

### 2.5 Optimization
- **Optimizer:** AdamW with betas (0.9, 0.95), eps 1e-6 and weight decay 0.01 (the same as
  `train_3layer.py`).
- **LR schedule:** linear warmup from 0 to 1e-4 over 100 steps, then cosine decay to 1e-5 at step
  5,000.
- **Batch and clipping:** global batch 16; `clip_grad_norm_(1.0)` once per step on the summed
  gradient.
- **Steps:** 5,000 steps, about 3.9 passes over the data, with a fresh permutation each pass
  (seed 0 + pass number). fp32, seed 0.

### 2.6 Implementation and speed
- **Batched gist rollout:** the generator runs a whole micro-batch at once, left-padded, with an
  attention mask and explicit position ids. The original one-at-a-time version is kept as
  `rollout_gists_serial`. This made steps 3.2x faster and evaluations 5x faster.
- **Two GPUs, data parallel** (`torchrun --nproc_per_node=2`):
  - each GPU takes 8 of the 16 examples (micro-batch 4 x accumulation 2);
  - gradients are summed across GPUs before clipping, so the update is the same as on one GPU;
  - rank 0 alone runs the checks, evaluations, logging and checkpoints;
  - weights are checked to be identical across GPUs at every evaluation.
- **P2P fix:** GPU-to-GPU P2P hangs on this machine, so the trainer sets `NCCL_P2P_DISABLE=1`.
- **Speed:** about 6.5 s per step plus about 50-85 s per evaluation, so about 11.5 h for the whole
  run.

### 2.7 Checks before training (all passed; `warmstart/step2_checks.json`)
- **(a)** Response ids are identical in teacher and receiver for all 21,138 examples.
- **(b)** Putting the real embeddings of a 16-token removed part in place of the gist gives KL ~0
  (max 1.5e-8). This is an exact alignment test.
- **(c)** With no gist, the mean KL on gap examples is 5.54, so the removed part matters.
- **(d)** Embedding2, LM_head and layers 25-27 all receive non-zero gradients; nothing frozen does.
- **(e)** Batched-rollout padding is exact. Swapping the pad token gives bit-identical gists,
  entropy and gradients under deterministic algorithms. The first soft token matches an unpadded
  batch to within 4e-6.

### 2.8 Evaluation and logging
- **Schedule:** step 0, every 10 steps up to step 300, then every 100.
- **Held-out A, every evaluation,** overall, by source and by removed-length bucket:
  - KL with the gist on gap and on no-gap examples;
  - recovery = (KL_nogist − KL_gist) / KL_nogist, as a mean of ratios and as a ratio of means;
  - the swap test (KL with a deranged gist minus KL with the example's own gist);
  - cross-example gist cosine;
  - rollout entropy;
  - distinct top-1 tokens.
- **Held-out B:** every 100 steps.
- **Attention to gist positions:** every 500 steps.
- **Warning:** the trainer warns when held-out entropy falls below 0.05 nats.
- **Outputs:** `logs/warmstart_train.jsonl`, `logs/warmstart_eval.jsonl`,
  `warmstart_loss_curve.png`, `warmstart_recovery.png`. Checkpoints are in
  `/data/a84460786/testfolder_checkpoints/warmstart/`.

### 2.9 Differences from `TASK_PROMPT.md`
| prompt | as built | discussed with the user |
|---|---|---|
| at least 90,000 examples | 20,754 (RLVR too small; 25% of its responses ran past 256 tokens) | shrink chosen 2026-09-30 at an estimate of ~34k; 20.7k reported, not discussed |
| no example repeated | 5,000 steps, multi-pass | yes |
| 13-gram decontamination on the prompt | on the task text only | yes |
| exclude response-language constraint types | All Lowercase + All Uppercase excluded (judgment call) | no |
| micro-batch 2 x 8, loss / 8 | micro-batch 4, step-wide token-weighted loss, 2 GPUs | yes |
| reuse `generate_softprompt`, right padding | batched re-implementation, left padding in the generator | partly |
| 20-step smoke run of the final setup | 20-step smoke ran the old code; final setup had 6- and 2-step smokes | user chose to skip |
| report the instruct checkpoint | it is the instruct model; first reported 2026-10-01 | — |

## 3. What happened in the run

**Held-out A** (recovery: higher is better; at step 0 the untrained gist already recovers 14%)

| step | LR | recovery | ratio of means | KL gap (no gist 3.707) | KL no-gap | SNI | RLVR | SQuAD | swap − own | cosine | entropy | distinct top-1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.136 | 0.140 | 3.189 | 0.360 | 0.110 | 0.258 | 0.019 | +0.694 | 0.195 | 2.977 | 12.1 |
| 10 | 1e-5 | 0.092 | 0.137 | 3.199 | 0.351 | 0.121 | 0.152 | −0.075 | +0.683 | 0.185 | 2.949 | 12.1 |
| 20 | 2e-5 | −0.030 | 0.011 | 3.664 | 0.238 | −0.032 | 0.141 | −0.285 | +0.214 | 0.237 | 3.239 | 12.3 |
| 40 | 4e-5 | −0.054 | −0.002 | 3.713 | 0.303 | −0.042 | 0.116 | −0.341 | +0.197 | 0.319 | 2.721 | 11.1 |
| 60 | 6e-5 | −0.024 | 0.027 | 3.607 | 0.288 | 0.023 | 0.093 | −0.322 | +0.246 | 0.296 | 2.466 | 11.0 |
| 70 | 7e-5 | 0.027 | 0.036 | 3.572 | 0.171 | −0.011 | 0.098 | 0.015 | +0.151 | 0.434 | 1.259 | 8.1 |
| 80 | 8e-5 | 0.023 | 0.059 | 3.488 | 0.135 | 0.026 | 0.098 | −0.097 | +0.103 | 0.449 | 0.476 | 5.4 |
| **90** | 9e-5 | 0.065 | 0.076 | 3.425 | 0.045 | 0.056 | 0.110 | 0.017 | **−0.002** | **0.999** | **0.005** | **1.0** |
| 100 | 1e-4 | 0.079 | 0.090 | 3.373 | 0.044 | 0.069 | 0.114 | 0.052 | 0.000 | 1.000 | 0.000 | 1.0 |
| 120 | 1e-4 | 0.112 | 0.122 | 3.256 | 0.045 | 0.094 | 0.129 | 0.130 | 0.000 | 1.000 | 0.000 | 1.0 |
| 140 | 1e-4 | **0.145** | 0.154 | 3.137 | 0.043 | 0.119 | 0.143 | 0.211 | 0.000 | 1.000 | 0.000 | 1.0 |

- **Held-out B (unseen tasks):** recovery 0.102 at step 0 and 0.050 at step 100. Swap minus own
  went from +0.579 to −0.000.
- **Attention to the gist** (step 0 only so far): 7.4% on positions 1-16, 7.0% on positions 2-16.

**Training** (means over 20 steps)

| steps | train KD | rollout entropy | median pre-clip grad norm | steps clipped |
|---|---|---|---|---|
| 1-20 | 1.153 | 3.05 | 1,530 | 20/20 |
| 21-40 | 1.252 | 3.06 | 628 | 20/20 |
| 41-60 | 1.392 | 2.66 | 257 | 20/20 |
| 61-80 | 1.249 | 1.44 | 200 | 20/20 |
| 81-100 | 1.207 | 0.12 | 2.8 | 17/20 |
| 101-120 | 1.167 | 0.00 | 1.9 | 15/20 |
| 121-140 | 1.164 | 0.00 | 1.7 | 18/20 |

The run went through three phases:
1. **Steps 10-40: the starting benefit was destroyed.** Held-out KL on gap examples rose from 3.19
   to the no-gist level of 3.71, while the training loss did not fall.
2. **Steps 60-90: entropy collapsed** as the learning rate rose through 6e-5 to 9e-5.
3. **From step 90 on: the constant prefix was tuned.** Held-out gap KL improves steadily with a
   gist that is identical for every input.

## 4. What the collapse is

I compared the step-0 checkpoint (`best.pt`) with the step-130 one (`latest.pt`) on the CPU, using
12 held-out A gap examples.

| | step 0 | step 130 |
|---|---|---|
| top-1 token at the 16 gist positions (one example) | `' Now' ',' ' the' ' pair' ' is' ' "' 'The' ' cat' ...` | `' prompt'` x16 |
| same for two other examples | input-dependent continuations of the removed text | `' prompt'` x16 |
| mean rollout entropy | 2.69 | 0.0000 |
| mean top-1 probability | 0.40 | 1.0000 |
| mean max logit | 18.4 | **53.6** |
| mean gap between top-1 and top-2 logits | 1.24 | **14.35** |
| cross-example cosine of gists | 0.158 | 1.000 |
| gist vector norm per position | 0.51 | 0.90 (the Embedding2 row of `' prompt'`) |

The weights barely moved: the relative change is 0.28% for Embedding2, 0.36% for LM_head and
0.34-0.41% for layers 25-27. The change is concentrated on one token:
- **LM_head:** the `' prompt'` row changed by 4.3%, more than any other of the 151,936 rows (the
  median row changed 0.02%). The next most-changed rows are `'>>\n'`, `' instruction'`,
  `'Prompt'` and `' instructions'`.
- **Embedding2:** the `' prompt'` row changed by 7.9%, again the largest (median 0.01%).

So the generator learned to write the placeholder word "prompt" sixteen times, whatever the input.

## 5. Why it happened

The evidence for each point is stated with it. Points 5.1-5.3 are supported by the measurements
above. Point 5.4 is a plausible but untested hypothesis.

### 5.1 A content-free placeholder scores well under this objective (main cause; strong evidence)
- **The baseline makes a placeholder look like information.** Recovery is measured against
  KL_nogist, where the removed part is simply deleted. That leaves the receiver with a malformed
  prompt (`Task:\n<|im_end|>`, `Question: <|im_end|>`, a trailing blank line). A gap of 3.7 nats
  therefore mixes two things: the information that is missing, and the fact that the prompt looks
  broken. A generic placeholder ("a prompt/instruction goes here") fixes the second without
  carrying any of the first.
- **Measured:** the constant `' prompt'` x16 reaches recovery 0.145 at step 140, more than the
  input-specific step-0 gist (0.136), and it is still improving. On no-gap examples it cuts KL from
  0.36 to 0.04, because a fixed placeholder that the receiver learns to ignore is close to ideal
  when nothing is missing. All three sources remove something that is an instruction, task or
  question, so one placeholder fits all of them.
- **Why it wins the optimization:** the part of the gradient that pushes towards a better
  placeholder points the same way for every example in every step, so it adds up. The part that
  carries example-specific content points in a different direction for each example and partly
  cancels within a batch of 16 and across steps. The weight changes match this: the only weights
  that moved meaningfully are a handful of "prompt / instruction" rows.

### 5.2 Huge, clipped gradients made the content signal noisy (supporting; measured)
- **Every update had a fixed size and a direction set by one example.** Pre-clip gradient norms
  started at about 1,500 (1e2-1e5 per example), so clipping to 1.0 was active on every step until
  the collapse. Earlier measurements (`MEMORY.md`, "Numerics of the gist rollout") showed a single
  example supplying about 97% of a step's gradient norm. Those dominant gradients are also the most
  sensitive to fp32 rounding (up to 62% error against fp64 on one micro-batch).
- **This fits steps 10-40,** where held-out gap KL got worse while the training loss did not
  improve: the updates were not descending the held-out objective.

### 5.3 The softmax rollout makes the collapse irreversible (supporting; measured)
- **Saturation shuts off the gradient.** Each soft token is `softmax(logits) @ Embedding2`. Once
  the top-1 / top-2 logit gap reaches about 14 (top-1 probability 1.0000), the softmax Jacobian
  `p(1 − p)` is about 0. Almost no gradient then reaches the generator through the rollout: the
  median pre-clip gradient norm fell from about 200 (steps 61-80) to about 2 (from step 81). The
  run cannot leave this state by itself.
- **The feedback loop helps the collapse along.** The rollout feeds each token back in as the
  next input, so a slightly more likely `' prompt'` at one position makes `' prompt'` more likely
  at the next. That fits all 16 positions becoming the same token. Only 0.4% of weight change was
  needed because the max logit grew from 18 to 54 along that one direction.

### 5.4 The learning rate set the timing (hypothesis; not tested)
- **Timing:** entropy fell fastest as the learning rate passed 6e-5 to 9e-5 (2.47 at step 60, 1.26
  at step 70, 0.005 at step 90).
- **Consistent evidence:** the 3-layer run, at a constant 1e-4, had also collapsed by step 1,000,
  and `stability_report.md` found large jumps at high learning rates.
- **Not tested:** I have not checked whether a lower learning rate only delays the collapse or
  prevents it. Given 5.1, my guess is that it would only delay it.

### 5.5 What is ruled out
- **A pipeline bug in alignment, padding or batching.** Checks (a)-(e) pass. The 1-GPU and 2-GPU
  runs give identical step-1 loss and gradient norm, and the step-0 evaluation is identical across
  every run.
- **The 2-GPU setup.** The single-GPU smoke runs showed the same early drop in recovery (step 10:
  0.097 on one GPU, 0.092 on two).
- **Too little signal in the data.** The no-gist KL is large (3.7 on held-out A), and the
  untrained gist already recovers 14%, so there was information to learn and a starting point
  that used it.

## 6. Implications and options (none started)
1. **Stop the current run.** It is in an absorbing state (5.3). Continuing would only tune the
   `' prompt'` prefix, at about 11 GPU-hours on two GPUs. The checkpoints and logs are saved.
2. **Change the baseline so a placeholder earns nothing.** Measure recovery against a
   *placeholder* baseline instead of deletion: replace the removed part with 16 generic tokens
   (e.g. `' prompt'` x16 or pad tokens). Better still, train against it, so the gist is only
   rewarded for information beyond "something was here". The learned constant prefix gives a
   ready-made baseline: KL_gap 3.137 at step 140 and 2.687 at step 210.
3. **Make collapse costly or impossible:**
   - `--entropy-coef > 0` (already supported);
   - keep LM_head frozen, or bound the rollout logits (temperature or logit cap), so the softmax
     cannot saturate;
   - add noise to the rollout.
4. **Reduce the dominance of one example per step:** per-example gradient clipping (the
   `MEMORY.md` note), so the content signal of all 16 examples counts.
5. **Lower peak learning rate or longer warmup:** only a delay on its own (5.4), but useful in
   combination with the options above.
6. **Monitor what matters:** stop or alert on swap minus own ≈ 0 and cosine ≈ 1, not only on
   entropy. Recovery and loss alone looked healthy after the collapse.

## 7. Artifacts
- **Code:**
  - `warmstart/common.py`, `warmstart/prepare_data.py`, `warmstart/train_warmstart.py`;
  - spec: `warmstart/TASK_PROMPT.md`;
  - commits `1dee1c3` (data and trainer), `4dfc514` (batching and loss), `8b57a57` (2-GPU).
- **Run results:**
  - `logs/warmstart_train.jsonl`, `logs/warmstart_eval.jsonl`, `logs/warmstart.out`,
    `warmstart_loss_curve.png`, `warmstart_recovery.png` (commit `aa08e71`, step 131);
  - checkpoints `best.pt` (step 0) and `latest.pt` (most recent evaluation) in
    `/data/a84460786/testfolder_checkpoints/warmstart/`.
- **Analysis for section 4:** a CPU comparison of `best.pt` and `latest.pt` (step 130) on 12
  held-out A gap examples, plus the per-row LM_head and Embedding2 weight changes.

## Appendix: full metric and loss tables (steps 0-210, from the logs)

### A1. Held-out A, overall, every evaluation

KL values are nats on response tokens. KL no-gist (gap) is 3.707 throughout. Recovery is the mean of per-example (KL_nogist − KL_gist) / KL_nogist; ratio of means is (mean KL_nogist − mean KL_gist) / mean KL_nogist.

| step | LR | KL gist, gap | KL swap, gap | KL gist, no-gap | recovery | ratio of means | swap − own | cross-gist cosine | rollout entropy | distinct top-1 | eval s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0e+00 | 3.189 | 3.883 | 0.360 | 0.136 | 0.140 | +0.694 | 0.195 | 2.977 | 12.1 | 78 |
| 10 | 1e-05 | 3.199 | 3.883 | 0.351 | 0.092 | 0.137 | +0.683 | 0.185 | 2.949 | 12.1 | 51 |
| 20 | 2e-05 | 3.664 | 3.878 | 0.238 | -0.030 | 0.011 | +0.214 | 0.237 | 3.239 | 12.3 | 57 |
| 30 | 3e-05 | 3.705 | 3.828 | 0.260 | -0.041 | 0.000 | +0.123 | 0.314 | 3.133 | 12.2 | 51 |
| 40 | 4e-05 | 3.713 | 3.910 | 0.303 | -0.054 | -0.002 | +0.197 | 0.319 | 2.721 | 11.1 | 51 |
| 50 | 5e-05 | 3.595 | 3.805 | 0.254 | -0.033 | 0.030 | +0.210 | 0.321 | 2.773 | 10.9 | 52 |
| 60 | 6e-05 | 3.607 | 3.853 | 0.288 | -0.024 | 0.027 | +0.246 | 0.296 | 2.466 | 11.0 | 52 |
| 70 | 7e-05 | 3.572 | 3.723 | 0.171 | 0.027 | 0.036 | +0.151 | 0.434 | 1.259 | 8.1 | 51 |
| 80 | 8e-05 | 3.488 | 3.590 | 0.135 | 0.023 | 0.059 | +0.103 | 0.449 | 0.476 | 5.4 | 51 |
| 90 | 9e-05 | 3.425 | 3.423 | 0.045 | 0.065 | 0.076 | -0.002 | 0.999 | 0.005 | 1.0 | 51 |
| 100 | 1e-04 | 3.373 | 3.373 | 0.044 | 0.079 | 0.090 | +0.000 | 1.000 | 0.000 | 1.0 | 86 |
| 110 | 1e-04 | 3.316 | 3.316 | 0.045 | 0.095 | 0.105 | +0.000 | 1.000 | 0.000 | 1.0 | 56 |
| 120 | 1e-04 | 3.256 | 3.256 | 0.045 | 0.112 | 0.122 | +0.000 | 1.000 | 0.000 | 1.0 | 52 |
| 130 | 1e-04 | 3.198 | 3.198 | 0.044 | 0.128 | 0.137 | +0.000 | 1.000 | 0.000 | 1.0 | 51 |
| 140 | 1e-04 | 3.137 | 3.137 | 0.043 | 0.145 | 0.154 | +0.000 | 1.000 | 0.000 | 1.0 | 54 |
| 150 | 1e-04 | 3.075 | 3.075 | 0.043 | 0.161 | 0.170 | +0.000 | 1.000 | 0.000 | 1.0 | 51 |
| 160 | 1e-04 | 3.011 | 3.011 | 0.043 | 0.176 | 0.188 | +0.000 | 1.000 | 0.000 | 1.0 | 54 |
| 170 | 1e-04 | 2.948 | 2.948 | 0.044 | 0.190 | 0.205 | +0.000 | 1.000 | 0.000 | 1.0 | 52 |
| 180 | 1e-04 | 2.883 | 2.883 | 0.045 | 0.205 | 0.222 | +0.000 | 1.000 | 0.000 | 1.0 | 52 |
| 190 | 1e-04 | 2.812 | 2.812 | 0.047 | 0.218 | 0.241 | +0.000 | 1.000 | 0.000 | 1.0 | 53 |
| 200 | 1e-04 | 2.742 | 2.742 | 0.048 | 0.231 | 0.260 | +0.000 | 1.000 | 0.000 | 1.0 | 77 |
| 210 | 1e-04 | 2.687 | 2.687 | 0.049 | 0.241 | 0.275 | +0.000 | 1.000 | 0.000 | 1.0 | 51 |

### A2. Held-out A recovery by source and by removed-length bucket

| step | SNI | RLVR | SQuAD | KL gist gap SNI / RLVR / SQuAD | len le64 | len 65_128 | len 129_208 |
|---|---|---|---|---|---|---|---|
| 0 | 0.110 | 0.258 | 0.019 | 4.58 / 1.72 / 1.88 | 0.146 | 0.060 | 0.256 |
| 10 | 0.121 | 0.152 | -0.075 | 4.43 / 1.90 / 2.04 | 0.083 | 0.125 | 0.082 |
| 20 | -0.032 | 0.141 | -0.285 | 5.23 / 1.86 / 2.41 | -0.037 | -0.029 | 0.026 |
| 30 | -0.047 | 0.094 | -0.229 | 5.30 / 1.91 / 2.38 | -0.042 | -0.064 | 0.031 |
| 40 | -0.042 | 0.116 | -0.341 | 5.31 / 1.82 / 2.54 | -0.060 | -0.057 | 0.017 |
| 50 | 0.017 | 0.080 | -0.330 | 5.00 / 1.95 / 2.52 | -0.048 | 0.007 | 0.010 |
| 60 | 0.023 | 0.093 | -0.322 | 5.04 / 1.94 / 2.50 | -0.040 | -0.004 | 0.075 |
| 70 | -0.011 | 0.098 | 0.015 | 5.19 / 1.99 / 1.85 | 0.041 | -0.029 | 0.046 |
| 80 | 0.026 | 0.098 | -0.097 | 4.93 / 2.01 / 2.08 | 0.031 | -0.009 | 0.033 |
| 90 | 0.056 | 0.110 | 0.017 | 4.86 / 2.03 / 1.90 | 0.074 | 0.032 | 0.059 |
| 100 | 0.069 | 0.114 | 0.052 | 4.79 / 2.02 / 1.83 | 0.090 | 0.041 | 0.072 |
| 110 | 0.081 | 0.122 | 0.090 | 4.72 / 2.00 / 1.75 | 0.109 | 0.051 | 0.082 |
| 120 | 0.094 | 0.129 | 0.130 | 4.64 / 1.98 / 1.67 | 0.127 | 0.063 | 0.093 |
| 130 | 0.106 | 0.136 | 0.172 | 4.58 / 1.96 / 1.58 | 0.145 | 0.075 | 0.102 |
| 140 | 0.119 | 0.143 | 0.211 | 4.49 / 1.95 / 1.50 | 0.163 | 0.089 | 0.114 |
| 150 | 0.132 | 0.152 | 0.247 | 4.41 / 1.93 / 1.42 | 0.181 | 0.103 | 0.126 |
| 160 | 0.147 | 0.160 | 0.273 | 4.31 / 1.92 / 1.37 | 0.195 | 0.118 | 0.143 |
| 170 | 0.163 | 0.171 | 0.291 | 4.21 / 1.90 / 1.33 | 0.209 | 0.133 | 0.164 |
| 180 | 0.178 | 0.181 | 0.307 | 4.11 / 1.88 / 1.30 | 0.223 | 0.148 | 0.182 |
| 190 | 0.195 | 0.190 | 0.319 | 3.99 / 1.86 / 1.27 | 0.235 | 0.165 | 0.201 |
| 200 | 0.213 | 0.197 | 0.328 | 3.86 / 1.85 / 1.25 | 0.245 | 0.183 | 0.220 |
| 210 | 0.227 | 0.202 | 0.333 | 3.76 / 1.83 / 1.24 | 0.253 | 0.198 | 0.236 |

Bucket sizes (gap examples): le64: 157, 65_128: 44, 129_208: 16. KL no-gist by source: sni 5.21, rlvr 2.31, squad 2.01.

### A3. Held-out B (unseen SNI tasks, every 100 steps) and attention (every 500 steps)

| step | KL gist | KL no-gist | recovery | ratio of means | swap − own | cross-gist cosine | attention on gist 1-16 / 2-16 |
|---|---|---|---|---|---|---|---|
| 0 | 3.533 | 3.927 | 0.102 | 0.100 | +0.579 | 0.203 | 0.074 / 0.070 |
| 100 | 3.695 | 3.927 | 0.050 | 0.059 | -0.000 | 1.000 | — |
| 200 | 3.046 | 3.927 | 0.200 | 0.224 | +0.000 | 1.000 | — |

### A4. Training, per 10 steps

Train KD is the step-wide token-weighted mean (section 2.4); every step uses new examples, so it is noisy.

| steps | LR at end | train KD mean | min | max | rollout entropy | median pre-clip grad norm | max grad norm | steps clipped | s/step |
|---|---|---|---|---|---|---|---|---|---|
| 1-10 | 1e-05 | 1.208 | 0.749 | 1.861 | 2.952 | 1973.6 | 1.73e+04 | 10/10 | 6.5 |
| 11-20 | 2e-05 | 1.097 | 0.426 | 1.707 | 3.145 | 854.2 | 4.01e+05 | 10/10 | 6.6 |
| 21-30 | 3e-05 | 1.377 | 0.757 | 2.579 | 3.228 | 725.9 | 2.26e+04 | 10/10 | 6.7 |
| 31-40 | 4e-05 | 1.127 | 0.811 | 1.837 | 2.898 | 362.8 | 5.22e+03 | 10/10 | 6.6 |
| 41-50 | 5e-05 | 1.353 | 0.685 | 3.112 | 2.776 | 257.4 | 1.8e+03 | 10/10 | 6.8 |
| 51-60 | 6e-05 | 1.430 | 1.090 | 2.106 | 2.551 | 372.5 | 1.59e+04 | 10/10 | 6.2 |
| 61-70 | 7e-05 | 1.399 | 0.526 | 1.986 | 2.054 | 338.9 | 2.95e+03 | 10/10 | 6.1 |
| 71-80 | 8e-05 | 1.100 | 0.598 | 1.884 | 0.822 | 27.9 | 7.98e+03 | 10/10 | 6.1 |
| 81-90 | 9e-05 | 1.291 | 0.750 | 2.020 | 0.244 | 13.8 | 204 | 10/10 | 6.6 |
| 91-100 | 1e-04 | 1.123 | 0.571 | 2.079 | 0.001 | 1.5 | 1.82 | 7/10 | 7.0 |
| 101-110 | 1e-04 | 1.180 | 0.729 | 2.070 | 0.000 | 2.0 | 2.64 | 7/10 | 6.8 |
| 111-120 | 1e-04 | 1.154 | 0.523 | 1.976 | 0.000 | 1.9 | 3.57 | 8/10 | 6.2 |
| 121-130 | 1e-04 | 1.216 | 0.807 | 2.279 | 0.000 | 1.7 | 3.64 | 9/10 | 6.0 |
| 131-140 | 1e-04 | 1.111 | 0.587 | 2.449 | 0.000 | 1.8 | 2.71 | 9/10 | 6.9 |
| 141-150 | 1e-04 | 0.921 | 0.602 | 1.269 | 0.000 | 1.5 | 2.43 | 8/10 | 6.1 |
| 151-160 | 1e-04 | 0.846 | 0.569 | 1.134 | 0.000 | 1.5 | 2.72 | 8/10 | 6.1 |
| 161-170 | 1e-04 | 1.195 | 0.568 | 1.676 | 0.000 | 2.1 | 5.76 | 9/10 | 6.0 |
| 171-180 | 1e-04 | 0.910 | 0.622 | 1.307 | 0.000 | 1.5 | 3 | 8/10 | 6.2 |
| 181-190 | 1e-04 | 1.021 | 0.769 | 1.369 | 0.000 | 2.5 | 3.47 | 9/10 | 6.1 |
| 191-200 | 1e-04 | 1.010 | 0.426 | 2.108 | 0.000 | 2.4 | 5.88 | 9/10 | 6.3 |
| 201-210 | 1e-04 | 1.033 | 0.357 | 1.746 | 0.000 | 1.8 | 2.67 | 9/10 | 6.2 |
| 211-218 | 1e-04 | 0.860 | 0.614 | 1.116 | 0.000 | 2.3 | 3.03 | 8/8 | 6.1 |
