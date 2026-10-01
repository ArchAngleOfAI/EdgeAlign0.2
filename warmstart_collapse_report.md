# Warm-start gist pretraining: pipeline and gist collapse

Snapshot at **step 140 of 5,000** (2026-10-01, ~18:00 UTC). The full run is still running on
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
   ready-made baseline: KL_gap 3.137 at step 140.
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
