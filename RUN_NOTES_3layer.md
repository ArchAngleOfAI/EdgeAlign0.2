# Run notes — `train_3layer.py` (launched 2026-09-29)

Status: **stopped at the user's request on 2026-09-30 ~13:55 UTC, at step 3029
of 10,000**. The run was stopped with SIGTERM, not a crash or NaN stop. The last
eval and checkpoint are at step 3000. The training log also contains steps
3001–3029, which no checkpoint covers. `--resume` would continue from step
3000 and re-run those 29 steps on the same data.

**Headline: the loss curve shows training instability after step ~2000.**
Held-out loss reached its best, **0.0105 at step 2600** (−97% vs 0.3930 at
step 0). From step ~2650 it regressed sharply and had not recovered by the
stop (0.2066 at step 3000), which is back to about its step-290 level. See
"Final results" below.

## Exact settings

| setting | value |
|---|---|
| model | Qwen3-0.6B (`/data/models/huggingface/qwen3-0.6b`): student encoder + frozen teacher + frozen receiver, fp32 |
| trainable | Embedding2 + untied LM_head + the encoder's **last 3** decoder layers = 358,357,760 params; everything else frozen |
| untie safety | `student_encoder.config.tie_word_embeddings = False`; startup assert that lm_head / embed_tokens / embedding2 have 3 distinct `data_ptr`s (passed) |
| rollout | soft-prompt length L = 16, full backprop through the rollout (no detach) |
| data | FineWeb-Edu `sample/10BT/000_00000.parquet` via `smoke_train.fineweb_edu_batches`, prompt length 128 |
| optimizer | AdamW (default betas/eps/weight decay), peak LR 1e-4 |
| LR schedule | step *k* uses LR = 1e-4 · min(k, 100)/100 → linear warmup 1e-6 … 1e-4 over steps 1–100, then constant (no decay) |
| batch | global 16 = micro-batch 2 × grad-accum 8; each micro loss divided by 8 before `backward()` |
| clipping | `clip_grad_norm_(trainable_params, max_norm=1.0)` once per optimizer step, after all 8 micro-batches, right before `optimizer.step()`; pre-clip norm logged every step |
| steps | 10,000 optimizer steps (= 80,000 micro-batches = 160,000 prompts ≈ 20.5M tokens) |
| seed | 0 |
| eval | 32 held-out prompts (8 × 4) at L=16, `no_grad`, same KD loss; at step 0, every 10 steps through 300, then every 100 to 10,000 (128 evals) |
| NaN/inf guard | the run exits with a `STOP:` message on any non-finite loss or grad norm, before `optimizer.step()` |
| code reuse | model build, unfreezing, trainable list, forward pass and KD loss come from `diagnostics/_common.py`, which imports `build_softprompt_generator`, `generate_softprompt`, `receiver_forward` and `fineweb_edu_batches` and holds the verbatim KD loss |

## How held-out / train overlap was prevented

1. **Different source file.** The held-out prompts come from FineWeb-Edu shard
   `001_00000.parquet`, row group 0. Training reads only `000_00000.parquet`,
   because `fineweb_edu_batches` hard-codes that URL, so the training stream
   cannot reach the held-out file no matter how many steps it runs. The
   held-out batches are packed by the **unmodified** `fineweb_edu_batches`
   (EOS-separated docs, 128-token chunks). The script swaps
   `smoke_train.FINEWEB_EDU_SHARD_URL` only while drawing those 8 batches,
   then restores it and asserts the restore before training starts.
2. **Startup document check.** Before training, `check_no_overlap` compared
   the held-out source documents (2 docs cover all 32 prompts) against every
   training document in row groups 0–39 of the training shard: 40,000 docs,
   ~41.9M GPT-2 tokens, ≥ 2× the 20.5M tokens the 10,000 steps consume. It
   compared both the FineWeb `id` and the SHA-1 of the text. Result: **no
   shared id or text hash** (logged in `logs/train_3layer.out`).

## Environment, hardware, timing

- **GPU:** 2 (A100-PCIE-40GB). GPU 0 raised a CUDA hardware error earlier
  (see `stability_report.md`). GPU 5 was occupied by other users' jobs at
  launch (32 GB used). GPU 2 passed a matmul + 20 GB allocation health check
  but is shared with another user's vLLM worker that shows 100% utilization,
  which likely makes this run slower than earlier timings suggested.
- **Python env:** new venv at `/data/a84460786/venvs/testfolder` (torch
  2.14.0+cu130, transformers 5.17.0, same versions as before). The old
  EdgeAlign venv was removed from disk on 2026-09-29. The `diagnostics/*.sh`
  runners still default to the old path; set
  `PY=/data/a84460786/venvs/testfolder/bin/python` to use them.
- **Smoke run** (20 steps, `--smoke`, GPU 2): **19.94 s per optimizer step**.
  A 10-step `--resume` continuation ran at 22.29 s/step and restored step,
  LR position and data position correctly (the train log continued without
  gaps from step 1 to 30).
- **Estimated total:** 19.94 s × 10,000 ≈ **55.4 h** of training plus 128
  evals/checkpoint writes. Measured mean over the first 300 steps: 20.32 s/step
  → **≈ 56.5 h** of training. (The run was stopped at step 3029, so it never reached 10,000.)
- **Checkpoints** (outside git, on `/data`):
  `/data/a84460786/testfolder_checkpoints/train_3layer/`. `latest.pt` is
  rewritten at every eval, `best.pt` is a hard link to the best-eval
  checkpoint, and `step_XXXXX.pt` hard links are kept every 1,000 steps.
  Each file is ~4.3 GB. Resume with
  `python train_3layer.py --resume`.

## Final results (run stopped at step 3029)

| | step | mean held-out loss | per-batch (8 batches × 4 prompts) |
|---|---|---|---|
| start | 0 | 0.3930 | 0.1920, 0.8578, 0.5906, 0.2354, 0.2255, 0.6223, 0.2461, 0.1744 |
| **best** | **2600** | **0.0105** | 0.0080, 0.0155, 0.0103, 0.0081, 0.0088, 0.0105, 0.0119, 0.0108 |
| final (last eval) | 3000 | 0.2066 | 0.1755, 0.2212, 0.2048, 0.1894, 0.2356, 0.2136, 0.2085, 0.2040 |

At the best step every batch is between 0.008 and 0.016. At the final eval
every batch is between 0.18 and 0.24, so the regression hits all 8 batches and
does not come from a single outlier.

### Instability after step 2000

The loss curve (`kd_loss_curve_3layer_lr1e-4_bs16_accum8_gradclip_trained_range.png`)
shows training becoming unstable after step ~2000. The evidence:

- **Held-out spikes grow.** Earlier spikes were small and short (held-out
  0.052 at step 1500). At step 2000 held-out rose to **0.0836** from 0.020 at
  step 1900, then recovered to 0.0107–0.0128 over steps 2100–2600. From step
  ~2650 it regressed again, to **0.1505** at step 2700, 0.0723 at 2800,
  **0.2117** at 2900 and **0.2066** at 3000, with no recovery before the stop.
- **Training loss shows the same thing.** Per-step training loss ranged
  0.0083–0.0691 over steps 2001–2600 and **0.0096–0.3157** over steps
  2601–3029. The 50-step moving average went from ~0.013 at step 2600 to
  ~0.23 at step 2900.
- **Gradients grow and clipping fires more often:**

  | steps | clipping active | median pre-clip grad norm | max pre-clip grad norm |
  |---|---|---|---|
  | 1001–2000 | 204 / 1000 (20%) | 0.65 | 1,783.9 |
  | 2001–2600 | 151 / 600 (25%) | 0.79 | 87.0 |
  | 2601–3029 | 221 / 429 (52%) | 1.03 | 409.0 |

  The median pre-clip norm rose steadily through training, and after step 2600
  it sits right at the clip threshold (1.0).
- **Context:** this is the third distinct regression in this run. The first was
  at step 157 (held-out 0.105 → 0.278), which recovered by step ~400. It
  follows the same fast-drop → sudden-jump pattern `stability_report.md` found
  at LR 1e-3. Here it happens at a constant peak LR of 1e-4, with warmup,
  gradient clipping at 1.0 and 3 trainable layers. This run does not show what
  causes it: the LR is constant after warmup, so it cannot tell LR from other
  factors.

**Checkpoints kept** (`/data/a84460786/testfolder_checkpoints/train_3layer/`,
~4.3 GB each): `best.pt` (step 2600, held-out 0.0105), `step_01000.pt`,
`step_02000.pt`, `step_03000.pt` (the same file as `latest.pt`, step 3000).

**Timing, measured:** 20.27 s per optimizer step on average over 3,029 steps
(GPU 2), plus eval/checkpoint overhead.

## Held-out loss at the step-300 commit (mean of 32 prompts)

| step | mean | per-batch |
|---|---|---|
| **0** | **0.3930** | 0.1920, 0.8578, 0.5906, 0.2354, 0.2255, 0.6223, 0.2461, 0.1744 |
| 150 (best so far) | 0.1046 | — (see `logs/eval_3layer.jsonl`) |
| **300** | **0.1789** | 0.1549, 0.1586, 0.1475, 0.1736, 0.1596, 0.2014, 0.2163, 0.2197 |

**Observation (not yet explained):** training loss jumped at step 157, from
~0.12–0.17 to ~0.26–0.36. Held-out loss rose with it, from 0.105 at step 150
to 0.278 at step 160, and all 8 batches rose together. After the jump the
pre-clip gradient norm stayed at ~0.3–0.8, so clipping was rarely active. Held-out
loss then fell slowly: 0.283 at step 170, 0.240 at step 260, 0.223 at step 290,
0.179 at step 300. This matches the drop → jump → slow-recovery pattern
`stability_report.md` found at LR 1e-3, here at a peak LR of 1e-4.
`best.pt` holds the step-150 weights.

## Files

- `train_3layer.py` — the training script
- `logs/train_3layer.jsonl`, `logs/eval_3layer.jsonl` — per-step (steps
  1–3029) and per-eval (steps 0–3000) logs, final
- `kd_loss_curve_3layer_lr1e-4_bs16_accum8_gradclip.png` — the run's own
  plot, x-axis 0–10,000, as last written at the step-3000 eval
- `replot_3layer.py` + `kd_loss_curve_3layer_lr1e-4_bs16_accum8_gradclip_trained_range.png` —
  the same plot, x-axis limited to the steps trained so far. The script only
  reads the logs, runs on CPU, and is safe to run during training.
