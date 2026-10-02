# MEMORY.md — Long-Term Project Memory

Durable knowledge about this project: goal, structure, conventions, and
major decisions. Update this file when something becomes true "for the
life of the project," not for in-flight task state (that belongs in
`SHORT_MEMORY.md`).

## Project Goal

The user has not written a formal goal statement. What the project does
(from the work since 2026-09-18): **soft-prompt ("gist") distillation on
Qwen3-0.6B.** A student encoder rolls out a short continuous prefix (the
"gist") from a prompt. A frozen receiver sees [gist ++ prompt] and is trained
(via the encoder) to match a frozen teacher that sees the plain prompt,
using the loss KL(teacher || receiver) on the prompt positions. The user's
ideas are often non-standard; see AGENT.md.

## Environment

- **Model:** `/data/models/huggingface/qwen3-0.6b` — Qwen3ForCausalLM,
  0.6B (hidden_size 1024, 28 layers, 16 attn heads / 8 KV heads, vocab
  151936, bf16, tied embeddings). Saved with `transformers==4.51.0`
  per its `config.json`.
- **Python env (since 2026-09-29):** `/data/a84460786/venvs/testfolder`
  (Python 3.12.3, `torch==2.14.0+cu130`, `transformers==5.17.0`,
  `accelerate`, `fsspec`, `aiohttp`, `pyarrow`, `matplotlib`). It was built with
  `venv --without-pip` + get-pip.py, because the system Python lacks ensurepip.
  The old env, `/home/a84460786/EdgeAlign/.venv`, is gone: the whole
  EdgeAlign directory was deleted on 2026-09-29 (not by the agent).
  `diagnostics/run_all.sh` and `diagnostics/run_stability.sh` still default to
  the old path; override with `PY=/data/a84460786/venvs/testfolder/bin/python`.
- **Cluster:** 8x NVIDIA A100-PCIE-40GB, shared with other users. Check
  `nvidia-smi` and set `CUDA_VISIBLE_DEVICES` before every run.
  **GPU 0 is faulty** ("invalid access of peer GPU memory ... hardware error"
  on 2026-09-29); GPU 2 shows a permanently-100% vLLM worker from another
  user but works. GPUs 5, 2, 7, 6 and 4 have been used successfully (6 + 4 for
  the 2-GPU run). Ownership by other users changes within hours; check
  `nvidia-smi` plus the process owner right before launching.
- **Long runs must be launched fully detached** (`setsid nohup ... < /dev/null &`).
  The 2026-09-30 warm-start smoke run died silently when its session ended.
- torchrun sets OMP_NUM_THREADS=1 per process; the warm-start full run used
  OMP_NUM_THREADS=14 (not shown to matter).
- **GPU topology:** no NVLink. Two PCIe groups, one per CPU socket: GPUs 0-3
  (NUMA 0) and GPUs 4-7 (NUMA 1). Pairs inside a group share a PCIe switch;
  pairs across groups go over the socket link.
- **Multi-GPU (NCCL): direct GPU-to-GPU P2P hangs on this machine** (found
  2026-10-01 on GPUs 4+6: the first NCCL broadcast never completes, both GPUs
  sit at 99% spinning). `NCCL_P2P_DISABLE=1` fixes it (~3.5 GB/s via host
  memory). `warmstart/train_warmstart.py` sets it itself under torchrun.
- Debugging running processes: `ptrace_scope=1`, so `py-spy dump` on an
  already-running process is refused; launch under `py-spy record
  --subprocesses -- ...` instead (py-spy is installed in the venv).
- **Disk:** `/` (home, repo) has only ~30 GB free. Put large files
  (checkpoints, venvs, pip cache) under `/data/a84460786/`.

## Repository Structure

- Git repo, branch `main`, remote `origin` =
  https://github.com/ArchAngleOfAI/EdgeAlign0.2 (same GitHub account as the
  old EdgeAlign repo). Auth uses a token stored via `credential.helper
  store` in `~/.git-credentials`; pushes work non-interactively.
- `.gitignore`: `__pycache__/`, `*.pyc`, `checkpoints/`, `*.pt`, `*.pt.tmp`, and `logs/*`
  except `logs/train_3layer.jsonl`, `logs/eval_3layer.jsonl`, `logs/warmstart_train.jsonl`,
  `logs/warmstart_eval.jsonl` and `logs/warmstart.out` (tracked). Checkpoints live outside the
  repo under /data/a84460786/testfolder_checkpoints/.
- Core code: `smoke_train.py` (FineWeb-Edu loader `fineweb_edu_batches`),
  `qwen_dual_embedding.py` (`build_softprompt_generator`),
  `distill_softprompt.py` (`generate_softprompt`, `receiver_forward`, original
  training loop), `train_3layer.py` (the 3-layer long-run trainer),
  `replot_3layer.py` (redraws its loss curve over the steps actually trained).
- Diagnostics: `diagnostics/` (shared `_common.py` with `load_all`,
  `forward_loss`, verbatim `kd_loss`; tests 1-7; `stability_vs_length.py`)
  and `diagnostics_gist/` (`_gist_common.py`, t1-t8).
- Reports: `report.md` (flat-loss diagnosis), `stability_report.md`,
  `RUN_NOTES_3layer.md`, `gist_health_report.md`, `warmstart_collapse_report.md`
  (warm-start run: pipeline, collapse, causes, options, full tables for steps 0-1353).
- `warmstart/` (since 2026-09-30): "fill in the removed part" gist pretraining.
  `common.py` (sequence construction, mid-sequence receiver, batched
  `rollout_gists` + per-example `rollout_gists_serial`), `prepare_data.py`
  (build / gen / klscore / finalize; data in /data/a84460786/warmstart_data/),
  `train_warmstart.py` (trainer; `--checks-only` runs checks (a)-(e) and writes
  `step2_checks.json`; `--smoke`; `--micro-batch`; `--resume`; data parallel
  when launched with `torchrun --nproc_per_node=N`), `TASK_PROMPT.md` (the
  user's original spec, verbatim, with a header listing agreed changes).

## Conventions & Decisions

- The user asks explicitly for each commit and push, and usually asks for
  a SHORT_MEMORY log entry plus a push after each finished piece of work.
  Commit messages end with the Co-Authored-By line.
- Diagnostic and experiment requests usually say: don't modify existing
  files, put new scripts in a new folder, reuse existing functions rather
  than reimplementing them, and commit JSON/logs/plots plus a report at the
  repo root.
- Held-out data for anything trained on FineWeb-Edu shard `000_00000` comes
  from shard `001_00000`, which training never reads (see RUN_NOTES_3layer.md).
- The user chose to keep FineWeb-Edu rather than
  `/data/r50058044/reskill_search/retriever/wiki-18.jsonl`. That file is
  another user's, and is a tar archive wrapping a FlashRAG wiki dump, not
  plain JSONL.
- The docstrings in `distill_softprompt.py` / `qwen_dual_embedding.py` still
  say "only Embedding2 trainable" in places. Since 2026-09-22 the untied
  LM_head and the last decoder layer are also trainable (train_3layer.py
  unfreezes the last 3 layers).

- Warm-start KD loss (user decision 2026-10-01): token-weighted mean over all
  response tokens of the optimizer step (all micro-batches together), so the
  objective does not depend on the micro-batch size.

## Numerics of the gist rollout (found 2026-10-01)

- The continuous 16-step rollout + BPTT is badly conditioned: per-example grad
  norms span ~1e-1 to ~1e5, and one example often dominates a step's gradient.
  fp32 rounding differences of ~1e-6 grow to percent level in gists and up to
  tens of percent in grads (serial fp32 vs fp64 reached 62% on one micro-batch;
  gradient direction still cos >= 0.99999 per micro-batch).
- Consequences: batched and per-example rollouts can never agree exactly, and
  changing the batch shape changes the step gradient (cos ~0.95-0.97 between
  micro-batch 1 and 4 on one step). Test batching via invariances at fixed
  shapes (pad-token swap), not via agreement with the serial version.
- transformers 5.17 `Qwen3RMSNorm` computes in fp32 even when the model is fp64,
  so ".double()" models are not a true fp64 reference (batch 1 vs batch 4 of
  identical inputs differ by 2.7e-7 in "fp64").
- GPU backward is not bit-reproducible (~4e-6 run to run); use
  `torch.use_deterministic_algorithms(True)` with
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` (set before CUDA init) for exactness tests.
- Eager attention gives NaN on fully-masked (left-pad) query rows; sdpa is fine.
- **Future improvement (user asked to keep this, 2026-10-01): one example can
  dominate a step.** Training clips once per optimizer step
  (`clip_grad_norm_(params, 1.0)` on the gradient summed over all 16 examples).
  That caps the step size (a 1e5 spike reaches AdamW at norm 1.0) but keeps each
  example's share of the direction: on one measured step a single example had
  grad norm 3.6e3 of the step's 3.7e3, so the update was almost all that example
  and the other 15 had almost no say. Options, none implemented:
  (1) cheap diagnostic first: log the largest per-example grad share per step to
  see how often one example dominates; (2) per-example gradient clipping
  (cap each example's grad before summing). This needs a per-example backward,
  which costs part of the batched-rollout speed-up. Possibly linked to the
  3-layer run's late instability.

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
   **output** projection). **Updated 2026-09-22:** LM_head is now UNTIED
   from Embedding1 (own cloned Parameter) and trainable, and the last decoder
   layer is trainable too. train_3layer.py unfreezes the last 3 and also sets
   `config.tie_word_embeddings=False`, because a `tie_weights()` call would
   otherwise re-tie them (report.md test 3). Used by the distillation encoder
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

## Key results so far (details in the reports)

- **Flat loss (report.md):** the flat 17k-step run trained Embedding2 only.
  It did improve (9% on paired batches), but per-batch noise (~0.2) hid it.
  The gradient path is intact. BPTT through the rollout amplifies gradients.
  More trainable layers help.
- **Stability (stability_report.md):** at LR 1e-3, shorter soft prompts jump
  less (L=100 +0.56, L=16 +0.18, L=8 +0.06). clip_grad_norm_ 1.0 does not
  remove the jump.
- **3-layer run (RUN_NOTES_3layer.md):** L=16, peak LR 1e-4 with warmup,
  global batch 16, clip 1.0. Best held-out 0.0105 @ step 2600 (from
  0.393). Unstable after step ~2000; stopped by the user at step 3029.
- **Warm-start (warmstart_collapse_report.md):** the "fill in the removed part" objective did NOT
  prevent collapse. By step 90 (LR 9e-5, in warmup) every input gave `' prompt'` x16. Main cause:
  recovery is measured against DELETING the span, so a generic placeholder earns large
  "recovery" (a learned constant prefix reached 0.393 on held-out A and 0.331 on held-out B at
  step 1300 vs 0.136 / 0.102 for the informative step-0 gist). Once the rollout is one-hot, the
  Embedding2 row of the token trains as a constant soft prompt. Loss and recovery both look like
  progress after the collapse; only swap minus own, cross-gist cosine and entropy reveal it.
  Stopped by the user at step 1353 of 5,000 (2026-10-01).
- **Gist health (gist_health_report.md):** from step 1000 on, the generator
  is collapsed to one prompt-independent one-hot token sequence, but the gist
  is "alive": it beats naive prefixes 14-16x, is more noise-sensitive than
  real text, and is steerable. Attention to gist positions 2-16 falls across
  training (0.125 -> 0.0035). KD loss alone did not reveal the collapse.

## Lessons from the gist runs (durable)

- **Recovery against deletion rewards placeholders.** Deleting the removed span leaves a
  malformed prompt, so a content-free placeholder earns large "recovery". Always compare a gist
  against a constant-prefix / placeholder baseline (the collapsed warm-start run gives one: 0.393
  on held-out A), not only against deletion.
- **Watch the collapse indicators, not loss/recovery.** Swap minus own ~ 0, cross-example
  cosine ~ 1 and rollout entropy ~ 0 are the signals; KD loss and recovery kept improving
  after both collapses (3-layer and warm-start).
- **The one-hot state is absorbing.** Once softmax(logits) saturates (top-1/top-2 gap ~14+),
  almost no gradient flows through the rollout, and `onehot @ Embedding2` makes the single
  Embedding2 row of the emitted token the only thing that trains, i.e. prefix-tuning of one
  constant vector.

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
- 2026-09-22: Trainable set expanded (untied LM_head + last layer), grad
  accumulation, soft-prompt-length sweep.
- 2026-09-29: Remote set to ArchAngleOfAI/EdgeAlign0.2. Flat-loss
  diagnostics (report.md) and stability grid (stability_report.md).
  EdgeAlign venv lost, new venv on /data. 3-layer long run launched.
- 2026-09-30: 3-layer run stopped at step 3029 (best 0.0105 @ 2600). Gist
  health check (gist_health_report.md). Warm-start pretraining data + trainer
  (warmstart/).
- 2026-10-01: Warm-start smoke test passed; batched gist rollout (3.2x faster
  steps), check (e), step-wide token-weighted KD loss. 2-GPU data parallel
  (7.05 s/step on GPUs 6+4, NCCL P2P disabled). Full run on GPUs 6+4 collapsed by
  step 90 to `' prompt'` x16; stopped by the user at step 1353 (warmstart_collapse_report.md).
  Warm-start logs and plots now tracked in git (commit 39b00ee).
