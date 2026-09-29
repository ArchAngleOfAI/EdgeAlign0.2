# Why is the KD loss flat? — Diagnostic report (2026-09-29)

All scripts are in `diagnostics/` (shared setup in `diagnostics/_common.py`, which
imports `generate_softprompt`, `receiver_forward`, `build_softprompt_generator` and
`fineweb_edu_batches` from the existing code rather than reimplementing them). Raw
output of every run is in `diagnostics/results/*.log`. The training code itself was
not modified. Unless stated otherwise, every run uses the same setup as the flat run:
batch size 4, prompt length 128, 100 soft tokens, LR 5e-5, AdamW, fp32, the same
deterministic FineWeb-Edu stream. All runs were on one A100 (GPU 5), with a single
seed.

## Summary

The evidence rules out Hypothesis A: the gradient path is intact, every trainable
parameter gets a healthy nonzero gradient, and on a fixed batch the loss drops quickly
(0.446 → 0.271 in 20 steps at the normal LR). The evidence partially supports
Hypothesis B, with one correction to its premise. **The flat 17k-step run
(`logs/distill_100k.log`) was launched on 2026-09-18, before the trainable set was
expanded. In that run only Embedding2 was trainable; LM_head and the last decoder
layer were frozen** (the log header reads `trainable (Embedding2): 155,582,464`). That
run was not stuck either. Its surviving step-10,000 checkpoint gives a 0.043 (9%) lower
KD loss than the untrained model on the same 20 batches. That gain is invisible in the
per-step log, because the per-step loss is measured on a fresh batch each step and
varies by about ±0.2 between batches. More trainable capacity makes a large
difference. In the same ~57 training steps, held-out loss fell 0.708 → 0.438 (−38%,
8/8 batches better) with the last 3 layers trainable, against 0.708 → 0.615 (−13%,
4/8 better) with only the last layer. **Best-guess root cause:** (1) the flat run
trained only Embedding2, which learns very slowly, and (2) per-step training loss on
fresh batches is too noisy to show the slow progress that did happen. A secondary
finding: backprop through the 100-step rollout amplifies gradients rather than
shrinking them, and at LR 1e-3 training becomes unstable.

## Hypothesis A findings (broken/vanishing gradient path) — not supported

**Test 1 — `check_gradients.py`.** One forward and backward pass on the first training
batch. The KD loss was 0.446160, identical to step 1 of `logs/distill_100k.log`, so
the diagnostics reproduce the training path exactly. No gradient is `None` or exactly
zero:

| parameter | ‖grad‖ | ‖grad‖/‖w‖ | fraction of elements with \|g\|>1e-8 |
|---|---|---|---|
| embedding2.weight | 1.421e+01 | 3.9e-02 | 0.558 |
| lm_head.weight | 9.995e+01 | 2.7e-01 | 0.655 |
| layers[-1].self_attn.q/k/v/o_proj | 7.66 / 6.23 / 18.94 / 11.54 | 0.17–0.57 | 1.000 |
| layers[-1].mlp.gate/up/down_proj | 28.60 / 39.23 / 52.26 | 0.53–1.09 | ~0.997 |
| layers[-1].q_norm / k_norm | 0.302 / 0.789 | 1.4e-02 / 3.1e-02 | 1.000 |
| layers[-1].input / post_attention_layernorm | 0.038 / 0.201 | 7.9e-05 / 8.9e-04 | 1.000 |

No parameter outside the trainable set received a gradient, and Embedding1
(`embed_tokens`) has `grad=None`. All 151,936 Embedding2 rows get some gradient, but
it is concentrated: 10 rows (` not`, ` is`, ` know`, ` do`, ` think`, `,`, ` but`,
` a`, ` and`, ` the`) hold 65.8% of the squared gradient norm.

**Test 2 — `check_gradfn.py`.** `soft_token.grad_fn` is never `None`. It is
`UnsafeViewBackward0` at step 1, 50 and 100, and the final softprompt has
`CatBackward0`. The graph grows as expected for backprop-through-time:

| rollout step | graph nodes | longest node chain | trainable leaves reached |
|---|---|---|---|
| 1 | 104 | 55 | embedding2, lm_head, all 11 last-layer params |
| 50 | 102,661 | 62,432 | same |
| 100 | 207,311 | 126,082 | same |

**Test 3 — `check_weight_identity.py`.** After `build_softprompt_generator`, the three
weights have distinct `data_ptr`s: lm_head `0x79930a000000`, embed_tokens
`0x7995b8000000`, embedding2 `0x799330000000`. All three are still distinct after a
real optimizer step. In that step lm_head moved by ‖Δ‖=0.449 and Embedding1 moved by
exactly 0. Calling `.train()`, `.eval()` or `.to()` does not re-tie the weights.

A search of the repo finds no call to `tie_weights`, `resize_token_embeddings`,
`save_pretrained` or `.generate()`. The only `from_pretrained` calls happen before the
untying. In transformers 5.17.0, `tie_weights()` is called only from `_from_config`,
`resize_token_embeddings`, `init_weights` and `_finalize_model_loading` (the last one
runs inside `from_pretrained`). `generation/utils.py` and `modeling_qwen3.py` never
call it. **Latent risk:** `config.tie_word_embeddings` is still `True` and
`_tied_weights_keys = {'lm_head.weight': 'model.embed_tokens.weight'}`, so an explicit
`encoder.tie_weights()` call does re-tie them (demonstrated at the end of the script).
Nothing in the current training path calls it.

**Test 4 — `lr_sanity_check.py`.** Three 20-step runs, each from a fresh init. I added
runs B and C: with fresh batches every step, the loss can never be bit-identical, so
only a fixed batch isolates the effect of the parameter updates.

| run | step 1 | step 5 | step 10 | step 20 | distinct values | ‖Δθ‖/‖θ‖ |
|---|---|---|---|---|---|---|
| A: fresh batches, LR 1e-3 (spec) | 0.446160 | 0.267749 | 0.498831 | 0.606223 | 20/20 | 8.6e-02 |
| B: fixed batch, LR 1e-3 | 0.446160 | 0.708763 | 0.678566 | 0.623522 | 20/20 | 8.2e-02 |
| C: fixed batch, LR 5e-5 | 0.446160 | 0.362281 | 0.395796 | 0.270681 | 20/20 | 5.8e-03 |

The loss is not bit-identical in any run. In B it falls from 0.446 to 0.139 by step 3,
then jumps to 0.709 at step 5 and only creeps down afterwards (0.624 at step 20). That
means LR 1e-3 is unstable, not ineffective. C shows the normal LR reduces the loss on a
fixed batch by 39% in 20 steps. In A, steps 2–3 reach 0.169 and 0.156 on batches whose
untrained loss is 0.431 and 0.756 (from test 5). Later steps drift above the untrained
values.

## Hypothesis B findings (good init + too little trainable capacity) — partially supported

**Test 5 — `naive_baseline_kd.py`.** 20 batches × 4 prompts × 128 tokens: exactly the
batches of steps 1–20 of the flat run. The comparisons are paired.

| prefix (100 vectors) | mean KD ± std over batches | mean diff vs (c) | batches lower than (c) |
|---|---|---|---|
| (a) 100 × pad token `<|endoftext|>` (id 151643) — the tokenizer has **no BOS**, so pad was used | 0.9213 ± 0.0917 | +0.4244 | 1/20 |
| (b) 100 × zero vector | 2.3377 ± 0.9979 | +1.8408 | 0/20 |
| (c) generated, untrained (training start point) | 0.4969 ± 0.1777 | 0 | — |
| (d) generated with Embedding2 from `checkpoints/embedding2_step10000.pt` (added test) | 0.4539 ± 0.1590 | −0.0430 | 12/20 |

The untrained generated softprompt is already far better than the naive prefixes, so
training starts from a genuinely good point. The flat run's checkpoint shows Embedding2
did move: ‖ΔE‖/‖E‖ = 0.102, every row changed, and the largest single-element change is
0.0171. It also gives a 9% lower loss than the start point. That is real learning, and
it is 4× smaller than the ±0.18 batch-to-batch spread, which is why the curve looks
flat. The flat log itself shows the same pattern: 250-step window means of 0.502 at
step 0, 0.499 at step 10k and 0.551 at step 17k, with a per-step std of about 0.20–0.30.

**Test 6 — `capacity_ablation.py`.** The 15-minute limit was hit before 200 steps in
both runs, at ~15.7 s per step. I added the 1-layer run as a matched control on the
same data. I also added a held-out eval: 8 fixed batches (stream batches 1001–1008,
never trained on), scored before and after training.

| trainable layers | steps done | train loss, mean of first 10 / steps 21–40 / last 10 | held-out before → after | paired diff | held-out batches improved |
|---|---|---|---|---|---|
| last 3 | 57 | 0.482 / 0.456 / 0.348 | 0.7079 → 0.4385 | −0.2694 | 8/8 |
| last 1 (current config) | 58 | 0.432 / 0.536 / 0.455 | 0.7079 → 0.6151 | −0.0928 | 4/8 |

Per-step losses are in `diagnostics/results/capacity_ablation_*.log`. Unlike the flat
run, the 3-layer training loss visibly trends down within 57 steps, and its held-out
gain is about 3× the 1-layer gain. Caveat: this is one run per setting, and the
held-out set is small (32 prompts).

**Test 7 — `rollout_length_gradient_decay.py`.** Part 1, as specified: the same batch
with the rollout truncated to L soft tokens.

| L | KD loss | ‖g embedding2‖ | ‖g lm_head‖ | ‖g last layer‖ |
|---|---|---|---|---|
| 10 | 0.2693 | 2.818 | 16.64 | 16.36 |
| 30 | 0.4271 | 31.72 | 249.4 | 196.9 |
| 60 | 0.4768 | 19.26 | 193.6 | 138.3 |
| 100 | 0.4462 | 14.21 | 99.95 | 75.35 |

The gradient does not decay with rollout length. It rises 11× from L=10 to L=30, then
eases off. Truncation also changes how many soft tokens the receiver sees, so I added
Part 2 to separate the two effects. At L=100, feeding each soft token back in detached
leaves only the direct path from each soft token to the receiver. The full run has
‖g embedding2‖ = 14.21; the detached run has 3.06. The recurrent (BPTT) part is
‖full − direct‖ = 13.81, i.e. **97% of Embedding2's gradient arrives through the
recurrence**, and cos(full, direct) = 0.238. Per position, the full gradient
‖dL/ds_k‖ is 1.26×, 3.4×, 12.5×, 4.4× and 5.8× the direct-only gradient at
k = 1, 2, 5, 10, 20, and ~1.0–1.1× for k ≥ 50. Gradient mass is concentrated on the
first ~20 soft tokens (‖dL/ds_k‖ ≈ 3–13 there vs. ≈ 0.03–0.08 for k ≥ 40). Backprop
through 100 steps therefore amplifies the gradient for the early soft tokens rather
than shrinking it. That is consistent with the LR 1e-3 instability in test 4, though
not proven to cause it.

## Existing diagnostic images

- **`kd_loss_vs_softprompt_length.png`** (92,907 bytes, 1350×825) — the untrained KD
  loss against soft-prompt length L = 1…200, averaged over 5 FineWeb-Edu prompts with a
  ±std band. It rises steeply from ~0.07 at L=1 to a peak of ~0.54 around L=40–50, then
  declines slowly to ~0.46 at L=200. The std band is wide, about ±0.2. The value at
  L=100 (~0.49) matches test 5 (c) (0.497) and test 7 (0.446 at L=100, 0.27 at L=10).
  A longer untrained prefix disturbs the receiver more, up to about 50 tokens.
- **`wlmT_wemb_heatmap.png`** (768,894 bytes, 1200×1050) — the 1024×1024 matrix
  W_LMᵀ·W_emb for the pretrained, tied Qwen3-0.6B weight, i.e. W1ᵀW1: the matrix both
  LM_head and Embedding2 are initialized from. It is dominated by a thin positive
  diagonal, and the off-diagonal is near zero, so the hidden dimensions are close to
  orthogonal. The colour scale (±~680) is set by a few outlier dimensions: a bright
  block in roughly the first ~60 dims, and faint stripes near dims ~105, ~430 and ~615.
  It describes the initialization only and says nothing directly about the training
  dynamics.

## Recommended next steps (prioritized, evidence-backed only)

1. **Judge progress with a fixed held-out paired eval, not the per-step training
   loss.** The real 9% gain from 10k steps was 4× smaller than the batch-to-batch
   spread (test 5), so the per-step log could not show it. `diagnostics/_common.eval_loss`
   plus a fixed batch list (as in test 6) can be reused.
2. **Increase the encoder's trainable depth.** With the last 3 layers trainable, the
   held-out gain in ~57 steps was 3× the gain with 1 layer (−0.269 vs −0.093) and
   improved all 8 held-out batches (test 6). Confirm with a longer run before
   committing to it.
3. **Re-establish the baseline curve for the current configuration.** The only long
   run used the older Embedding2-only setup. No long run exists yet with LM_head and
   the last layer trainable, so "flat for 17k steps" has not been shown for the
   current code.
4. **Do not truncate backprop through the rollout.** 97% of Embedding2's gradient comes
   through the recurrence (test 7), so detaching the feedback would throw most of the
   training signal away.
5. **Do not raise the LR to 1e-3 as-is.** It destabilized training on a fixed batch
   (test 4, run B), and BPTT amplifies early-token gradients by up to 12× (test 7). If
   a higher LR is wanted, test it together with gradient-norm clipping (untested here).
6. **(Low priority) Set `student_encoder.config.tie_word_embeddings = False` after
   untying.** This removes the latent re-tie risk shown in test 3. It does not
   explain the flat loss.
