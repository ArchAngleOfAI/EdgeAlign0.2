# Training stability vs. soft-prompt length, LR and gradient clipping (2026-09-29)

Follow-up to `report.md` (tests 4 and 7). Script: `diagnostics/stability_vs_length.py`,
runner: `diagnostics/run_stability.sh`. Per-step logs are in
`diagnostics/results/stability_L{L}_lr{LR}_clip{on|off}.log`; each log ends with a
`SUMMARY_JSON` line containing every number below. No existing file was modified.

**Setup.** Every run starts from a fresh init (seed 0) and trains 30 AdamW steps on
**one fixed batch**: the first batch of the stream, the same batch as test 4 run B.
Batch size is 4, prompt length 128, fp32. The trainable set is Embedding2 + LM_head
+ the last decoder layer, with full backprop through the rollout. The loss at step
*k* is measured before update *k*, so step 1 is the untrained loss. The gradient norm
is the total L2 norm over all trainable parameters, before clipping. "Clip on" means
`clip_grad_norm_(params, max_norm=1.0)`. Held-out loss is measured on stream batches
1001–1008 (32 prompts), before and after training, at the run's own L.
`_common.eval_loss` is fixed at 100 soft tokens, so the script has a local copy of it
that takes L. Runs were on one A100 (GPU 5). GPU 0 raised a CUDA hardware error
("invalid access of peer GPU memory … or a hardware error") during a smoke test, so
it was not used.

**Skipped runs.** All four **L=100** runs of the grid were cancelled at the user's
request before they started. The only L=100 data point is test 4 run B from
`report.md`: same batch, same init, LR 1e-3, no clip, but only **20** steps and no
gradient-norm or held-out measurements. It appears below as a reference row.

## Summary

(1) **Shorter prompts are more stable at LR 1e-3, but L=16 is not fully stable.**
At L=100, the loss fell 0.446 → 0.139 and then jumped +0.564 in one step, ending 40%
*above* its starting value (test 4 run B). At L=16 the pattern is the same but
smaller: the loss fell to 0.107, jumped +0.184, and ended 23% below its start. At
L=8 no step rose by more than 0.1: the largest rise was +0.059, and the loss ended
21% below its start. (2) **Clipping at max_norm=1.0 did not remove the instability
where it appeared.** At L=16, LR 1e-3, the clipped run still had a +0.159 jump. At
L=8 there was no >0.1 instability to remove in either run. At L=100 clipping was not
tested. (3) **Clipping did not slow learning on the training batch at LR 5e-5.** The
clipped runs ended lower than the unclipped ones, at both L=8 (0.082 vs 0.099) and
L=16 (0.127 vs 0.233). The held-out results are mixed. One further observation: every
LR 1e-3 run shows a large mean held-out improvement, but only 4 of 8 held-out batches
improved in each. After training, the loss is nearly the same on every held-out batch
(~0.16–0.32). The two hard batches fall from ~1.0–1.3, and most easy batches get
slightly worse.

## Summary table

| L | LR | clip | loss step 1 | min (step) | final (step 30) | final vs step 1 | biggest rise (step) | rises >0.1 | final > min+0.1 | pre-clip grad norm mean / max / CV | NaN/inf | ‖Δθ‖/‖θ‖ | held-out before → after | paired diff | improved /8 | s/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 5e-5 | off | 0.2347 | 0.0988 (30) | 0.0988 | -58% | +0.0048 (6) | 0 | no | 18.3 / 260.9 / 2.62 | no | 4.95e-03 | 0.4065 → 0.3238 | -0.0827 | 5 | 1.5 |
| 8 | 5e-5 | on | 0.2347 | 0.0779 (26) | 0.0816 | -65% | +0.0045 (28) | 0 | no | 3.9 / 42.0 / 2.01 | no | 4.25e-03 | 0.4065 → 0.3758 | -0.0307 | 7 | 1.4 |
| 8 | 1e-3 | off | 0.2347 | 0.1853 (30) | 0.1853 | -21% | +0.0591 (3) | 0 | no | 9.8 / 256.0 / 4.66 | no | 1.02e-01 | 0.4065 → 0.2047 | -0.2017 | 4 | 1.2 |
| 8 | 1e-3 | on | 0.2347 | 0.0917 (4) | 0.1503 | -36% | +0.0493 (5) | 0 | no | 8.8 / 234.2 / 4.75 | no | 5.03e-02 | 0.4065 → 0.1911 | -0.2154 | 4 | 1.2 |
| 16 | 5e-5 | off | 0.3248 | 0.1978 (21) | 0.2329 | -28% | +0.0608 (5) | 0 | no | 82.4 / 1576.7 / 3.42 | no | 4.96e-03 | 0.4755 → 0.4840 | +0.0086 | 5 | 2.8 |
| 16 | 5e-5 | on | 0.3248 | 0.1230 (28) | 0.1274 | -61% | +0.0448 (6) | 0 | no | 423.9 / 9650.2 / 4.07 | no | 3.12e-03 | 0.4755 → 0.4216 | -0.0539 | 4 | 2.5 |
| 16 | 1e-3 | off | 0.3248 | 0.1065 (3) | 0.2485 | -23% | +0.1838 (4) | 1 | yes | 4.4 / 85.0 / 3.55 | no | 7.61e-02 | 0.4755 → 0.2852 | -0.1902 | 4 | 2.5 |
| 16 | 1e-3 | on | 0.3248 | 0.1285 (2) | 0.2248 | -31% | +0.1593 (3) | 1 | no (0.096) | 1.4 / 22.9 / 2.99 | no | 4.87e-02 | 0.4755 → 0.2771 | -0.1984 | 4 | 2.6 |
| 100 | 5e-5 | off | skipped (cancelled at user request) | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100 | 5e-5 | on | skipped (cancelled at user request) | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100 | 1e-3 | off | skipped (cancelled at user request) | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100 | 1e-3 | on | skipped (cancelled at user request) | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100 — test 4 run B, **20 steps**, reference | 1e-3 | off | 0.4462 | 0.1390 (3) | 0.6235 (step 20) | +40% | +0.5635 (5) | 1 | yes | not recorded | loss finite; grads not recorded | 8.19e-02 | not measured | — | — | not recorded |

The untrained loss differs by length (0.235 at L=8, 0.325 at L=16, 0.446 at L=100 on
this batch), so the comparisons below use each run's change relative to its own
step 1. "Final vs step 1" is (final/step1 − 1).

## Q1 — Are L=8 and L=16 more stable than L=100 at LR 1e-3?

Yes, and the effect grows with length, but L=16 still shows the same failure pattern
on a smaller scale. All three runs below have clipping off.

- **L=100** (test 4 run B, 20 steps): 0.446 → 0.172 → **0.139** (step 3) → 0.145 →
  **0.709** (step 5). One rise of +0.564, then a slow decline to 0.624 at step 20,
  which is **40% above** the step-1 loss.
- **L=16:** 0.325 → 0.129 → **0.107** (step 3) → 0.290 → **0.350** (step 5). One
  rise of +0.184 (step 4), then a steady decline to 0.249 at step 30, which is
  **23% below** step 1. The final loss is 0.142 above the minimum.
- **L=8:** 0.235 → **0.187** (step 2) → 0.246 (step 3). The largest rise is +0.059,
  and no rise exceeds 0.1. The loss then declines steadily to 0.185 at step 30
  (**21% below** step 1). Here the minimum is the final step.

L=16 and L=100 share the same shape: a fast drop over 2–3 steps, one jump, then a slow
monotone decline with small gradients. From step 7 on, the pre-clip gradient norm stays
at ≤0.6 for L=8 and ≤4.0 for L=16. The jump shrinks with length (+0.564 →
+0.184 → +0.059), but at 30 steps the LR 1e-3 runs end above where LR 5e-5 ends at the
same length (L=8: 0.185 vs 0.099; L=16: 0.249 vs 0.233).

## Q2 — Does clipping at max_norm=1.0 remove the instability?

- **L=16, LR 1e-3: no.** With clipping the loss went 0.325 → **0.128** (step 2) →
  **0.288** (step 3), a rise of +0.159 versus +0.184 without clipping, and it ended at
  0.225 versus 0.249. The jump happens one step earlier and is 13% smaller, and the
  final loss is 0.096 above the minimum (just under the 0.1 threshold). The jump
  itself is still there.
- **L=8, LR 1e-3: no instability to remove.** Neither run had a rise over 0.1
  (largest: +0.049 clipped, +0.059 unclipped). Clipping changed the path, though. The
  clipped run went lower early (**0.092** at step 4 vs 0.187 at step 2) and then
  drifted up to 0.150 by step 30, finishing 0.059 above its minimum. It ended lower
  than the unclipped run (0.150 vs 0.185).
- **L=100: not tested** (runs cancelled).

At LR 1e-3, clipping was active on only a few steps: pre-clip norm > 1 on 3 of 30
steps at L=16 and 5 of 30 at L=8, almost all among the first five steps. It was
active on the step whose update produced the jump (L=16 clipped: norms 22.9 and 8.9
at steps 1–2, jump at step 3). The jump occurred anyway. This is consistent with how
AdamW works: its early updates are roughly ±LR per element whatever the gradient's
scale, so rescaling the gradient norm does little to the size of those early steps.
That is an explanation consistent with the numbers, not something this grid isolated.

## Q3 — Does clipping slow learning at LR 5e-5?

**Not on the training batch.** Clipping was active on 21/30 steps (L=8) and 30/30
steps (L=16):

- L=8: final loss 0.082 clipped vs 0.099 unclipped; minimum 0.078 vs 0.099.
- L=16: final loss 0.127 clipped vs 0.233 unclipped; minimum 0.123 vs 0.198. The
  unclipped run rose from 0.198 to 0.237 at step 22 and stayed there.

The clipped runs moved the parameters slightly less (‖Δθ‖/‖θ‖ 4.25e-3 vs 4.95e-3 at
L=8, 3.12e-3 vs 4.96e-3 at L=16), yet reached lower training loss.

**Held-out: mixed.** At L=8, the unclipped run has the larger mean held-out gain
(−0.083, 5/8 improved) versus the clipped run (−0.031, 7/8 improved). Almost all of
the unclipped gain comes from one batch: batch 6 went 0.961 → 0.273. At L=16 the
clipped run is better (−0.054 vs +0.009). The unclipped run made batch 8 much worse
(0.210 → 0.661).

The pre-clip gradient norm at LR 5e-5 is large and erratic, especially at L=16: mean
82 / max 1,577 unclipped, and mean 424 / max 9,650 clipped, with CV 3.4–4.1. At LR
1e-3, after the first few steps, it falls to ~0.2–0.5.

## Held-out behaviour at LR 1e-3 (observation)

In all four LR 1e-3 runs, the held-out losses after training sit in a narrow band:
0.17–0.23 (L=8, off), 0.16–0.21 (L=8, on), 0.24–0.32 (L=16, off) and 0.23–0.30
(L=16, on). Before training they ranged from 0.14 to 1.34. The large mean gains
(−0.19 to −0.22) come entirely from the two hard batches (0.96 → ~0.2–0.3 and
1.27–1.34 → ~0.2–0.3). The easy batches mostly get worse, e.g. L=8 clip off: 0.137
→ 0.196, 0.145 → 0.227. That is why only 4/8 batches improve in every LR 1e-3 run.
The mean held-out number alone overstates the benefit of LR 1e-3. This grid does not
show why the held-out losses converge like this.

## Did L=100 / LR 1e-3 / no-clip reproduce test 4 run B?

That run was not executed (cancelled), so it was not reproduced within this grid.
Two indirect checks passed. The step-1 loss is identical across
all four runs at each length (0.234664 at L=8, 0.324847 at L=16), and the step-1 gradient
norm matches to 4 significant figures (16.98 and 22.88). Init and data are deterministic,
apart from last-digit GPU nondeterminism in the gradient. In `report.md`, the same code path reproduced the
training log's step-1 loss exactly (0.446160).

## Caveats

- One seed, one fixed training batch, and 30 steps per run (20 for the L=100
  reference, which also lacks gradient-norm and held-out numbers).
- The held-out set is 8 batches = 32 prompts. Two of them dominate the mean (see
  above).
- No L=100 run in this grid. Every L=100 comparison relies on test 4 run B.
- Fixed-batch training measures optimization stability, not generalization.
- Thresholds (a rise of 0.1, final > min + 0.1) are the ones specified. L=16/LR
  1e-3/clip-on misses the second threshold by 0.004.

## Recommended next steps (only what this evidence supports)

1. **If LR 1e-3 is considered at all, prefer short prompts.** L=8 showed no >0.1 jump
   at 1e-3; L=16 and L=100 did. Even so, with the same clip setting, LR 1e-3 ended with
   higher training loss than LR 5e-5 at both short lengths within 30 steps, so this grid gives no reason to
   move off 5e-5.
2. **Don't rely on `clip_grad_norm_(max_norm=1.0)` to fix the LR 1e-3 jump.** It did
   not remove it at L=16. If stability at higher LR is the goal, the evidence points
   to controlling the early Adam step size (e.g. LR warmup). That is untested here.
3. **Clipping at max_norm=1.0 is safe to keep at LR 5e-5.** It did not slow
   training-batch learning at L=8 or L=16. The held-out effect is inconclusive
   (better at L=16, worse mean at L=8).
4. **Report per-batch held-out numbers, not only the mean.** At LR 1e-3, the mean
   improved by ~0.2 while half the batches got worse.
5. **Run the cancelled L=100 cells if the L=100 comparison matters.** Without them,
   Q2 at L=100 is unanswered and Q1 rests on a 20-step reference run with fewer
   measurements.
