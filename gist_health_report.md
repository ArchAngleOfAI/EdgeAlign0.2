# Gist health check: 3-layer soft-prompt generator (2026-09-30)

Scripts are in `diagnostics_gist/`: `t1_swap.py` … `t8_steer.py`, the shared setup in
`_gist_common.py`, and the runner `run_all.sh`. Raw results (JSON), plots and console logs are
in `diagnostics_gist/results/`. No existing file was modified.

**Setup.** Everything reuses the training code rather than reimplementing it:

- model build and unfreezing: `diagnostics/_common.load_all` → `build_softprompt_generator`,
  with 3 trainable layers, as in `train_3layer.py`
- checkpoint loading: `train_3layer.load_trainable_state`
- rollout: `distill_softprompt.generate_softprompt`
- receiver: `distill_softprompt.receiver_forward`
- loss: the verbatim training KD loss

The model is Qwen3-0.6B in fp32. The gist has L=16 vectors and is placed as a **prefix** before a
128-token prompt, as in training. Every KL is computed on the 128 prompt positions after the gist,
aligned with the teacher exactly as in training. Everything ran on GPU 5.

**Checkpoints.** All three were found.

| name | what it is | held-out loss (32 prompts) |
|---|---|---|
| step0 | fresh init, seed 0 (`load_all`) | 0.3930 |
| step1000 | `step_01000.pt` (step 1000) | 0.0308 |
| best | `best.pt` = **step 2600** (not `latest.pt`) | **0.0105** |

These match the training run's own eval log exactly (0.392998, 0.0308, 0.010494), which confirms
that checkpoint loading and the evaluation path reproduce training.

**Data.** There are two sets of prompts:

- **The 32 held-out prompts** from `train_3layer.heldout_batches`. They come from FineWeb-Edu
  shard `001_00000`, row group 0.
- **256 extra prompts, used in test 2 only.** They are the next 64 batches of 4 from the same
  shard and the same packing. The script checks that the stream's first 8 batches match the 32
  held-out prompts, so the extra prompts are disjoint from them.

**None of these prompts were seen in training.** Training reads only shard `000_00000`
(`fineweb_edu_batches` hard-codes that file), so it never reads shard `001_00000` at all.
`RUN_NOTES_3layer.md` also records the startup check that the held-out source documents share no
id or text hash with any training document the run could reach.

**Seeds.** No step uses unseeded randomness; the data stream is deterministic. The seeds used:

| seed | used for |
|---|---|
| 0 | init (`load_all` calls `torch.manual_seed(0)`) |
| 0, 1, 2 | test 1 derangements (CPU `torch.Generator`) |
| 0, 1, 2 | rand_tok draws |
| 0, 1, 2 | test 5 noise (GPU `torch.Generator`) |
| 0 | `torch.manual_seed(0)` in test 8 |

## Summary

(1) **Yes, the generator has collapsed to an input-independent gist.** By step 1000, and fully by
the best checkpoint, the generator's softmax is one-hot at every rollout step (entropy ≈ 0,
top-1 probability 1.00). Every prompt gets the same 16-step token sequence: `" it"`, `"gle"`,
`"gle"`, then `" it"` ×13. Gists of different prompts have cosine similarity 0.996, and
swapping one prompt's gist for another's changes the loss by less than 1e-7. (2) **The best gist is
not dead in the sense the rules define.** Every naive prefix scores ≥14× worse than it (0.148–3.29
vs 0.0105), and the receiver reacts *more* strongly to noise on the best gist than on real text
(77× at f=0.01). It is also fully steerable when its vectors are optimised directly, and partly
steerable through the generator (+9.6 nats in 50 steps). One signal does point toward invisibility:
the best gist's positions 2–16 receive only 0.35% of the receiver's attention, against 2.6% for
real text and 12.5% for the step0 gist. The receiver uses the gist mainly through its first vector,
which acts as an attention sink. (3) **These properties changed with training.** The collapse
happened between step0 and step1000. Attention to gist positions 2–16 fell at every checkpoint
(12.5% → 2.2% → 0.35%). Sensitivity dropped from step0 to step1000 and then rose again at best, and
generator steerability was higher at best than at step0. So the checkpoints show no consistent
movement toward an invisible gist, except in attention to positions 2–16.

## Test 1: swap test

Teacher always sees plain prompt A; A's gist is replaced by B's (B = a fixed random derangement of
the 32 prompts). Swapped − own is paired per prompt.

| checkpoint | own mean | derangement seed | swapped mean | paired diff (mean ± std) | fraction swapped > own |
|---|---|---|---|---|---|
| step0 | 0.39300 | 0 | 0.41228 | +0.01929 ± 0.813 | 0.500 |
| step0 | | 1 | 0.42386 | +0.03087 ± 0.844 | 0.438 |
| step0 | | 2 | 0.38303 | −0.00997 ± 0.777 | 0.500 |
| step1000 | 0.030775 | 0 | 0.031990 | +0.001215 ± 0.0135 | 0.5625 |
| step1000 | | 1 | 0.031438 | +0.000664 ± 0.0125 | 0.5625 |
| step1000 | | 2 | 0.031562 | +0.000787 ± 0.0125 | 0.375 |
| best | 0.010494 | 0 | 0.010494 | −0.0000000 ± 4.2e-8 | 0.531 |
| best | | 1 | 0.010494 | −0.0000000 ± 4.0e-8 | 0.469 |
| best | | 2 | 0.010494 | −0.0000000 ± 4.3e-8 | 0.438 |

At step0 swapping moves individual prompts a lot (std ≈ 0.8), but not in a consistent direction:
the mean difference is small and has both signs. At step1000, swapping slightly hurts on average
(+0.0007 to +0.0012, positive for all 3 seeds). At best, swapping makes no difference: the
per-prompt differences are ~4e-8, which is float noise, because the 32 gists are identical.

## Test 2: gist variation across prompts (N = 288: 32 held-out + 256 extra)

"Cross-prompt cos" is the mean cosine between gists of different prompts at that position.
"std/mean" is ‖std over prompts‖ / ‖mean over prompts‖.

| position | step0 cos | step0 std/mean | step1000 cos | step1000 std/mean | best cos | best std/mean |
|---|---|---|---|---|---|---|
| 1 | 0.110 | 3.30 | 0.939 | 0.286 | 0.987 | 0.128 |
| 2 | 0.145 | 2.71 | 0.962 | 0.181 | 1.000 | 0.000 |
| 3 | 0.163 | 2.49 | 0.925 | 0.258 | 0.987 | 0.103 |
| 4 | 0.152 | 2.69 | 0.981 | 0.159 | 0.969 | 0.203 |
| 5–16 | 0.160–0.210 | 2.26–2.65 | 1.000 | 0.000 | 1.000 | 0.000 |
| **mean over 16** | **0.173** | **2.54** | **0.988** | **0.055** | **0.996** | **0.027** |

Mean cosine between different positions *within* one gist: step0 0.214 (min 0.053, max 0.565),
step1000 0.784 (0.696–1.000), best 0.784 (0.702–0.786). The 16 vectors are not all the same
vector. They are two distinct vectors: `" it"` at 14 positions and `"gle"` at 2, which explains
the ~0.78 value. At best, positions 2 and 5–16 are bit-identical across all 288 prompts. Positions
1, 3 and 4 differ for a small subset of the 256 extra prompts; all 32 held-out prompts are identical
there (test 3).

## Test 3: inside the rollout (32 held-out prompts)

| step | step0 entropy mean (min–max) | step0 top-1 p mean | step0 most common top-1 | step1000 entropy mean / max | step1000 top-1 (prompts) | best entropy max | best top-1 (prompts) |
|---|---|---|---|---|---|---|---|
| 1 | 2.46 (…–6.4) | 0.482 | `,` (4/32) | 0.000 / 1.0e-5 | `" it"` (28) | 5.1e-7 | `" it"` (32) |
| 2 | 2.71 (…–5.3) | 0.420 | `" the"` (5) | 0.000 / 9.6e-3 | `"gle"` (30) | 5.3e-17 | `"gle"` (32) |
| 3 | 3.03 (…–6.0) | 0.425 | `" the"` (4) | 0.000 / 1.0e-5 | `"gle"` (30) | 3.9e-18 | `"gle"` (32) |
| 4 | 3.03 (…–6.9) | 0.427 | `" the"` (5) | 0.000 / 9.5e-6 | `" it"` (31) | 6.2e-5 | `" it"` (32) |
| 5–16 | 2.61–3.37 (max ≤ 7.5) | 0.366–0.503 | `" the"`, `" was"`, `" and"`, … (3–7 each) | 0.000 / ≤1.3e-5 | `" it"` (32) | ≤4.9e-6 | `" it"` (32) |

At step1000 and best, the minimum top-1 probability is ≥ 0.9988 at every step (1.0000 at best).
**Distinct top-1 tokens per prompt across the 16 steps:** step0 mean 12.1 (min 6, max 16);
step1000 and best exactly 2 for every prompt.

**Nearest receiver tokens (cosine) for 5 example prompts.** At step0 the soft vectors track each
prompt's likely continuation. For example, for "The Heilwood Company store was a product of
necessity…" the nearest tokens per position are `not, a, as, a, partnership, ., The, store, was,
located, by, the, town, ,, a, owner`. For another prompt, position 1 is nearest `" of"` at cos
0.978. At step1000 and best, all 5 prompts give the identical list: position 1 → `" it"` (cos 0.872
at step1000, 0.799 at best), then `它`, `" It"`, `"It"`, `"\tit"`; positions 2–3 → `"gle"`;
positions 4–16 → `" it"`. The full top-5 lists are in `results/t3_rollout.json`.

## Test 4: naive-prefix baselines (32 held-out prompts, training KD loss)

text16 = the 16 tokens `You | are | a | careful | coding | assistant | . | Read | the | following
| text | and | continue | it | accurately | .` (ids 2610 … 13). The sentence is exactly 16 Qwen
tokens, so it needed no trimming or extension. pad16 = 16 × `<|endoftext|>` (id 151643; the
tokenizer has no BOS). rand_tok draws uniformly from the ordinary-token ids [0, 151643). The drawn
ids are in `results/t4_baselines.json`.

| prefix | mean | per-batch (8 × 4 prompts) |
|---|---|---|
| **gist best (step 2600)** | **0.01049** | 0.0080, 0.0155, 0.0103, 0.0081, 0.0088, 0.0105, 0.0119, 0.0108 |
| gist step1000 | 0.03077 | 0.0196, 0.0411, 0.0356, 0.0187, 0.0272, 0.0372, 0.0482, 0.0187 |
| rand_tok seed 2 | 0.14846 | 0.1485, 0.1580, 0.1681, 0.1286, 0.1791, 0.1477, 0.1507, 0.1070 |
| text16 | 0.16466 | 0.1648, 0.1655, 0.1978, 0.1474, 0.1880, 0.1527, 0.1384, 0.1628 |
| rand_tok seed 0 | 0.18687 | 0.2035, 0.2226, 0.2008, 0.1583, 0.1731, 0.1891, 0.2062, 0.1414 |
| rand_tok seed 1 | 0.19615 | 0.1917, 0.2059, 0.2171, 0.1701, 0.2041, 0.1892, 0.2359, 0.1553 |
| rand_tok (mean of 3) | 0.17716 | — |
| gist step0 | 0.39300 | 0.1920, 0.8578, 0.5906, 0.2354, 0.2255, 0.6223, 0.2461, 0.1744 |
| pad16 | 0.39363 | 0.3419, 0.3818, 0.3877, 0.4216, 0.4907, 0.4127, 0.3399, 0.3728 |
| zeros | 3.28501 | 3.4336, 3.5281, 4.0899, 3.0689, 2.8524, 3.2462, 2.4790, 3.5820 |

The best gist beats every naive prefix on every batch, by 14× over the best naive prefix
(rand_tok seed 2) and 16× over text16.

## Test 5: sensitivity to noise

KL(receiver with clean prefix ‖ receiver with noisy prefix) on the prompt positions, 32 prompts.
Each of the 16 vectors gets Gaussian noise with ‖noise‖ = f·‖vector‖. Values are mean ± std over
3 noise seeds. Plot: `results/sensitivity.png`.

| prefix | f = 0.01 | f = 0.05 | f = 0.1 | f = 0.5 |
|---|---|---|---|---|
| gist step0 | 4.71e-4 ± 1.2e-4 | 1.15e-2 ± 1.5e-3 | 3.50e-2 ± 3.2e-3 | 1.40e-1 ± 2.4e-2 |
| gist step1000 | 7.38e-5 ± 2.7e-6 | 1.75e-3 ± 6.0e-6 | 5.62e-3 ± 3.1e-4 | 2.85e-2 ± 2.1e-3 |
| gist best | 7.63e-4 ± 8.0e-5 | 1.22e-2 ± 1.2e-3 | 1.89e-2 ± 8.0e-4 | 4.54e-2 ± 3.2e-3 |
| text16 | 9.90e-6 ± 9.8e-7 | 2.37e-4 ± 2.3e-5 | 9.11e-4 ± 8.8e-5 | 1.81e-2 ± 2.6e-3 |

Compared with text16, the best gist is 77×, 51×, 21× and 2.5× more sensitive at f = 0.01, 0.05,
0.1 and 0.5. Step1000 is 7×, 7×, 6× and 1.6× more sensitive. By checkpoint, sensitivity is not
monotonic: step0 → step1000 falls about 6× at f=0.01, then step1000 → best rises about 10×.
Test 7 rules out a size effect for the trained gists, whose vectors are token-sized. The step0
gist's vectors are about half that size, so its absolute noise is smaller at the same f.

## Test 6: attention on the gist

A separate receiver copy was loaded with `attn_implementation="eager"`. Its weights are identical
to the normal receiver (checked), and its logits match `receiver_forward` to within 3e-4 (4.8e-3
for zeros). The values below are averaged over heads, over the 128 prompt query positions and over
the 32 prompts. Plot: `results/attention.png`.

| prefix | attention on all 16 prefix positions (mean over 28 layers) | on position 0 alone | on positions 1–15 (difference) | layers 0–2, all 16 positions |
|---|---|---|---|---|
| gist step0 | 0.565 | 0.440 | **0.125** | 0.072, 0.151, 0.164 |
| gist step1000 | 0.553 | 0.531 | 0.022 | 0.005, 0.008, 0.031 |
| gist best | 0.552 | 0.548 | **0.0035** | 0.001, 0.004, 0.016 |
| text16 | 0.559 | 0.534 | 0.026 | 0.087, 0.154, 0.121 |
| zeros | 0.025 | 0.002 | 0.023 | 0.009, 0.009, 0.005 |

From layer 3 on, all non-zero prefixes draw 37–81% of the attention per layer, almost all of it on
position 0, the usual attention sink. With zeros the sink disappears (2.5% in total), which fits
zeros being by far the worst prefix in test 4. What the trained gists change is how much attention
the other positions get: positions 1–15 receive 0.35% at best, against 2.6% for text16 and 12.5%
for step0. In layers 0–2 the trained gists also get almost no attention.

## Test 7: vector norms

The receiver's `embed_tokens` row norms: mean 0.925, 5th percentile 0.714, 95th percentile 1.099.
Ordinary tokens only: 0.926, 0.720, 1.099.

| checkpoint | mean | min | max | by position (mean over prompts) |
|---|---|---|---|---|
| step0 | 0.489 | 0.169 | 1.160 | 0.45–0.54 at every position |
| step1000 | 0.895 | 0.867 | 1.088 | 0.87 at positions 1 and 4–16, 1.07 at positions 2–3 |
| best | 0.916 | 0.889 | 1.103 | 0.89 at positions 1 and 4–16, 1.10 at positions 2–3 |

The trained gist vectors are token-sized, inside the embedding 5–95% band. The step0 gist vectors
are about half that size, because each is a convex mixture of many embedding rows. At step1000 and
best, the min and max per position are equal across the 32 prompts: every prompt gets the same
vectors.

## Test 8: steerability probe (target `"{"`, token id 90, at the last prompt position)

The table reports the mean log p("{") over the 32 held-out prompts after k updates; the gain is
step 50 − step 0. Held-out KD is the training KD loss on the same 32 prompts with the steered
prefix. **(a) Direct:** one (16, H) prefix per prompt, initialised from the start, optimised with
Adam for 50 steps. **(b) Generator:** the training trainable set, optimised with AdamW at LR 1e-4
with clipping at 1.0.

| start | method | step 0 | 10 | 25 | 50 | gain | held-out KD before → at step 50 |
|---|---|---|---|---|---|---|---|
| gist step0 | (a) LR 1e-3 | −17.619 | −6.934 | −0.050 | −0.0012 | +17.62 | 0.3930 → 1.798 |
| gist step0 | (a) LR 1e-2 | −17.619 | −1.760 | −0.085 | −0.0008 | +17.62 | 0.3930 → 1.973 |
| gist step1000 | (a) LR 1e-3 | −17.275 | −6.644 | −0.082 | −0.0012 | +17.27 | 0.0308 → 1.126 |
| gist step1000 | (a) LR 1e-2 | −17.275 | −4.275 | −0.049 | −0.0007 | +17.27 | 0.0308 → 1.898 |
| gist best | (a) LR 1e-3 | −17.355 | **−10.956** | **−1.238** | −0.0090 | +17.35 | 0.0105 → 1.043 |
| gist best | (a) LR 1e-2 | −17.355 | −6.719 | −0.967 | −0.0207 | +17.33 | 0.0105 → 1.709 |
| text16 | (a) LR 1e-3 | −17.060 | −8.654 | −0.597 | −0.0041 | +17.06 | 0.1647 → 2.882 |
| text16 | (a) LR 1e-2 | −17.060 | −4.770 | −0.451 | −0.0010 | +17.06 | 0.1647 → 3.421 |
| zeros | (a) LR 1e-3 / 1e-2 | −20.492 | NaN | NaN | NaN | — | 3.285 → NaN |
| step0 checkpoint | (b) generator | −17.619 | −17.007 | −16.649 | −16.690 | **+0.93** | 0.3930 → 0.1633 |
| best checkpoint | (b) generator | −17.355 | −16.355 | −14.131 | −7.775 | **+9.58** | 0.0105 → 0.1631 |

- **(a) Every non-zero start is fully steerable.** In 50 steps, log p("{") goes from about −17 to
  about 0, i.e. near-certain "{". The best gist is the slowest at both LRs; at step 25 with LR 1e-3
  it is at −1.24, against −0.05 to −0.60 for the others. Steering disturbs everything else heavily:
  held-out KD rises to 1.0–3.4. The best gist disturbs least at LR 1e-3.
- **Zeros NaN.** At an exactly-zero prefix the gradient is already NaN at all 16 positions, before
  any update; this was checked separately. With a 1e-8 start instead, the gradient is finite but
  about 5.6e8 in size. Gradient-based steering is simply not defined at the zeros start, so this is
  a property of the starting point, not a steerability result.
- **(b) Through the generator, the best checkpoint is about 10× more steerable than step0**
  (+9.58 vs +0.93 nats in 50 steps). The step0 generator's pre-clip gradient norm averaged 10,097
  (max 166,103), against 800 (max 3,086) for best. Both averages are far above the 1.0 clip
  threshold. Only the mean and max were logged, so the fraction of clipped steps is not known. Steering from step0 also *lowered* its held-out KD (0.393 → 0.163), while
  steering from best raised it (0.0105 → 0.163).

## Interpretation (applying the given rules)

| rule | condition | evidence | met? |
|---|---|---|---|
| healthy: prompt-specific and steerable | swap changes the loss, gists differ across prompts, sensitivity similar to text16 | best: swap effect ~4e-8, cross-prompt cos 0.996 | **no** |
| **collapsed but alive** | gists nearly identical across prompts, swap has no effect, sensitivity fine | best: cos 0.996, swap Δ ≈ 0, sensitivity 2.5–77× *above* text16, directly steerable to about 0 | **yes, for best (and essentially for step1000)** |
| dead prefix | naive prefixes reach a loss similar to best AND sensitivity is low vs text16 | naive ≥ 0.148 vs 0.0105 (14× worse); sensitivity higher than text16 | **no** |
| training drives the gist toward invisibility | sensitivity or steerability decreases step0 → step1000 → best | sensitivity: 4.7e-4 → 7.4e-5 → 7.6e-4 (not monotonic); direct steerability: best slowest at step 10–25 but reaches about 0; generator steerability: best 10× *higher* than step0; attention to positions 1–15: 0.125 → 0.022 → 0.0035 (monotonic decrease) | **mixed**: only attention to gist positions 2–16 decreases monotonically |

**The evidence matches "collapsed but alive."** From step1000 on, the generator emits one fixed,
prompt-independent sequence of token-like vectors (`it gle gle it …`). Its loss is 14–16× better
than naive or real-text prefixes, so the prefix still matters to the receiver: it is a learned
constant "low-interference" prefix, not an invisible one. The receiver still responds strongly to
changes in it. The one sign of a drift toward invisibility is that attention reaches the gist
almost only through its first position, and that share shrinks at every checkpoint.

## Caveats

- **Sample sizes.** 32 held-out prompts, 288 in test 2 only; 3 derangement seeds, 3 noise seeds and
  3 random-token draws; one init seed.
- **Test 8 uses one target token** (`"{"`, id 90) with 50 steps per run. Part (a) optimises a
  separate prefix per prompt, and part (b) ran at one LR. The zeros start gives NaN gradients by
  construction.
- **One real-text reference** (text16). Sensitivity comparisons with it depend on that sentence.
- **Noise is scaled to each vector's own size.** The trained gists are token-sized, but the step0
  gist is about half-size (test 7), so step0 sensitivity is measured at smaller absolute noise.
- **Attention is averaged over heads and query positions.** The eager copy's logits differ from the
  normal receiver by up to 3e-4 (4.8e-3 for zeros).
- **Not tested here:** the late-run instability (after step ~2650), and whether a gist can be
  prompt-specific under a different objective.

## Recommended next steps (only what this evidence supports)

1. **Add prompt-specificity checks to training monitoring.** The held-out KD loss kept improving
   (0.031 → 0.0105) after the gist had already collapsed by step 1000. KD loss alone cannot detect
   this failure. The swap test (test 1), cross-prompt cosine (test 2) and rollout entropy (test 3)
   each detect it cheaply.
2. **If prompt-specific gists are the goal, the current objective and setup did not produce
   them.** The best checkpoint is functionally a constant prefix. Any change to push toward
   prompt-specific gists would be a new experiment; this evidence does not say which change would
   work.
3. **Check whether a single learned constant prefix matches the best gist's 0.0105.** The best
   generator already outputs one constant prefix for every prompt, so this would show directly
   whether the generator adds anything beyond a fixed learned prefix.
4. **The evidence here is mixed on "invisibility."** Sensitivity and steerability do not decline,
   but attention to gist positions 2–16 does. To decide whether that matters, a follow-up could
   test whether the receiver uses positions 2–16 at all. This report did not measure that
   separately (for example, by ablating those positions).
