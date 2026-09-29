"""Follow-up to report.md: training stability vs. soft-prompt length, LR and grad clipping.

Grid (each run from a fresh init, seed 0 via _common.load_all):
  L in {8, 16, 100}  x  LR in {5e-5, 1e-3}  x  clip in {off, on (clip_grad_norm_, max_norm=1.0)}
Each run trains --steps (30) optimizer steps on ONE FIXED batch (the first batch
of the stream -- the batch of test 4 run B), batch size 4, prompt length 128,
AdamW, fp32, the current trainable set (Embedding2 + LM_head + last 1 decoder
layer), full backprop through the rollout. The loss at step k is measured
BEFORE update k, so step 1 is the untrained loss (same convention as test 4).

Per step it logs kd_loss, the total L2 gradient norm over all trainable params
BEFORE clipping (clip_grad_norm_'s return value when clipping, the same norm via
torch.nn.utils.get_total_norm otherwise) and a NaN/inf flag. Each run's log goes
to diagnostics/results/stability_L{L}_lr{LR}_clip{on|off}.log and ends with a
`SUMMARY_JSON {...}` line holding the run's summary numbers.

Held-out eval: 8 fixed batches (stream batches 1001-1008, as in test 6) scored
before and after training with the run's own L. _common.eval_loss always uses
100 soft tokens, so `eval_loss_at_length` below is the same loop with
num_soft_tokens=L passed through to _common.forward_loss.
"""

import argparse
import json
import math
import os
import statistics

import torch

from _common import (
    REPO_ROOT, Timer, forward_loss, get_device, get_tokenizer, load_all, rel_change, setup_stdout,
    snapshot, take_batches, trainable_params,
)

RESULTS_DIR = os.path.join(REPO_ROOT, "diagnostics", "results")
EVAL_SKIP, EVAL_N = 1000, 8
MAX_NORM = 1.0
SPIKE = 0.1


def lr_tag(lr):
    mantissa, exp = f"{lr:e}".split("e")
    return f"{float(mantissa):g}e{int(exp)}"  # 5e-05 -> "5e-5", 0.001 -> "1e-3"


def run_name(L, lr, clip):
    return f"stability_L{L}_lr{lr_tag(lr)}_clip{'on' if clip else 'off'}"


@torch.no_grad()
def eval_loss_at_length(encoder, embedding2, teacher, receiver, batches, device, num_soft_tokens):
    vals = [forward_loss(encoder, embedding2, teacher, receiver, b.to(device), num_soft_tokens).item()
            for b in batches]
    return sum(vals) / len(vals), vals


def run(L, lr, clip, steps, train_batch, eval_batches, device):
    name = run_name(L, lr, clip)
    path = os.path.join(RESULTS_DIR, name + ".log")
    lines = []

    def log(msg):
        print(f"[{name}] {msg}")
        lines.append(msg)

    encoder, embedding2, _, teacher, receiver = load_all(device)
    params = trainable_params(encoder, embedding2)
    log(f"L={L} lr={lr:g} clip={'max_norm=%.1f' % MAX_NORM if clip else 'off'} steps={steps} "
        f"batch={tuple(train_batch.shape)} trainable={sum(p.numel() for p in params):,}")

    eval_before, per_before = eval_loss_at_length(encoder, embedding2, teacher, receiver,
                                                   eval_batches, device, L)
    log(f"held-out BEFORE: {eval_before:.6f}  per-batch={[round(v, 6) for v in per_before]}")

    snap = snapshot(params)
    opt = torch.optim.AdamW(params, lr=lr)
    ids = train_batch.to(device)
    losses, gnorms, nonfinite = [], [], []
    torch.cuda.synchronize()
    t = Timer()
    for step in range(1, steps + 1):
        opt.zero_grad()
        loss = forward_loss(encoder, embedding2, teacher, receiver, ids, L)
        loss.backward()
        if clip:
            gn = torch.nn.utils.clip_grad_norm_(params, max_norm=MAX_NORM)
        else:
            gn = torch.nn.utils.get_total_norm([p.grad for p in params])
        opt.step()
        lv, gv = loss.item(), gn.item()
        bad = not (math.isfinite(lv) and math.isfinite(gv))
        losses.append(lv)
        gnorms.append(gv)
        nonfinite.append(bad)
        log(f"step {step:2d}  kd_loss={lv:.6f}  grad_norm_preclip={gv:.4e}  nan_or_inf={bad}")
    torch.cuda.synchronize()
    sec_per_step = t() / steps
    rc, _ = rel_change(params, snap)

    eval_after, per_after = eval_loss_at_length(encoder, embedding2, teacher, receiver,
                                                 eval_batches, device, L)
    log(f"held-out AFTER:  {eval_after:.6f}  per-batch={[round(v, 6) for v in per_after]}")

    rises = [b - a for a, b in zip(losses, losses[1:])]
    min_loss = min(losses)
    gmean = statistics.mean(gnorms)
    diffs = [a - b for a, b in zip(per_after, per_before)]
    summary = {
        "run": name, "L": L, "lr": lr, "clip": clip, "steps": steps,
        "loss_step1": losses[0], "loss_min": min_loss, "loss_min_step": losses.index(min_loss) + 1,
        "loss_final": losses[-1],
        "max_rise": max(rises), "max_rise_step": rises.index(max(rises)) + 2,
        "n_rises_gt_0.1": sum(r > SPIKE for r in rises),
        "final_gt_min_plus_0.1": losses[-1] > min_loss + SPIKE,
        "gnorm_mean": gmean, "gnorm_max": max(gnorms),
        "gnorm_cv": statistics.pstdev(gnorms) / gmean if gmean else float("nan"),
        "any_nan_or_inf": any(nonfinite),
        "rel_param_change": rc,
        "heldout_before": eval_before, "heldout_after": eval_after,
        "heldout_paired_diff": statistics.mean(diffs), "heldout_improved": sum(d < 0 for d in diffs),
        "sec_per_step": sec_per_step,
        "losses": losses, "gnorms": gnorms,
    }
    log("SUMMARY_JSON " + json.dumps(summary))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    del encoder, embedding2, teacher, receiver, opt, params, snap
    torch.cuda.empty_cache()
    return summary


def main():
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[8, 16, 100])
    ap.add_argument("--lrs", type=float, nargs="+", default=[5e-5, 1e-3])
    ap.add_argument("--clip", choices=["off", "on"], nargs="+", default=["off", "on"])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip runs whose results log already exists.")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = get_device()
    tok = get_tokenizer()
    train_batch = take_batches(tok, 1)[0]
    eval_batches = take_batches(tok, EVAL_N, skip=EVAL_SKIP)

    for L in args.lengths:
        for lr in args.lrs:
            for c in args.clip:
                clip = c == "on"
                if args.skip_existing and os.path.exists(
                        os.path.join(RESULTS_DIR, run_name(L, lr, clip) + ".log")):
                    print(f"skip existing {run_name(L, lr, clip)}")
                    continue
                s = run(L, lr, clip, args.steps, train_batch, eval_batches, device)
                print(f"DONE {s['run']}: step1={s['loss_step1']:.4f} min={s['loss_min']:.4f}"
                      f"@{s['loss_min_step']} final={s['loss_final']:.4f} max_rise={s['max_rise']:+.4f} "
                      f"heldout {s['heldout_before']:.4f}->{s['heldout_after']:.4f} "
                      f"({s['sec_per_step']:.1f}s/step)")


if __name__ == "__main__":
    main()
