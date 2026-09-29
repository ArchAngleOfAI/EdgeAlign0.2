"""Test 4 (Hypothesis A): does a 20x larger LR move the loss at all?

Three 20-step runs, each from a fresh build_softprompt_generator init, same
loop/optimizer/trainable set as distill_softprompt.py (batch_size=4 like the
flat 17k-step run, no grad accumulation):

  A. fresh-batches, lr=1e-3 -- the loop exactly as in training, only LR changed.
  B. fixed-batch,  lr=1e-3 -- the SAME first batch every step.
  C. fixed-batch,  lr=5e-5 -- control at the real training LR.

Why B/C were added: in A every step sees a new batch, so the per-step loss can
never be bit-identical (data noise alone changes it; the flat run's per-step
std is ~0.2). Only with a fixed batch does "loss changes step to step" isolate
the effect of the parameter updates. Loss at step k is measured BEFORE update k,
so step 1 is the untrained loss.
"""

import argparse

import torch

from _common import (
    LR, Timer, forward_loss, get_device, get_tokenizer, load_all, rel_change, setup_stdout,
    snapshot, take_batches, trainable_params,
)

REPORT = (1, 5, 10, 20)


def run(label, lr, batches, device, steps):
    encoder, embedding2, _, teacher, receiver = load_all(device)
    params = trainable_params(encoder, embedding2)
    snap = snapshot(params)
    opt = torch.optim.AdamW(params, lr=lr)
    losses, t = [], Timer()
    for step in range(1, steps + 1):
        opt.zero_grad()
        loss = forward_loss(encoder, embedding2, teacher, receiver, batches[step - 1].to(device))
        loss.backward()
        opt.step()
        losses.append(loss.item())
        print(f"[{label}] step {step:2d}/{steps}  kd_loss={losses[-1]:.6f}  ({t():.0f}s)")
    rc, abs_c = rel_change(params, snap)
    print(f"[{label}] SUMMARY lr={lr:g}: " + "  ".join(f"step{s}={losses[s - 1]:.6f}" for s in REPORT)
          + f"  | distinct loss values: {len(set(losses))}/{steps}"
          + f"  | ||dtheta||/||theta||={rc:.3e}")
    del encoder, embedding2, teacher, receiver, opt
    torch.cuda.empty_cache()
    return losses


def main():
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    device = get_device()
    fresh = take_batches(get_tokenizer(), args.steps)
    fixed = [fresh[0]] * args.steps

    a = run("A fresh lr=1e-3", args.lr, fresh, device, args.steps)
    b = run("B fixed lr=1e-3", args.lr, fixed, device, args.steps)
    c = run("C fixed lr=5e-5", LR, fixed, device, args.steps)

    print("\nFINAL TABLE (kd_loss at step)")
    print(f"{'run':22s}" + "".join(f"{'step ' + str(s):>12s}" for s in REPORT))
    for name, ls in [("A fresh lr=1e-3", a), ("B fixed lr=1e-3", b), ("C fixed lr=5e-5", c)]:
        print(f"{name:22s}" + "".join(f"{ls[s - 1]:12.6f}" for s in REPORT))


if __name__ == "__main__":
    main()
